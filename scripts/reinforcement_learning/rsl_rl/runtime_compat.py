"""Shared RSL-RL 3.x/5.x runtime and checkpoint compatibility.

Checkpoint conversions use a content-addressed temporary cache, leaving trained
artifacts untouched. Inference never loads optimizer state. Cross-format training
resumes retain the iteration and weights but start a fresh optimizer.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
from importlib.metadata import version

from packaging.version import Version
import torch


def rsl_rl_version() -> Version:
    return Version(version("rsl-rl-lib"))


def runner_cfg(agent_cfg) -> dict:
    """Return an independent config accepted by the installed runner."""
    cfg = copy.deepcopy(agent_cfg)
    from isaaclab_rl import rsl_rl as lab_rsl

    migrate = getattr(lab_rsl, "handle_deprecated_rsl_rl_cfg", None)
    if migrate is not None:
        cfg = migrate(cfg, str(rsl_rl_version()))
    result = cfg.to_dict()
    groups = dict(result.get("obs_groups") or {})
    if result.get("class_name", "OnPolicyRunner") == "OnPolicyRunner":
        if rsl_rl_version() >= Version("4"):
            groups.setdefault("actor", groups.pop("policy", ["policy"]))
            groups.setdefault("critic", groups["actor"])
        else:
            groups.setdefault("policy", groups.pop("actor", ["policy"]))
            groups.setdefault("critic", groups["policy"])
    result["obs_groups"] = groups
    return result


def policy_module(runner):
    """Return the inference model across runner API versions."""
    if hasattr(runner.alg, "get_policy"):
        return runner.alg.get_policy()
    for name in ("actor", "policy", "actor_critic"):
        module = getattr(runner.alg, name, None)
        if module is not None:
            return module
    raise TypeError(f"Unsupported RSL-RL algorithm: {type(runner.alg).__name__}")


def policy_observations(obs):
    """Keep grouped observations intact and wrap the Q1 flat actor input.

    Both RSL-RL 3 and 5 inference policies consume grouped observations. Older
    play callers may still supply the flat ``policy`` tensor (92-D for skate).
    """
    if torch.is_tensor(obs):
        from tensordict import TensorDict

        return TensorDict({"policy": obs}, batch_size=obs.shape[:-1])
    return obs


def checkpoint_for_runtime(path: str | Path) -> str:
    """Translate MLP weights and empirical normalizers when crossing v5."""
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    modern = rsl_rl_version() >= Version("5")
    if modern == ("actor_state_dict" in payload) or "student_state_dict" in payload:
        return str(path)
    converted = dict(payload)
    if modern:
        if "model_state_dict" not in payload:
            return str(path)
        actor, critic = {}, {}
        for key, value in payload["model_state_dict"].items():
            if key in {"std", "log_std"}:
                actor["distribution." + ("std_param" if key == "std" else "log_std_param")] = value
            elif key.startswith("actor_obs_normalizer."):
                actor["obs_normalizer." + key.removeprefix("actor_obs_normalizer.")] = value
            elif key.startswith("critic_obs_normalizer."):
                critic["obs_normalizer." + key.removeprefix("critic_obs_normalizer.")] = value
            elif key.startswith("actor."):
                actor["mlp." + key.removeprefix("actor.")] = value
            elif key.startswith("critic."):
                critic["mlp." + key.removeprefix("critic.")] = value
            else:
                raise ValueError(f"Unsupported legacy checkpoint parameter {key!r} in {path}; only MLP policies are supported")
        converted.pop("model_state_dict")
        converted.update(actor_state_dict=actor, critic_state_dict=critic)
        target = "5"
    else:
        if "actor_state_dict" not in payload:
            return str(path)
        state = {}
        for name in ("actor", "critic"):
            for key, value in payload[f"{name}_state_dict"].items():
                if name == "actor" and key in {"distribution.std_param", "distribution.log_std_param"}:
                    state["std" if key.endswith(".std_param") else "log_std"] = value
                elif key.startswith("obs_normalizer."):
                    state[f"{name}_obs_normalizer." + key.removeprefix("obs_normalizer.")] = value
                elif key.startswith("mlp."):
                    state[f"{name}." + key.removeprefix("mlp.")] = value
                else:
                    raise ValueError(f"Unsupported v5 checkpoint parameter {key!r} in {path}; only MLP policies are supported")
            converted.pop(f"{name}_state_dict")
        converted["model_state_dict"] = state
        target = "3"
    converted.setdefault("infos", None)
    converted.setdefault("iter", 0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = Path(tempfile.gettempdir()) / "q1-rsl-rl-checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{digest}_rsl{target}.pt"
    # Always replace atomically, including when training hot-reloads a checkpoint.
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(converted, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[INFO] Converted checkpoint for RSL-RL {target}: {path.name}")
    return str(destination)


def load_checkpoint(runner, path: str | Path, *, load_optimizer: bool = False, load_iteration: bool = True):
    """Load policy weights with the correct runner API and checkpoint format."""
    compatible = checkpoint_for_runtime(path)
    if Path(compatible).resolve() != Path(path).expanduser().resolve() and load_optimizer:
        print("[INFO] Checkpoint format changed; optimizer state starts fresh.")
        load_optimizer = False
    parameters = inspect.signature(runner.load).parameters
    if "load_cfg" in parameters:
        result = runner.load(compatible, load_cfg={"actor": True, "critic": True, "student": True,
            "teacher": True, "optimizer": load_optimizer, "iteration": load_iteration,
            "rnd": load_optimizer}, map_location=runner.device)
    else:
        result = runner.load(compatible, load_optimizer=load_optimizer, map_location=runner.device)
    if not load_iteration:
        runner.current_learning_iteration = 0
    return result


def reset_noise_std(runner, std: float) -> None:
    """Set exploration scale on legacy ActorCritic and modern distributions."""
    if std <= 0:
        raise ValueError("Noise standard deviation must be positive")
    model = policy_module(runner)
    distribution = getattr(model, "distribution", None)
    with torch.no_grad():
        for owner, name, is_log in ((distribution, "std_param", False), (distribution, "log_std_param", True),
                                   (model, "std", False), (model, "log_std", True)):
            parameter = getattr(owner, name, None) if owner is not None else None
            if parameter is not None:
                parameter.fill_(float(torch.log(torch.tensor(std))) if is_log else std)
                return
    raise TypeError(f"Unsupported noise distribution: {type(model).__name__}")


def freeze_observation_normalizers(runner) -> list[str]:
    """Retain checkpoint normalization while stopping observation-stat updates.

    Use the empirical normalizer's sample cutoff instead of eval(): runner.learn
    switches the model back to train mode. The cutoff is a runtime option and
    must be requested again when resuming training from a saved checkpoint.
    """
    roots = {"policy": policy_module(runner)}
    critic = getattr(runner.alg, "critic", None)
    if critic is not None:
        roots["critic"] = critic
    names = {"obs_normalizer", "actor_obs_normalizer", "critic_obs_normalizer",
             "student_obs_normalizer", "teacher_obs_normalizer"}
    frozen, seen = [], set()
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            if name.rsplit(".", 1)[-1] not in names or id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, torch.nn.Identity):
                continue
            if not all(hasattr(module, attr) for attr in ("until", "count", "update", "_mean", "_std")):
                raise TypeError(f"Unsupported observation normalizer: {type(module).__name__}")
            module.until = int(module.count.item())
            frozen.append(f"{root_name}.{name}")
    return frozen


def export_policy(runner, path: str | Path) -> None:
    """Use runner-native exporters when available, including v5 TensorDict models."""
    path = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    if hasattr(runner, "export_policy_to_jit"):
        runner.export_policy_to_jit(path=path, filename="policy.pt")
        runner.export_policy_to_onnx(path=path, filename="policy.onnx")
    else:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
        model = policy_module(runner)
        normalizer = next((getattr(model, n) for n in ("actor_obs_normalizer", "student_obs_normalizer") if hasattr(model, n)), None)
        export_policy_as_jit(model, normalizer=normalizer, path=path, filename="policy.pt")
        export_policy_as_onnx(model, normalizer=normalizer, path=path, filename="policy.onnx")
