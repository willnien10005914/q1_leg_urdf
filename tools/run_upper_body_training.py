#!/usr/bin/env python3
"""Single-GPU, bounded upper-body-v6 curriculum PPO supervisor with independent acceptance gates.

Isaac workers inherit stdout/stderr, so systemd/tee retains the full live log.
Only a fresh 128-episode confirmation can publish a new user-facing policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = ("stand", "w", "s", "a", "d", "yaw_left", "yaw_right", "sequence")
STAGE_CASES = {0: ("stand", "inplace"), 1: ("stand", "left", "right", "pair"), 2: CASE_NAMES}
STOP = False
CHILD = None


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profile(profile):
    """Keep the user's requested budget and acceptance procedure explicit."""
    required = {"num_envs": 10240,
                "num_steps_per_env": 24, "validation_episodes": 32,
                "confirmation_episodes": 128}
    if "training_deadline_utc" in profile:
        raise ValueError("The expired v4 deadline is not part of the upper-body-v6 plan")
    required["max_iterations"]=20000
    for key, expected in required.items():
        if type(profile.get(key)) is not int or profile[key] != expected:
            raise ValueError(f"{key} must remain {expected} for this authorized training plan")
    for key in ("chunk_iterations", "consecutive_passes_to_confirm", "plateau_window_evaluations"):
        if type(profile.get(key)) is not int or profile[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if profile["chunk_iterations"] > profile["max_iterations"]:
        raise ValueError("A training chunk cannot exceed the total budget")
    early = profile.get("early_chunk_iterations", 100)
    if type(early) is not int or not 1 <= early <= profile["chunk_iterations"]:
        raise ValueError("early_chunk_iterations must fit within one ordinary chunk")
    if type(profile.get("early_chunk_until")) is not int or profile["early_chunk_until"] < 0:
        raise ValueError("early_chunk_until must be nonnegative")
    if profile.get("stage_passes_to_advance") != 2:
        raise ValueError("Each preparatory stage requires two independent validations")
    if profile["consecutive_passes_to_confirm"] < 3 or profile["plateau_window_evaluations"] < 4:
        raise ValueError("Confirmation needs three validations; plateau needs at least four")
    if profile.get("max_reward_revisions", 0) not in (0, 1, 2):
        raise ValueError("At most two documented reward revisions are allowed")
    threshold = profile.get("plateau_relative_improvement")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("plateau_relative_improvement must be between zero and one")
    if profile.get("robot") != "q1_leg_upper_body_v6":
        raise ValueError("Unexpected upper-body-v6 training contract")
    if profile.get("budget_mode") != "new_updates" or profile.get("source_iteration") != 13115:
        raise ValueError("V6 authorizes 20000 fresh updates; source 13115 is provenance, not completed V6 work")
    if profile.get("training_episode_lengths_s") != [20, 20, 20]:
        raise ValueError("Every V6 training stage needs the full 20-second recovery horizon")
    if profile.get("source_checkpoint_sha256") != "b4077ac2a5d3fd98de67e43165cc753263c01d60e33240562303f912e11dbab9":
        raise ValueError("V6 must initialize from the frozen user-reviewed 13115 policy")
    return profile


def next_chunk_size(completed, profile, stage=2):
    if type(completed) is not int or completed < 0 or completed > profile["max_iterations"]:
        raise ValueError("Completed iteration count is outside the configured budget")
    requested = profile["early_chunk_iterations"] if stage < 2 or completed < profile["early_chunk_until"] else profile["chunk_iterations"]
    return min(requested, profile["max_iterations"] - completed)


def plateau_diagnostic(evaluations, profile, revision, stage):
    """Plateau is a review trigger, never a successful training outcome."""
    window = profile["plateau_window_evaluations"]
    candidates = [item for item in evaluations if item.get("kind") == "validation"
                  and item.get("reward_revision") == revision and item.get("stage") == stage]
    recent = candidates[-window:]
    if len(recent) < window or recent[-1].get("passed") is True:
        return None
    scores = [item.get("score") for item in recent]
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
        raise ValueError("A non-finite evaluation score cannot be used for plateau detection")
    improvement = (max(scores)-scores[0])/max(abs(scores[0]), 1e-6)
    if improvement >= profile["plateau_relative_improvement"]:
        return None
    failed = Counter(gate for item in recent for gate in item.get("failed_gates", []))
    return {"reason": "Validation score plateau with required gates still failing",
            "reward_revision": revision, "stage": stage, "iterations": [item["completed_iterations"] for item in recent],
            "scores": scores, "relative_improvement": improvement,
            "relative_improvement_threshold": profile["plateau_relative_improvement"],
            "failed_gate_counts": dict(failed.most_common()),
            "acceptance_gates_changed": False, "successful": False}


def validation_streak(previous, report):
    return previous + 1 if report.get("passed") is True else 0


def stage_gate_passed(report, stage, expected_episodes):
    """Fail closed on partial, proxy, missing-case or contradictory reports."""
    checks = report.get("checks", {})
    score = report.get("score")
    return (report.get("passed") is True and report.get("episodes", 0) >= expected_episodes
            and report.get("backend") == "isaaclab" and report.get("proxy") is False
            and report.get("stage") == stage
            and isinstance(score, (float, int)) and math.isfinite(score) and score == 1.
            and bool(checks) and all(value is True for value in checks.values())
            and report.get("lower_body_passed") is True
            and isinstance(report.get("lower_body_checks"), dict) and bool(report["lower_body_checks"])
            and all(value is True for value in report["lower_body_checks"].values())
            and isinstance(report.get("upper_body_checks"), dict) and bool(report["upper_body_checks"])
            and all(value is True for value in report["upper_body_checks"].values())
            and not report.get("failed_gates", [])
            and all(report.get("cases", {}).get(name, {}).get("episodes", 0) >= 1 for name in STAGE_CASES[stage]))


def confirmation_passed(report, expected_episodes=128):
    return stage_gate_passed(report, 2, expected_episodes)


def profile_gait(config_path):
    return Path(json.loads(Path(config_path).read_text()).get("gait_file", "config/v5_gaits_lift6.json"))


def source_manifest(config_path):
    sources = [config_path, ROOT/profile_gait(config_path), ROOT/"config/lift_control_v5.json", ROOT/"config/upper_body_control_v6.json",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/upper_body_env.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/upper_body_objective.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/upper_body_evaluation.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/upper_body_stability.py", ROOT/"train_upper_body_native.sh",
               ROOT/"tools/run_upper_body_training.py", ROOT/"scripts/isaac_env.sh",
               ROOT/"scripts/reinforcement_learning/rsl_rl/train_upper_body.py",
               ROOT/"scripts/reinforcement_learning/rsl_rl/runtime_compat.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/lift_env.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/lift_objective.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/lift_symmetry.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/lift_evaluation.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/interactive_evaluation.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/v4_reference.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/agent_cfg.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/isaac_env.py",
               ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/robot.py",
               ROOT/"urdf/q1_inertia/quanta_robot_legged.urdf"]
    return {str(path.resolve().relative_to(ROOT.resolve())): sha256(path) for path in sources}


def timestamp(path):
    return path.stat().st_mtime_ns if path.exists() else None


def fresh_json(path, before, description):
    if not path.exists() or timestamp(path) == before:
        raise RuntimeError(f"{description} was not freshly written: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error


def checkpoint_valid(path):
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("data.pkl") for name in archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def interrupt(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class Supervisor:
    def __init__(self, profile, config_path, run, resume=False):
        self.profile = profile
        self.config_path = config_path
        self.run = run
        self.run.mkdir(parents=True, exist_ok=True)
        self.state_file = run/"supervisor_state.json"
        manifest = source_manifest(config_path)
        existing_monitor = resume and self.state_file.exists()
        if existing_monitor:
            self.state = json.loads(self.state_file.read_text())
            if self.state.get("profile") != profile or self.state.get("source_sha256") != manifest:
                raise RuntimeError("Profile or training sources changed; use a new run to preserve experiment traceability")
        else:
            snapshot = self.run/"source_snapshot"
            if snapshot.exists():
                raise RuntimeError("New run already contains an experiment source snapshot")
            for relative in manifest:
                destination = snapshot/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/relative, destination)
            self.state = {"schema": 1, "robot": profile["robot"], "state": "created",
                          "backend": "isaaclab", "proxy": False, "skill": "upper_body_curriculum",
                          "ready": False, "started_at": utcnow(), "run": str(run),
                          "profile_path": str(config_path), "profile": profile,
                          "source_sha256": manifest, "num_envs": profile["num_envs"],
                          "max_iterations": profile["max_iterations"], "completed_iterations": 0,
                          "num_steps_per_env": profile["num_steps_per_env"],
                          "expected_max_transitions": profile["num_envs"]*profile["num_steps_per_env"]*profile["max_iterations"],
                          "budget_mode": "new_updates", "source_iteration": profile["source_iteration"],
                          "source_checkpoint_sha256": profile["source_checkpoint_sha256"],
                          "acceptance_inherited": False,
                          "reward_revision": 0, "reward_revisions": [], "evaluations": [],
                          "curriculum_stage": 0, "stage_history": [],
                          "consecutive_passes": 0, "confirmation_attempts": 0,
                          "log": profile.get("log", "outputs/upper_body_training.log")}
        if resume:
            result_path = run/"training_result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                completed = result.get("completed_iterations")
                next_chunk_size(completed, profile)
                if (result.get("num_envs") != profile["num_envs"]
                        or result.get("contract", {}).get("robot") != profile["robot"]
                        or not checkpoint_valid(run/"resume.pt")
                        or (result.get("checkpoint_sha256") and result["checkpoint_sha256"] != sha256(run/"resume.pt"))
                        or result.get("budget_mode") != "new_updates"
                        or result.get("source_iteration") != profile["source_iteration"]
                        or result.get("initialization", {}).get("source_sha256") != profile["source_checkpoint_sha256"]):
                    raise RuntimeError("Run cannot resume from an invalid checkpoint, contract or environment count")
                if completed < self.state["completed_iterations"]:
                    raise RuntimeError("Saved checkpoint predates supervisor progress")
                self.state["completed_iterations"] = completed
                if not existing_monitor:
                    self.state["reward_revision"] = result.get("reward_revision", 0)
                    self.state["curriculum_stage"] = result.get("curriculum_stage", 0)
                    if self.state["curriculum_stage"] != 0:
                        raise RuntimeError("Only stage-0 standalone initialization may be adopted without existing gate history")
                    self.state["adopted_worker_iterations"] = completed
                    self.state["adopted_initialization"] = result.get("initialization")
                    self.state["source_snapshot_note"] = "Adopted completed standalone worker chunk; monitor source manifest captured at adoption"
            elif not existing_monitor or self.state["completed_iterations"]:
                raise RuntimeError("No resumable training_result.json exists in the requested run")
        self.state["pid"] = os.getpid()
        self.save()

    def save(self, **updates):
        self.state.update(updates)
        self.state["updated_at"] = utcnow()
        self.state["elapsed_seconds"] = round((datetime.now(timezone.utc)-datetime.fromisoformat(self.state["started_at"])).total_seconds(), 1)
        self.state["completed_transitions"] = self.state["completed_iterations"]*self.profile["num_envs"]*self.profile["num_steps_per_env"]
        atomic_json(self.state_file, self.state)
        atomic_json(ROOT/"outputs/upper_body_training_status.json", self.state)
        atomic_json(ROOT/"outputs/training_status.json", self.state)

    def assert_sources_unchanged(self):
        if source_manifest(self.config_path) != self.state["source_sha256"]:
            raise RuntimeError("Training sources changed during this experiment; retained checkpoints and stopped")

    def child(self, command, phase):
        global CHILD
        error_file = self.run/"worker_error.json"
        previous_error = timestamp(error_file)
        child_started = time.monotonic()
        self.save(state=phase, error=None, current_command=[str(part) for part in command])
        print("V6_SUPERVISOR " + json.dumps({"state":phase,"completed_iterations":self.state["completed_iterations"],"command":[str(x) for x in command]}), flush=True)
        CHILD = subprocess.Popen([str(part) for part in command], cwd=ROOT,
                                 start_new_session=True)
        self.save(child_pid=CHILD.pid)
        try:
            while CHILD.poll() is None:
                timeout = self.profile.get("validation_worker_timeout_s", 420) if phase != "training" else self.profile.get("training_worker_timeout_s", 10800)
                if time.monotonic() - child_started > timeout:
                    os.killpg(CHILD.pid, signal.SIGTERM)
                    raise RuntimeError(f"{phase} worker exceeded its declared {timeout}s timeout")
                if STOP:
                    try:
                        CHILD.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                        CHILD.wait(timeout=15)
                    raise KeyboardInterrupt
                snapshots = list(self.run.glob("model_*.pt"))
                if phase == "training" and snapshots:
                    indices = [int(path.stem.split("_")[-1])+1 for path in snapshots if path.stem.split("_")[-1].isdigit()]
                    if indices:
                        self.state["live_iteration_estimate"] = min(max(indices), self.state.get("chunk_target_iterations", self.profile["max_iterations"]))
                self.save()
                time.sleep(5)
            result = CHILD.returncode
        finally:
            if CHILD.poll() is None:
                try:
                    os.killpg(CHILD.pid, signal.SIGTERM)
                    CHILD.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if CHILD.poll() is None:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                        CHILD.wait(timeout=15)
            CHILD = None
            self.save(child_pid=None)
        if STOP:
            raise KeyboardInterrupt
        if error_file.exists() and timestamp(error_file) != previous_error:
            error = fresh_json(error_file, previous_error, "worker exception")
            raise RuntimeError(f"{phase} native worker error: {error.get('type')}: {error.get('error')}")
        if result != 0:
            raise RuntimeError(f"{phase} worker failed with exit code {result}; see inherited training log")

    def train_chunk(self):
        self.assert_sources_unchanged()
        completed = self.state["completed_iterations"]
        count = next_chunk_size(completed, self.profile, self.state["curriculum_stage"])
        if count<=0:return False
        result_path = self.run/"training_result.json"
        before, previous_checkpoint = timestamp(result_path), timestamp(self.run/"resume.pt")
        command = ["bash", ROOT/"train_upper_body_native.sh", "--output-dir", self.run,
                   "--training-profile",self.config_path,
                   "--num_envs", self.profile["num_envs"], "--iterations", count,
                   "--completed", completed, "--reward-revision", self.state["reward_revision"],
                   "--curriculum-stage", self.state["curriculum_stage"], "--headless"]
        if completed:
            command += ["--checkpoint", self.run/"resume.pt"]
        else:
            if self.profile.get("bootstrap"):
                bootstrap = (ROOT/self.profile["bootstrap"]).resolve()
                if not checkpoint_valid(bootstrap):
                    raise RuntimeError(f"Missing or invalid initialization checkpoint: {bootstrap}")
                if sha256(bootstrap) != self.profile["source_checkpoint_sha256"]:
                    raise RuntimeError("Initialization checkpoint differs from frozen 13115 source")
                command += ["--checkpoint", bootstrap, "--initialize-policy"]
        self.save(chunk_iterations=count, chunk_target_iterations=completed+count)
        started=time.monotonic()
        self.child(command, "training")
        self.assert_sources_unchanged()
        result = fresh_json(result_path, before, "training_result.json")
        if (result.get("completed_iterations") != completed+count
                or result.get("num_envs") != self.profile["num_envs"]
                or result.get("reward_revision") != self.state["reward_revision"]
                or result.get("curriculum_stage") != self.state["curriculum_stage"]
                or result.get("contract", {}).get("robot") != self.profile["robot"]
                or timestamp(self.run/"resume.pt") == previous_checkpoint
                or not checkpoint_valid(self.run/"resume.pt")
                or result.get("checkpoint_sha256") != sha256(self.run/"resume.pt")
                or result.get("budget_mode") != "new_updates"
                or result.get("source_iteration") != self.profile["source_iteration"]
                or result.get("initialization", {}).get("source_sha256") != self.profile["source_checkpoint_sha256"]):
            raise RuntimeError("Training result or resumable checkpoint does not match the requested chunk")
        # Never overwrite a previous segment's policy: resume.pt is a pointer,
        # while each verified training segment remains available for diagnosis.
        archive=self.run/"chunks"/f"{completed+count:07d}_s{self.state['curriculum_stage']}_r{self.state['reward_revision']}"
        if archive.exists():raise RuntimeError("Refusing to overwrite a retained training segment")
        archive.mkdir(parents=True)
        for name in ("resume.pt","contract.json","training_result.json"):
            shutil.copy2(self.run/name,archive/name)
        self.save(completed_iterations=completed+count, live_iteration_estimate=completed+count,
                  seconds_per_iteration=(time.monotonic()-started)/count,last_segment=str(archive))
        return True

    def evaluate(self, episodes, seed, kind):
        self.assert_sources_unchanged()
        completed = self.state["completed_iterations"]
        stage = self.state["curriculum_stage"]
        output = self.run/f"evaluation_{completed}_s{stage}_{seed}.json"
        before = timestamp(output)
        checkpoint = self.run/"resume.pt"
        command = ["bash", ROOT/"train_upper_body_native.sh", "--evaluate", "--episodes", episodes,
                   "--training-profile",self.config_path,
                   "--num_envs", episodes, "--seed", seed, "--completed", completed,
                   "--checkpoint", checkpoint, "--output-dir", self.run,
                   "--reward-revision", self.state["reward_revision"], "--curriculum-stage", stage, "--headless"]
        self.validation_worker(command, "confirming" if kind == "confirmation" else "validating", output)
        self.assert_sources_unchanged()
        report = fresh_json(output, before, "independent evaluation")
        contract = json.loads((self.run/"contract.json").read_text())
        score = report.get("score")
        if (report.get("episodes") != episodes or report.get("evaluation_seed") != seed
                or report.get("completed_iterations") != completed
                or report.get("checkpoint_sha256") != sha256(checkpoint)
                or report.get("urdf_sha256") != contract.get("urdf_sha256")
                or report.get("gait_sha256") != contract.get("gait_sha256")
                or report.get("stage") != stage
                or report.get("control_sha256") != contract.get("control_sha256")
                or report.get("upper_body_control_sha256") != contract.get("control_sha256")
                or report.get("upper_body_evaluator_sha256") != self.state["source_sha256"]["source/leg_humanoid_lab/leg_humanoid_lab/upper_body_evaluation.py"]
                or report.get("robot") != self.profile["robot"] or report.get("schema") != 6
                or report.get("environment_sha256") != self.state["source_sha256"]["source/leg_humanoid_lab/leg_humanoid_lab/upper_body_env.py"]
                or report.get("base_environment_sha256") != self.state["source_sha256"]["source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py"]
                or report.get("proxy") is not False or report.get("backend") != "isaaclab"
                or not isinstance(score, (float, int)) or not math.isfinite(score)
                or not isinstance(report.get("passed"), bool)
                or not all(name in report.get("cases", {}) for name in STAGE_CASES[stage])):
            raise RuntimeError("Evaluation metadata does not match its checkpoint, source or fresh-seed suite")
        if report["passed"] and not stage_gate_passed(report, stage, episodes):
            raise RuntimeError("Evaluation claims success with incomplete or contradictory full-gate checks")
        summary = {"kind":kind, "completed_iterations":completed, "reward_revision":self.state["reward_revision"],
                   "stage":stage,
                   "episodes":episodes, "seed":seed, "passed":report["passed"], "score":score,
                   "failed_gates":report.get("failed_gates", []), "report":str(output),
                   "report_sha256":sha256(output), "checkpoint_sha256":report["checkpoint_sha256"],
                   "source_sha256":self.state["source_sha256"], "evaluated_at":utcnow()}
        self.state["evaluations"].append(summary)
        self.save(last_evaluation=summary, best_score=max(score, self.state.get("best_score", 0.)))
        previous_best=self.state.get("best_by_stage",{}).get(str(stage),{})
        if score>previous_best.get("score",-1.):
            best_folder=self.run/"best"/f"s{stage}_{completed:07d}"
            best_folder.mkdir(parents=True,exist_ok=False)
            for name in ("resume.pt","contract.json","training_result.json"):
                shutil.copy2(self.run/name,best_folder/name)
            shutil.copy2(output,best_folder/"evaluation.json")
            best=dict(self.state.get("best_by_stage",{}));best[str(stage)]={"score":score,"path":str(best_folder),"completed_iterations":completed}
            self.save(best_by_stage=best)
        print("V6_VALIDATION " + json.dumps(summary), flush=True)
        return report

    def validation_worker(self, command, phase, expected_output=None):
        """Retry one failed validation process; never repeat a PPO chunk.

        Kit can crash during startup before producing any evaluation. A second
        fresh process uses the same immutable checkpoint and seed. Metadata is
        still checked by evaluate(); repeated errors retain the usual stop.
        """
        if phase not in ("validating", "confirming"):
            raise ValueError('Only read-only validation workers may be retried')
        for attempt in range(2):
            before = timestamp(expected_output) if expected_output else None
            try:
                self.child(command, phase)
                if expected_output:
                    fresh_json(expected_output, before, "validation worker output")
                return
            except RuntimeError as error:
                if attempt or STOP:
                    raise
                self.save(validation_worker_restart={"at":utcnow(),"phase":phase,
                    "attempt":1,"completed_iterations":self.state["completed_iterations"],
                    "reason":str(error),"training_chunk_repeated":False})
                print('V6_VALIDATION_RESTART '+json.dumps(self.state['validation_worker_restart']),flush=True)
                time.sleep(3)

    def publication_report_identity(self, report, checkpoint, contract):
        """Bind a complete native report to immutable weights and frozen sources."""
        sources = self.state["source_sha256"]
        return (checkpoint_valid(checkpoint)
                and report.get("checkpoint_sha256") == sha256(checkpoint)
                and report.get("robot") == self.profile["robot"] and report.get("schema") == 6
                and report.get("urdf_sha256") == contract.get("urdf_sha256")
                and report.get("gait_sha256") == contract.get("gait_sha256")
                and report.get("control_sha256") == contract.get("control_sha256")
                and report.get("upper_body_control_sha256") == contract.get("control_sha256")
                and report.get("environment_sha256") == sources.get("source/leg_humanoid_lab/leg_humanoid_lab/upper_body_env.py")
                and report.get("base_environment_sha256") == sources.get("source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py")
                and report.get("upper_body_evaluator_sha256") == sources.get("source/leg_humanoid_lab/leg_humanoid_lab/upper_body_evaluation.py"))

    def preceding_stage2_validations(self):
        """Reload three full sealed reports; summaries alone can never publish."""
        needed = self.profile["consecutive_passes_to_confirm"]
        history = [item for item in self.state["evaluations"] if item.get("kind") == "validation"]
        if (self.state["curriculum_stage"] != 2 or self.state["consecutive_passes"] < needed
                or len(history) < needed):
            raise RuntimeError("Publication requires three preceding complete stage-2 validation reports")
        recent = history[-needed:]
        proofs = []; seeds = set(); previous_iteration = -1
        for summary in recent:
            completed = summary.get("completed_iterations")
            revision = summary.get("reward_revision")
            path = Path(summary.get("report", ""))
            if (summary.get("stage") != 2 or revision != self.state["reward_revision"]
                    or type(completed) is not int or completed <= previous_iteration
                    or summary.get("passed") is not True or summary.get("score") != 1.
                    or summary.get("source_sha256") != self.state["source_sha256"]
                    or not path.is_file() or summary.get("report_sha256") != sha256(path)):
                raise RuntimeError("Preceding stage-2 validation history is incomplete, changed or not consecutive")
            report = json.loads(path.read_text())
            seed = report.get("evaluation_seed")
            archive = self.run/"chunks"/f"{completed:07d}_s2_r{revision}"
            checkpoint = archive/"resume.pt"
            contract_path = archive/"contract.json"
            result_path = archive/"training_result.json"
            if not contract_path.is_file() or not result_path.is_file():
                raise RuntimeError("Preceding validation is missing its retained checkpoint provenance")
            contract = json.loads(contract_path.read_text())
            result = json.loads(result_path.read_text())
            if (not stage_gate_passed(report, 2, self.profile["validation_episodes"])
                    or report.get("episodes") != self.profile["validation_episodes"]
                    or report.get("completed_iterations") != completed
                    or type(seed) is not int or seed != summary.get("seed") or seed in seeds
                    or report.get("checkpoint_sha256") != summary.get("checkpoint_sha256")
                    or not self.publication_report_identity(report, checkpoint, contract)
                    or result.get("checkpoint_sha256") != report["checkpoint_sha256"]
                    or result.get("completed_iterations") != completed or result.get("curriculum_stage") != 2
                    or result.get("reward_revision") != revision or result.get("budget_mode") != "new_updates"
                    or result.get("initialization", {}).get("source_sha256") != self.profile["source_checkpoint_sha256"]):
                raise RuntimeError("Preceding full validation does not match its native gates, source or retained weights")
            proofs.append({**report, "reward_revision":revision,
                "validation_provenance":{"report":str(path),"report_sha256":summary["report_sha256"],
                    "immutable_checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
                    "contract_sha256":sha256(contract_path),"training_result_sha256":sha256(result_path),
                    "source_sha256":self.state["source_sha256"],"evaluated_at":summary.get("evaluated_at")}})
            seeds.add(seed); previous_iteration = completed
        if previous_iteration != self.state["completed_iterations"]:
            raise RuntimeError("Newest preceding validation does not cover the confirmed current weights")
        return proofs

    def promote(self, report):
        if not confirmation_passed(report, self.profile["confirmation_episodes"]):
            raise RuntimeError("Only a full independent confirmation may promote a policy")
        self.assert_sources_unchanged()
        preceding = self.preceding_stage2_validations()
        source = self.run/"resume.pt"
        contract = json.loads((self.run/"contract.json").read_text())
        confirmation = self.state.get("last_evaluation", {})
        confirmation_path = Path(confirmation.get("report", ""))
        if (not self.publication_report_identity(report, source, contract)
                or report.get("completed_iterations") != self.state["completed_iterations"]
                or report.get("checkpoint_sha256") != preceding[-1].get("checkpoint_sha256")
                or report.get("evaluation_seed") in {item["evaluation_seed"] for item in preceding}
                or confirmation.get("kind") != "confirmation" or confirmation.get("passed") is not True
                or confirmation.get("source_sha256") != self.state["source_sha256"]
                or not confirmation_path.is_file() or confirmation.get("report_sha256") != sha256(confirmation_path)
                or json.loads(confirmation_path.read_text()) != report):
            raise RuntimeError("Confirmed checkpoint, source or sealed confirmation report changed before publication")
        destination = ROOT/"checkpoints/isaac/upper_body_v6"
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/"latest.pt").exists():
            archive = destination/"published_history"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            archive.mkdir(parents=True)
            for name in ("latest.pt", "contract.json", "evaluation.json"):
                if (destination/name).exists():
                    shutil.copy2(destination/name, archive/name)
        temporary = destination/"latest.pt.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(destination/"latest.pt")
        atomic_json(destination/"contract.json", json.loads((self.run/"contract.json").read_text()))
        published = {**report, "checkpoint":"latest.pt", "checkpoint_sha256":sha256(destination/"latest.pt"),
                     "confirmation_episodes":self.profile["confirmation_episodes"], "ready":True,
                     "published_from":str(source), "published_at":utcnow(),
                     "source_sha256":self.state["source_sha256"],
                     "training_profile":self.profile,
                     "reward_revision":self.state["reward_revision"],
                     "preceding_stage2_validations":preceding,
                     "confirmation_provenance":{"report":str(confirmation_path),
                         "report_sha256":sha256(confirmation_path),"checkpoint_sha256":sha256(source),
                         "source_sha256":self.state["source_sha256"],"evaluation_seed":report["evaluation_seed"]},
                     "initial_actor_provenance":{"source_iteration":self.profile["source_iteration"],
                         "source_checkpoint_sha256":self.profile["source_checkpoint_sha256"],
                         "inherited_completed_iterations":0,"budget_mode":"new_updates"}}
        # Evaluation is the final publication marker, after weights and contract.
        atomic_json(destination/"evaluation.json", published)
        self.save(state="passed", ready=True, evaluation=published, published_checkpoint=str(destination/"latest.pt"))

    def process_validation(self):
        completed = self.state["completed_iterations"]
        stage = self.state["curriculum_stage"]
        report = self.evaluate(self.profile["validation_episodes"], self.profile["validation_seed_base"]+stage*100000+completed, "validation")
        self.save(consecutive_passes=validation_streak(self.state["consecutive_passes"], report))
        if stage < 2 and self.state["consecutive_passes"] >= self.profile["stage_passes_to_advance"]:
            transition={"from":stage,"to":stage+1,"completed_iterations":completed,"validation":self.state["last_evaluation"],"at":utcnow()}
            self.state["stage_history"].append(transition)
            atomic_json(self.run/f"stage_{stage}_passed.json",transition)
            self.save(curriculum_stage=stage+1,consecutive_passes=0,stage_started_iterations=completed)
            print("V6_STAGE_ADVANCE "+json.dumps(transition),flush=True)
            return False
        if stage == 2 and self.state["consecutive_passes"] >= self.profile["consecutive_passes_to_confirm"]:
            attempt = self.state["confirmation_attempts"]
            self.save(confirmation_attempts=attempt+1)
            confirmation = self.evaluate(self.profile["confirmation_episodes"], self.profile["confirmation_seed"]+1000*attempt, "confirmation")
            if confirmation_passed(confirmation, self.profile["confirmation_episodes"]):
                self.promote(confirmation)
                self.save(state="complete", ready=True, artifact_status="awaiting_v6_artifact_pipeline")
                return True
            self.save(consecutive_passes=0)
        diagnostic = plateau_diagnostic(self.state["evaluations"], self.profile, self.state["reward_revision"], stage)
        if diagnostic:
            revision = self.state["reward_revision"]
            diagnostic.update(detected_at=utcnow(), completed_iterations=completed, cases=report["cases"])
            if revision < self.profile["max_reward_revisions"] and completed < self.profile["max_iterations"]:
                diagnostic.update(next_reward_revision=revision+1,
                                  action="Enable next predeclared reward revision; keep all acceptance gates unchanged")
                path = self.run/f"reward_revision_{revision+1}.json"
                atomic_json(path, diagnostic)
                self.state["reward_revisions"].append({"from":revision,"to":revision+1,"completed_iterations":completed,"diagnostic":str(path)})
                self.save(reward_revision=revision+1, consecutive_passes=0, plateau_diagnostic=str(path))
                print("V6_REWARD_REVISION " + json.dumps(diagnostic), flush=True)
            else:
                path = self.run/f"unresolved_plateau_{completed}.json"
                atomic_json(path, diagnostic)
                self.save(plateau_unresolved=True, plateau_diagnostic=str(path))
        return False

    def run_training(self):
        if self.state.get("ready"):
            return 0
        completed = self.state["completed_iterations"]
        validations = [item for item in self.state["evaluations"] if item["kind"] == "validation"]
        if completed and (not validations or validations[-1]["completed_iterations"] < completed):
            if self.process_validation():
                return 0
        while self.state["completed_iterations"] < self.profile["max_iterations"]:
            if STOP:
                raise KeyboardInterrupt
            if not self.train_chunk():
                self.save(state="budget_exhausted",ready=False,error="No remaining authorized updates; retained checkpoints without relaxing gates")
                return 2
            if self.process_validation():
                return 0
        self.save(state="budget_exhausted", ready=False,
                  error=f"Requested {self.profile['max_iterations']}-iteration budget completed without an independent full-gate confirmation")
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/upper_body_v6.json")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print((ROOT/"outputs/upper_body_training_status.json").read_text())
        return 0
    config_path = args.config.resolve()
    profile = validate_profile(json.loads(config_path.read_text()))
    (ROOT/"outputs").mkdir(exist_ok=True)
    lock = (ROOT/"outputs/upper_body_training.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A upper-body-v6 training supervisor already owns this GPU queue")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    run = args.resume_run.resolve() if args.resume_run else ROOT/"checkpoints/isaac/upper_body_v6/runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    supervisor = None
    try:
        supervisor = Supervisor(profile, config_path, run, resume=args.resume_run is not None)
        return supervisor.run_training()
    except KeyboardInterrupt:
        if supervisor:
            supervisor.save(state="interrupted", error="Supervisor interrupted; resume the same run to preserve the remaining budget")
        return 130
    except Exception as error:
        if supervisor and supervisor.state.get("state") != "artifact_error":
            supervisor.save(state="runtime_error", error=f"{type(error).__name__}: {error}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
