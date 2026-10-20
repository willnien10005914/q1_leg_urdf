#!/usr/bin/env python3
"""Publish a confirmed v4 policy, its real recording and the matching README.

Invoked by the curriculum supervisor after policy promotion. Resumption reuses
media only when checkpoint and content hashes still match. GoFile responses are
local secrets; only the public download URL is copied into published artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get("Q1_LEG_ROOT", Path(__file__).resolve().parents[1]))
START = "<!-- V4_STATUS_START -->"
END = "<!-- V4_STATUS_END -->"
VIDEO = "docs/results/q1_leg_interactive_v4.mp4"
GIF = "docs/images/interactive-v4.gif"
EVALUATION = "docs/results/interactive_v4_evaluation.json"
CHECKPOINT_FOLDER = "checkpoints/isaac/interactive_v4"
COMMIT_MESSAGE = "Train command-conditioned locomotion and mirrored sidestep PPO"

# Deliberately enumerate individual project files. Never stage a whole directory,
# git add '.', GoFile JSON, failed candidates, logs or training intermediates.
PUBLISH_PATHS = frozenset((
    ".gitignore", "README.md", "config/interactive_v4.json", "config/v4_gaits.json",
    "docs/SIDESTEP_V4_MOTION_REVIEW.md", "docs/reference/sidestep_v4_contact_sheet.jpg",
    "docs/V4_ARCHITECTURE.md", "docs/TRAINING_V4_REVIEW.md", "docs/V4_ACCEPTANCE.md",
    GIF, VIDEO, "docs/results/q1_leg_interactive_v4.json", EVALUATION,
    "play_interactive_web.sh", "record_interactive_v4.sh", "start_native_web.sh",
    "start_interactive_training.sh", "train_interactive_native.sh", "scripts/run_interactive_training.sh",
    "tools/finalize_interactive_v4.py", "tools/interactive_curriculum.py",
    "tools/run_interactive_training.py",
    "tools/build_v4_gaits.py", "tools/video_to_gif.py", "tools/status.py",
    "scripts/reinforcement_learning/rsl_rl/play_interactive_web.py",
    "scripts/reinforcement_learning/rsl_rl/record_interactive_v4.py",
    "scripts/reinforcement_learning/rsl_rl/train_interactive.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/interactive_evaluation.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/v4_reference.py",
    "web/app.js", "web/index.html", "web/style.css", "web/serve.py",
    "tests/verify_keyboard_web.py", "tests/test_interactive_control.py",
    "tests/test_interactive_gates.py", "tests/test_interactive_finalizer.py",
    f"{CHECKPOINT_FOLDER}/latest.pt", f"{CHECKPOINT_FOLDER}/contract.json",
    f"{CHECKPOINT_FOLDER}/evaluation.json",
))


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        if private:
            os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_gate(root):
    script = root / "scripts/reinforcement_learning/rsl_rl/play_interactive_web.py"
    spec = importlib.util.spec_from_file_location("q1_interactive_policy_gate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy_gate(root)


def checked_url(value):
    if not isinstance(value, str) or not re.fullmatch(r"https://gofile\.io/d/[A-Za-z0-9]+", value):
        raise ValueError("Upload did not return a valid public GoFile download page")
    return value


def reusable_upload(response, video):
    """Never reuse an upload solely because its local filename matches."""
    if not isinstance(response, dict) or response.get("status") != "ok":
        return None
    data = response.get("data", {})
    if not isinstance(data, dict) or data.get("md5") != digest(video, "md5"):
        return None
    try:
        return checked_url(data.get("downloadPage"))
    except ValueError:
        return None


def recording_valid(video, metadata, checkpoint_sha):
    try:
        modes = {event["mode"] for event in metadata["mode_changes"]}
        expected = metadata["expected_duration"]
        return bool(
            Path(video).is_file() and Path(video).stat().st_size > 1024
            and metadata.get("controller_version") == "q1_leg_v4"
            and metadata.get("checkpoint_sha256") == checkpoint_sha
            and metadata.get("confirmation_episodes", 0) >= 128
            and metadata.get("succeeded") is True and metadata.get("completed") is True
            and metadata.get("survived") is True and metadata.get("actual_failed") is False
            and metadata.get("final_mode") == "stand"
            and metadata.get("final_sequence_active") is False
            and metadata.get("no_reset_count") == 0
            and metadata.get("reset_count_after_start") == 0
            and metadata.get("auto_resets_suppressed") == 0
            and metadata.get("frames", 0) == metadata.get("expected_frames")
            and metadata.get("frames", 0) > 500
            and isinstance(expected, (int, float)) and math.isfinite(expected)
            and abs(metadata["duration"] - expected) < .03
            and {"sidestep_left", "sidestep_right"}.issubset(modes)
            and metadata.get("video_sha256") == digest(video))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def completed_iterations(report, root):
    value = report.get("completed_iterations")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    for file in (root / CHECKPOINT_FOLDER / "training_result.json",
                 root / "outputs/interactive_training_status.json",
                 root / "outputs/interactive_v4_training_status.json"):
        data = read_json(file)
        value = data.get("completed_iterations")
        if (data.get("checkpoint_sha256") == report.get("checkpoint_sha256")
                and isinstance(value, int) and not isinstance(value, bool) and value > 0):
            return value
    raise ValueError("Confirmed evaluation must identify the actual completed_iterations")


def replace_readme(text, block):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one V4_STATUS_START / END marker pair")
    before, tail = text.split(START, 1)
    _old, after = tail.split(END, 1)
    return before + START + "\n" + block.strip() + "\n" + END + after


def readme_block(report, config, recording, url, iterations):
    episodes = report.get("confirmation_episodes", report.get("episodes"))
    envs = config["num_envs"]
    return f"""**v4 鍵盤控制與左右閃身 PPO 已通過驗證。**

訓練使用 **{envs:,} 個平行 envs**，實際完成 **{iterations:,} iterations**（上限 {config['max_iterations']:,}）。
採用 DirectRLEnv 與同一個指令條件 PPO，連續切換站立、方向行走與雙側閃身；通過 **{episodes} 回合獨立確認**，包含各方向、抬腳、落腳與閃身動作門檻。評估詳見 [完整結果]({EVALUATION})。

![訓練完成的 WASD 移動與左右閃身]({GIF})

示範影片由原生 Isaac Sim／PhysX 執行，seed **{recording['seed']}**，完整 **{recording['duration']:.2f} 秒**，完成前後左右移動、左右各一次閃身並回站立，全程沒有重置機器人。[GoFile 完整影片]({url})。

```bash
cd ~/projects/q1_leg_urdf
bash start_native_web.sh
```

開啟 http://127.0.0.1:8876/web/ ，選「即時 PPO / Isaac PhysX」並啟用鍵盤。
**W／S** 前後、**A／D** 左右、**Q／E** 轉向；放開按鍵停止，**Space** 可中止動作。
按「閃身步 · 左右各一次 → 站立」執行有限序列；序列期間方向輸入鎖定，結束後可繼續鍵盤控制。

重新錄製驗證：`bash record_interactive_v4.sh --seed 42`。
這些成果為原生 CAD 模擬，尚未經實機驗證。"""


class Finalizer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root / "outputs/interactive_finalization.json"
        self.logfile = self.root / "outputs/interactive_finalization.log"
        self.result = read_json(self.output)
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.logfile, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.close(fd)
        os.chmod(self.logfile, 0o600)

    def log(self, message):
        with self.logfile.open("a") as stream:
            stream.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")

    def progress(self, state, **values):
        self.result.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        write_json(self.output, self.result, private=True)
        self.log(state)

    def run(self, arguments, label, *, private_output=False, check=True, timeout=None, env=None):
        self.log(label)
        environment = (os.environ if env is None else env).copy()
        environment["Q1_LEG_ROOT"] = str(self.root)
        process = subprocess.run([str(item) for item in arguments], cwd=self.root,
                                 text=True, capture_output=True, timeout=timeout, env=environment,
                                 umask=0o077 if private_output else -1)
        if not private_output:
            with self.logfile.open("a") as stream:
                stream.write(process.stdout)
                stream.write(process.stderr)
        # Upload output can contain account tokens on failure. It is never logged
        # or included in an exception; parse only its protected response file.
        if check and process.returncode:
            raise RuntimeError(f"{label} failed (exit {process.returncode})")
        return process

    def git(self, *arguments, **kwargs):
        return self.run(["git", *arguments], "git " + " ".join(arguments[:2]), **kwargs)

    def preflight(self):
        branch = self.git("branch", "--show-current").stdout.strip()
        if branch != "main":
            raise ValueError("Automatic publication requires the existing main checkout")
        remote = self.git("remote", "get-url", "--push", "origin", private_output=True).stdout.strip()
        if remote.rstrip("/").removesuffix(".git") not in (
            "git@github.com:willnien10005914/q1_leg_urdf",
            "ssh://git@github.com/willnien10005914/q1_leg_urdf",
        ):
            raise ValueError("origin must use the existing SSH URL for willnien10005914/q1_leg_urdf")
        staged = self.git("diff", "--cached", "--name-only", "-z").stdout.split("\0")
        if any(path and path not in PUBLISH_PATHS for path in staged):
            raise ValueError("Unrelated files are already staged; leaving the index unchanged")
        self.git("status", "--short")
        text = (self.root / "README.md").read_text()
        replace_readme(text, "preflight")
        return text

    def record(self, checkpoint_sha):
        video = self.root / VIDEO
        metadata = read_json(video.with_suffix(".json"))
        if recording_valid(video, metadata, checkpoint_sha):
            self.log("Reusing complete recording with matching policy and video SHA256")
            return metadata
        attempts = self.result.get("recording_attempts", [])
        attempted_seeds = {item.get("seed") for item in attempts}
        for seed in (42, 43, 44):
            if seed in attempted_seeds:
                continue
            self.progress("recording", recording_seed=seed)
            environment = os.environ.copy()
            environment["OUT"] = VIDEO
            attempt = self.run(["bash", "record_interactive_v4.sh", "--seed", seed],
                               f"Record native v4 control sequence seed {seed}",
                               check=False, env=environment)
            metadata = read_json(video.with_suffix(".json"))
            success = attempt.returncode == 0 and recording_valid(video, metadata, checkpoint_sha)
            candidate = video.with_name(video.stem + ".candidate.mp4")
            candidate_report = candidate.with_suffix(".json")
            attempt_info = {"seed": seed, "succeeded": success, "exit_code": attempt.returncode}
            if not success:
                archive = self.root / "outputs/interactive_recordings" / checkpoint_sha / f"seed{seed}"
                archive.mkdir(parents=True, exist_ok=True)
                for file in (candidate, candidate_report):
                    if file.is_file():
                        shutil.copy2(file, archive / file.name)
                attempt_info["metadata"] = str((archive / candidate_report.name).relative_to(self.root))
            attempts.append(attempt_info)
            self.progress("recording", recording_attempts=attempts)
            if success:
                return metadata
        raise RuntimeError("No complete v4 recording succeeded for seeds 42, 43 or 44; candidates remain local")

    def make_gif(self, video_sha):
        gif = self.root / GIF
        if (self.result.get("gif_video_sha256") == video_sha and gif.is_file()
                and self.result.get("gif_sha256") == digest(gif)
                and 1024 < gif.stat().st_size < 20 * 1024 * 1024):
            self.log("Reusing GIF with matching video and GIF hashes")
            return digest(gif)
        self.progress("converting_gif")
        for width, colors in ((640, 128), (540, 96), (480, 64)):
            self.run([self.root / ".venv/bin/python", "tools/video_to_gif.py", VIDEO, GIF,
                      "--fps", "10", "--max-width", width, "--colors", colors, "--force"],
                     f"Convert verified movie to GIF ({width}px, 10 FPS)")
            if 1024 < gif.stat().st_size < 20 * 1024 * 1024:
                gif_sha = digest(gif)
                self.progress("gif_ready", gif_video_sha256=video_sha, gif_sha256=gif_sha)
                return gif_sha
        raise RuntimeError("GIF exceeds the 20 MiB publication limit")

    def upload(self, video_sha):
        video = self.root / VIDEO
        response_file = video.with_suffix(".gofile.json")
        url = reusable_upload(read_json(response_file), video)
        if url:
            self.log("Reusing GoFile upload with matching MD5")
            return url
        if self.result.get("uploaded_video_sha256") == video_sha and self.result.get("url"):
            self.log("Reusing completed upload with matching locally recorded SHA256")
            return checked_url(self.result["url"])
        self.progress("uploading")
        self.run([self.root / ".venv/bin/python", "tools/upload_gofile.py", VIDEO, response_file],
                 "Upload verified native recording to GoFile", private_output=True)
        if response_file.exists():
            os.chmod(response_file, 0o600)
        url = reusable_upload(read_json(response_file), video)
        if not url:
            raise RuntimeError("GoFile response did not confirm the uploaded file MD5 and public URL")
        return url

    def publish(self):
        gate = load_gate(self.root)
        if not gate["ready"]:
            raise ValueError(gate["error"])
        sha = gate["checkpoint_sha256"]
        if self.result.get("checkpoint_sha256") != sha:
            self.result = {}
        self.result.pop("error", None)
        self.progress("validating", checkpoint_sha256=sha)
        self.preflight()
        report = read_json(self.root / CHECKPOINT_FOLDER / "evaluation.json")
        config = read_json(self.root / "config/interactive_v4.json")
        iterations = completed_iterations(report, self.root)
        metadata = self.record(sha)
        video_sha = digest(self.root / VIDEO)
        gif_sha = self.make_gif(video_sha)
        # Preserve the prior video hash until upload() decides whether its cached
        # URL belongs to these bytes. Do not relabel an older upload as current.
        url = self.upload(video_sha)
        self.progress("uploaded", url=url, uploaded_video_sha256=video_sha)
        self.progress("updating_readme", video=VIDEO, gif=GIF, video_sha256=video_sha,
                      gif_sha256=gif_sha, url=url, completed_iterations=iterations)
        report = {**report, "media": {"video": VIDEO, "video_sha256": video_sha,
                  "gif": GIF, "gif_sha256": gif_sha, "gofile_url": url,
                  "recording_seed": metadata["seed"], "recording_duration": metadata["duration"],
                  "recording_attempts": self.result.get("recording_attempts", [])}}
        write_json(self.root / EVALUATION, report)
        # Read again so unrelated README sections changed during rendering survive.
        readme = self.root / "README.md"
        readme.write_text(replace_readme(readme.read_text(), readme_block(report, config, metadata, url, iterations)))

        # Revalidate just before staging. Never publish a different checkpoint
        # promoted by another process while this recording was being rendered.
        current_gate = load_gate(self.root)
        if not current_gate["ready"] or current_gate.get("checkpoint_sha256") != sha:
            raise RuntimeError("Confirmed checkpoint changed during finalization; rerun for the new policy")
        self.preflight()
        paths = sorted(path for path in PUBLISH_PATHS if (self.root / path).is_file())
        for path in paths:
            if (self.root / path).stat().st_size >= 95 * 1024 * 1024:
                raise ValueError(f"Publication file exceeds the GitHub size limit: {path}")
        self.progress("committing")
        # -f applies only to the exact allowlist and never exposes GoFile secrets.
        self.git("add", "-f", "--", *paths)
        staged = [path for path in self.git("diff", "--cached", "--name-only", "-z").stdout.split("\0") if path]
        if any(path not in PUBLISH_PATHS for path in staged):
            raise ValueError("An unrelated staged change appeared; refusing to include it in the commit")
        self.git("diff", "--cached", "--check")
        if staged:
            self.git("commit", "-m", COMMIT_MESSAGE)
        commit = self.git("rev-parse", "HEAD").stdout.strip()
        self.progress("pushing", commit=commit)
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=20")
        self.git("push", "origin", "main", env=environment)
        remote_head = self.git("ls-remote", "origin", "refs/heads/main", env=environment).stdout.split()
        if not remote_head or remote_head[0] != commit:
            raise RuntimeError("Remote main did not confirm the just-pushed commit")
        self.progress("complete", commit=commit, url=url, video=VIDEO, gif=GIF,
                      checkpoint_sha256=sha, video_sha256=video_sha, gif_sha256=gif_sha)
        print(json.dumps({key: self.result[key] for key in
                          ("state", "url", "video", "gif", "commit", "checkpoint_sha256")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    finalizer = Finalizer(args.project_root)
    try:
        finalizer.publish()
    except BaseException as error:
        finalizer.progress("failed", error=str(error))
        print(f"v4 finalization failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
