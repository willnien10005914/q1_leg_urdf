"""Pure local checks for safe, resumable v4 artifact publication decisions."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("interactive_finalizer", ROOT / "tools/finalize_interactive_v4.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def recording(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"local-video-test" * 200)
    metadata = {
        "controller_version": "q1_leg_v4", "checkpoint_sha256": "selected-sha",
        "confirmation_episodes": 128, "succeeded": True, "survived": True,
        "actual_failed": False, "completed": True, "final_mode": "stand",
        "final_sequence_active": False, "no_reset_count": 0,
        "reset_count_after_start": 0, "auto_resets_suppressed": 0,
        "frames": 935, "expected_frames": 935, "duration": 18.7,
        "expected_duration": 18.7, "video_sha256": MODULE.digest(video),
        "post_sequence_resume": {"accepted": True, "full_period": True,
            "sequence_completed_before_resume": True, "duration_s": 2.,
            "walk_frames": 100, "frames": 100, "displacement_m": .08, "forward_displacement_m": .08},
        "final_stand_speed_m_s": .01,
        "mode_changes": [{"mode": "sidestep_left"}, {"mode": "sidestep_right"}, {"mode": "stand"}],
    }
    return video, metadata


def test_video_reuse_requires_complete_same_policy_and_content(tmp_path):
    video, metadata = recording(tmp_path)
    assert MODULE.recording_valid(video, metadata, "selected-sha")
    assert not MODULE.recording_valid(video, metadata, "older-policy")
    for key, bad in (("succeeded", False), ("survived", False), ("actual_failed", True),
                     ("completed", False), ("final_mode", "recover"),
                     ("final_sequence_active", True), ("auto_resets_suppressed", 1),
                     ("frames", 934), ("confirmation_episodes", 127),
                     ("final_stand_speed_m_s", .08)):
        assert not MODULE.recording_valid(video, {**metadata, key: bad}, "selected-sha"), key
    video.write_bytes(b"different-video" * 200)
    assert not MODULE.recording_valid(video, metadata, "selected-sha")


def test_video_rejects_missing_post_sequence_takeover(tmp_path):
    video, metadata = recording(tmp_path)
    for key, value in (("accepted", False), ("full_period", False),
                       ("sequence_completed_before_resume", False),
                       ("duration_s", .5), ("displacement_m", .01),
                       ("forward_displacement_m", -.1), ("walk_frames", 0)):
        bad = {**metadata, "post_sequence_resume": {**metadata["post_sequence_resume"], key: value}}
        assert not MODULE.recording_valid(video, bad, "selected-sha"), key


def test_upload_cache_requires_matching_file_not_filename(tmp_path):
    video, _metadata = recording(tmp_path)
    response = {"status": "ok", "data": {"md5": MODULE.digest(video, "md5"),
                "downloadPage": "https://gofile.io/d/Example123", "guestToken": "private-test-token"}}
    assert MODULE.reusable_upload(response, video) == "https://gofile.io/d/Example123"
    response["data"]["md5"] = "other-video"
    assert MODULE.reusable_upload(response, video) is None
    with pytest.raises(ValueError):
        MODULE.checked_url("https://gofile.io/d/Example123?token=private-test-token")


def test_readme_only_changes_marked_block_and_uses_actual_iterations():
    before = f"# Existing README\n\nPrevious text\n{MODULE.START}\nOld status\n{MODULE.END}\n\nKeep this section\n"
    block = MODULE.readme_block({"episodes": 128}, {"num_envs": 10240, "max_iterations": 15000},
                                {"seed": 43, "duration": 18.7}, "https://gofile.io/d/Example123", 4500)
    after = MODULE.replace_readme(before, block)
    assert after.startswith("# Existing README\n\nPrevious text\n")
    assert after.endswith("\n\nKeep this section\n")
    assert "4,500 iterations" in after and "seed **43**" in after
    assert f"]({MODULE.GIF})" in after
    assert "128／128" not in after
    with pytest.raises(ValueError):
        MODULE.replace_readme("README without markers", block)


def test_actual_iteration_count_never_defaults_to_budget(tmp_path):
    with pytest.raises(ValueError):
        MODULE.completed_iterations({"episodes": 128}, tmp_path)
    assert MODULE.completed_iterations({"completed_iterations": 4250}, tmp_path) == 4250


def test_published_training_profile_uses_actual_deadline_budget(tmp_path):
    config=tmp_path / 'config/interactive_v4.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'num_envs':10240,'max_iterations':15000}))
    assert MODULE.training_profile({},tmp_path)['max_iterations']==15000
    profile={'num_envs':10240,'max_iterations':801}
    assert MODULE.training_profile({'training_profile':profile},tmp_path)==profile
    for bad in ({'num_envs':10240,'max_iterations':0},[],{'num_envs':True,'max_iterations':801}):
        with pytest.raises(ValueError):MODULE.training_profile({'training_profile':bad},tmp_path)
    assert 'config/interactive_v4_deadline.json' in MODULE.PUBLISH_PATHS


def test_existing_unrelated_staged_change_stays_untouched(tmp_path):
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=tmp_path, check=True,
                              text=True, capture_output=True).stdout
    git("init", "-b", "main")
    git("remote", "add", "origin", "git@github.com:willnien10005914/q1_leg_urdf.git")
    (tmp_path / "README.md").write_text(f"{MODULE.START}\nworking\n{MODULE.END}\n")
    (tmp_path / "unrelated.txt").write_text("Do not stage this with project media")
    git("add", "unrelated.txt")
    finalizer = MODULE.Finalizer(tmp_path)
    with pytest.raises(ValueError, match="Unrelated files"):
        finalizer.preflight()
    assert git("diff", "--cached", "--name-only").splitlines() == ["unrelated.txt"]


def test_upload_process_output_never_appears_in_log(tmp_path):
    finalizer = MODULE.Finalizer(tmp_path)
    finalizer.run([sys.executable, "-c", "print('private-test-token')"],
                  "local private-output test", private_output=True)
    assert "private-test-token" not in finalizer.logfile.read_text()
    assert finalizer.logfile.stat().st_mode & 0o777 == 0o600
    assert all(".gofile.json" not in path and ".candidate" not in path for path in MODULE.PUBLISH_PATHS)
