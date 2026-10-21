"""The recording issues a finite sequence, then accepts fresh keyboard input."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, str(SCRIPT))
sys.path.insert(0, str(ROOT / "source/leg_humanoid_lab"))
from record_interactive_v4 import recording_plan
from leg_humanoid_lab.interactive_control import KeyboardController


def test_recording_returns_to_stand_then_accepts_w_and_stops():
    plan = recording_plan()
    assert abs(plan[-1]["end"] - 18.7) < 1e-8
    assert all(a["end"] == b["start"] for a, b in zip(plan, plan[1:]))
    controller = KeyboardController()
    last_segment, modes, resume, control = None, [], [], None
    for i in range(round(plan[-1]["end"] / .02)):
        t = i * .02
        segment = next(item for item in plan if item["start"] <= t < item["end"])
        if segment["name"] != last_segment:
            if segment["name"] in ("stop", "final_stop"):
                controller.stop(t)
            elif segment["name"] == "sidestep_sequence":
                assert controller.sequence(t) is True
            elif segment["name"] == "W_resume":
                assert control["mode"] == "stand" and not control["sequence_active"]
            last_segment = segment["name"]
        if segment["name"] in ("W_forward", "S_backward", "A_left", "D_right", "W_resume"):
            accepted = controller.velocity(*segment["command"], t)
            if segment["name"] == "W_resume":
                resume.append(accepted)
        control = controller.tick(.02, t)
        if not modes or modes[-1] != control["mode"]:
            modes.append(control["mode"])
    assert modes.count("sidestep_left") == modes.count("sidestep_right") == 1
    assert len(resume) == 100 and all(value is True for value in resume)
    assert control["mode"] == "stand" and not control["sequence_active"]
