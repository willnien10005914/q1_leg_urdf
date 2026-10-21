#!/usr/bin/env python3
"""Export truthful deadline diagnostics, without simulation, publication or secrets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("Q1_LEG_ROOT", Path(__file__).resolve().parents[1]))
TERMINAL = {"deadline_reached", "budget_exhausted", "runtime_error", "interrupted",
            "complete", "passed", "artifact_error"}
CASE_NAMES = ("stand", "w", "s", "a", "d", "yaw_left", "yaw_right", "sequence")
FORBIDDEN = ("token", "secret", "password", "authorization", "cookie", "current_command")


def cleaned(value):
    if isinstance(value, dict):
        return {key: cleaned(item) for key, item in value.items()
                if not any(word in key.lower() for word in FORBIDDEN)}
    if isinstance(value, list):
        return [cleaned(item) for item in value]
    return value


def number(value, default=0.):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pair(values, scale=1., percent=False):
    if not isinstance(values, list) or len(values) != 2:
        return "未觀測"
    return " / ".join(f"{number(v) * scale:.2f}{'%' if percent else ''}" for v in values)


def metric(value, precision=3):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "未觀測"
    if value >= 1e5:
        return "未完整觀測（失敗懲罰值）"
    return f"{value:.{precision}f}"


def svg_chart(summary):
    cases = summary["evaluation"]["cases"]
    label = "ONGOING — NOT ACCEPTED" if summary["ongoing"] else "ACCEPTED" if summary["accepted"] else "NOT ACCEPTED"
    color = "#15803d" if summary["accepted"] else "#b91c1c"
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">',
           '<rect width="1200" height="650" fill="#f8fafc"/>',
           '<g font-family="DejaVu Sans, sans-serif" fill="#0f172a">',
           f'<text x="30" y="42" font-size="29" font-weight="bold" fill="{color}">{escape(label)}</text>',
           f'<text x="30" y="72" font-size="17">Native PhysX PPO · {summary["num_envs"]:,} envs · {summary["completed_iterations"]:,} saved updates</text>',
           '<text x="30" y="115" font-size="21">20-second survival by command</text>',
           '<text x="620" y="115" font-size="21">Mean peak sole clearance (cm)</text>',
           '<text x="620" y="142" font-size="14">Left / right foot; 2.5 cm threshold; stand excluded</text>']
    survival_x, survival_w = 145, 335
    threshold_x = survival_x + survival_w * .875
    svg.append(f'<path d="M{threshold_x} 145V545" stroke="#64748b" stroke-dasharray="6 5"/>')
    svg.append(f'<text x="{threshold_x-22}" y="568" font-size="13">87.5%</text>')
    peaks = [number(v) * 100 for name, case in cases.items() if name != "stand"
             for v in case.get("foot_peak_clearance_m_per_foot", [])]
    ceiling, lift_x, lift_w = max(5., max(peaks, default=0.)), 755, 335
    lift_threshold = lift_x + lift_w * 2.5 / ceiling
    svg.append(f'<path d="M{lift_threshold} 155V545" stroke="#64748b" stroke-dasharray="6 5"/>')
    svg.append(f'<text x="{lift_threshold-20}" y="568" font-size="13">2.5 cm</text>')
    for index, name in enumerate(CASE_NAMES):
        case = cases.get(name, {})
        y = 175 + index * 45
        survival = number(case.get("survival_rate"))
        fill = "#16a34a" if survival >= .875 else "#ef4444"
        svg.extend((f'<text x="30" y="{y+4}" font-size="16">{escape(name)}</text>',
                    f'<rect x="{survival_x}" y="{y-15}" width="{survival_w}" height="23" rx="3" fill="#e2e8f0"/>',
                    f'<rect x="{survival_x}" y="{y-15}" width="{min(1.,max(0.,survival))*survival_w:.2f}" height="23" rx="3" fill="{fill}"/>',
                    f'<text x="495" y="{y+3}" font-size="15">{survival:.1%}</text>'))
        if name == "stand":
            svg.append(f'<text x="620" y="{y+3}" font-size="15">stand: no lift requested</text>')
            continue
        svg.append(f'<text x="620" y="{y+3}" font-size="16">{escape(name)}</text>')
        for foot, value in enumerate(case.get("foot_peak_clearance_m_per_foot", [])[:2]):
            cm = max(0., number(value) * 100)
            fill = "#16a34a" if cm >= 2.5 else "#f97316"
            svg.extend((f'<rect x="{lift_x}" y="{y-17+foot*14}" width="{lift_w}" height="11" fill="#e2e8f0"/>',
                        f'<rect x="{lift_x}" y="{y-17+foot*14}" width="{cm/ceiling*lift_w:.2f}" height="11" fill="{fill}"/>',
                        f'<text x="1100" y="{y-7+foot*15}" font-size="12">{cm:.3f}</text>'))
    svg.extend(('<text x="30" y="610" font-size="14">Bars summarize the actual first-episode evaluation; passing selected bars is not full acceptance.</text>',
                '</g></svg>'))
    return "\n".join(svg) + "\n"


def markdown(summary):
    evaluation = summary["evaluation"]
    label = "ONGOING / 尚在執行；不是結案報告" if summary["ongoing"] else "通過確認" if summary["accepted"] else "NOT ACCEPTED / 尚未合格"
    lines = [f"# 2026-10-21 九點期限訓練結果：{label}", "",
        f"狀態 `{summary['state']}`；實際 **{summary['num_envs']:,} envs**，保存 **{summary['completed_iterations']:,}／{summary['max_iterations']:,} iterations**。使用者允許依九點期限調整更新預算；停止訓練不代表學會動作。", "",
        f"本表使用 {evaluation.get('episodes',0)} 回合、seed `{evaluation.get('evaluation_seed','unknown')}` 的原生 CAD／PhysX 評估，對應 **{summary['evaluation_completed_iterations']:,} iterations** 快照。檢查項目通過比例 **{number(evaluation.get('score')):.1%}**，不是任務成功率。", "",
        "![實際存活與抬腳量測](images/interactive_deadline_metrics.svg)", "",
        "| 類別 | 存活 | 正確方向位移 m / yaw rad | 速度 RMSE m/s / yaw rad/s | 雙腳峰值 cm（左／右） | 雙腳抬腳成功率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in CASE_NAMES:
        case = evaluation["cases"].get(name, {})
        yaw = name.startswith("yaw_")
        displacement = case.get("signed_yaw_displacement_rad") if yaw else case.get("directional_displacement_m")
        velocity = case.get("yaw_rmse_rad_s") if yaw else case.get("velocity_rmse_m_s")
        peaks = "不要求抬腳" if name == "stand" else pair(case.get("foot_peak_clearance_m_per_foot"),100)
        lifts = "—" if name == "stand" else pair(case.get("foot_lift_success_rate_per_foot"),100,True)
        lines.append(f"| {name} | {number(case.get('survival_rate')):.1%} | {metric(displacement)} | {metric(velocity)} | {peaks} | {lifts} |")
    lines.extend(("", "抬腳門檻為腳底離地 2.5 cm，且卸載的抬腳成功比例至少 40%；每類存活至少 87.5%。數值為第一回合量測，跌倒後資料不納入。未完整觀測的停止／續走等測量以評估器失敗懲罰值記錄，不能解讀成實際速度。", "", "## 左右閃身細節", "",
                  "| 方向 | 前腳峰值 cm / 成功率 | 後腳峰值 cm / 成功率 | 前腳前／側落點 m | 落腳率 |", "|---|---:|---:|---:|---:|"))
    for side in ("left", "right"):
        item = evaluation["cases"].get("sequence", {}).get("directions", {}).get(side, {})
        lines.append(f"| {side} | {number(item.get('lead_peak_clearance_m'))*100:.3f} / {number(item.get('lead_lift_success_rate')):.1%} | {number(item.get('trailing_peak_clearance_m'))*100:.3f} / {number(item.get('trailing_lift_success_rate')):.1%} | {metric(item.get('lead_forward_m'))} / {metric(item.get('lead_signed_lateral_m'))} | {number(item.get('lead_landing_rate')):.1%} |")
    lines.extend(("", "## 未通過的檢查", ""))
    lines.extend(f"- `{name}`" for name in evaluation.get("failed_gates", []))
    if not evaluation.get("failed_gates"):
        lines.append("目前評估未列出失敗檢查；是否接受仍以獨立 128 回合確認及完整發布 gate 為準。")
    lines.extend(("", "完整 cases、checks、failed_gates、來源與權重雜湊見 [結構化結果](results/interactive_v4_deadline_20261021.json)。未合格候選的影片僅用於診斷，不能稱為成功示範；本工具不修改權重、驗收 gate 或 README。", ""))
    return "\n".join(lines)


def summarize(root, state_path, allow_active=False):
    root, state_path = Path(root), Path(state_path)
    state = json.loads(state_path.read_text())
    ongoing = state.get("state") not in TERMINAL
    if ongoing and not allow_active:
        raise ValueError("Training/evaluation is still active; refuse a final report (use --allow-active only for an ONGOING preview)")
    last = state.get("last_evaluation") or {}
    report_path = Path(last["report"])
    if not report_path.is_absolute():
        report_path = root / report_path
    report = json.loads(report_path.read_text())
    if report.get("proxy") is not False or report.get("backend") != "isaaclab":
        raise ValueError("Only the actual native Isaac Lab evaluation may be summarized")
    if not isinstance(report.get("cases"), dict) or not isinstance(report.get("checks"), dict):
        raise ValueError("Native report must contain complete cases and checks")
    profile = state.get("profile") or {}
    accepted = bool(not ongoing and state.get("ready") is True and report.get("passed") is True
                    and report.get("confirmation_episodes", report.get("episodes",0)) >= 128
                    and report["checks"] and all(value is True for value in report["checks"].values()))
    evaluation_keys = ("episodes", "confirmation_episodes", "evaluation_seed", "duration_s", "source", "scope",
        "backend", "proxy", "passed", "score", "gates", "cases", "checks", "failed_gates", "per_episode",
        "urdf_sha256", "gait_sha256", "environment_sha256", "evaluator_sha256", "checkpoint_sha256",
        "push_measurement", "hand_measurement", "reset_between_transitions", "timeline")
    summary = {"schema":1, "generated_at":datetime.now(timezone.utc).isoformat(),
        "state":state.get("state"), "ongoing":ongoing, "accepted":accepted,
        "num_envs":state.get("num_envs",profile.get("num_envs")),
        "completed_iterations":state.get("completed_iterations",0),
        "max_iterations":state.get("max_iterations",profile.get("max_iterations",0)),
        "adopted_worker_iterations":state.get("adopted_worker_iterations",0),
        "started_at":state.get("started_at"), "updated_at":state.get("updated_at"),
        "profile":cleaned(profile), "reward_revision":state.get("reward_revision",0),
        "source_sha256":cleaned(state.get("source_sha256",{})),
        "evaluation_completed_iterations":report.get("completed_iterations",last.get("completed_iterations",0)),
        "evaluation_report_sha256":sha256(report_path),
        "evaluation":cleaned({key:report[key] for key in evaluation_keys if key in report})}
    checkpoint = Path(state.get("run", "")) / "resume.pt"
    if checkpoint.is_file():
        summary["latest_checkpoint_sha256"] = sha256(checkpoint)
    files = (root / "docs/results/interactive_v4_deadline_20261021.json",
             root / "docs/DEADLINE_20261021.md", root / "docs/images/interactive_deadline_metrics.svg")
    contents = (json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n", markdown(summary), svg_chart(summary))
    for path, content in zip(files, contents):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--state",type=Path)
    parser.add_argument("--allow-active",action="store_true")
    args=parser.parse_args()
    paths=summarize(args.project_root,args.state or args.project_root/"outputs/interactive_training_status.json",args.allow_active)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
