# Sidestep foot-lift review

This change prepares a revised objective and verifies existing weights. **No long training was started and no checkpoint was replaced.**

## Findings

Native CAD sole vertices and FK show that the existing MediaPipe joint reference reaches about 1.6 cm maximum clearance on the left and 6.7 cm on the right. It also permits substantial horizontal movement while the reference feet are near ground height. The original reward and gates emphasize survival, posture, pose, root path and lateral excursion; they do not require each foot to leave the floor at the intended time.

The existing `model_7800.pt` policy was evaluated against the proposed foot objectives in 32 native Isaac Sim environments, using fresh seed 60000. The evaluation did not publish or replace the existing policy evaluation or replay.

| Diagnostic | Existing policy |
|---|---:|
| Survival | 32 / 32 |
| Mean episode lateral excursion | 35.8 cm |
| Left sole: mean per-episode peak clearance | 3.82 cm |
| Right sole: mean per-episode peak clearance | 0.41 cm |
| Left: successful airborne lift at planned peak times | 4.05% |
| Right: successful airborne lift at planned peak times | 0% |
| Loaded stance-foot speed | 0.0477 m/s |
| Planned contact agreement | 87.2% |
| Proposed stepping gate | Failed |

Raw results: `outputs/foot_lift_diagnostic.json`. Clearance is the minimum world height of original CAD sole vertices, with robot and foot rotations applied. Convex hull vertices give the same minimum support height with less computation. Stance speed uses the foot body origin in world coordinates; it is an approximate slip metric. This test uses a different seed from the original 31/32 confirmation and does not replace that result.

## Prepared changes

`tools/prepare_foot_targets.py` identifies four ankle-separation bursts in the source MediaPipe data. It prepares sixteen alternating robot foot swings, eight per side, with a planned 4 cm peak clearance and single support. These are conservative robot targets derived from source timing, not calibrated human foot heights or an exact reconstruction of source jumps.

`docs/reference/sidestep_foot_targets.json` records source hashes and target timing. `config/sidestep_foot_lift.json` enables the objective only for a future fine-tuning run. Existing reference joint poses, observation layout, actor input and default playback remain unchanged.

The opt-in reward adds clearance tracking, stance/swing contact timing, loaded stance slip and swing scuff penalties. Proposed gates require both feet to reach at least 2.5 cm while unloaded at 65% of planned peak samples, stance-foot speed at most 0.10 m/s, and planned contact agreement at least 70%, alongside the existing survival, root, pose and excursion gates. Standing still, lifting only one foot, or excessive stance sliding cannot pass this profile.

The prepared budget is 6144 environments / 8000 sidestep iterations. It has **not been launched**. Successful new stepping behavior must be established by future training, new-seed evaluation and recording.

## Verification

- Eight Python tests passed, covering original robot contracts, forced budgets, foot schedule, one-sided lift rejection and slide rejection.
- A full 32-episode native evaluation exercised the new reward and measurements without learning or numerical errors.
- SHA-256 checks confirmed the existing checkpoint, published evaluation, training status and original joint reference were unchanged.

Reproduce the diagnostic:

```bash
cd ~/projects/q1_leg_urdf
bash inspect_foot_lift.sh
```

This uses Isaac Sim for evaluation only. It does not train or promote a policy.
