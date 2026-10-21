# 正式 V5 策略錄影入口

`record_lift_v5.sh` 只接受 `checkpoints/isaac/lift_v5/latest.pt`，且須先通過與即時播放器相同的完整 Stage 2、至少 128 回合獨立確認。Stage 0／1 候選與不一致的模型、控制、來源雜湊均被拒絕；沒有候選繞過參數。

```bash
# CPU 檢查；尚未合格會結束且不啟動 Isaac。
bash record_lift_v5.sh --validate-only

# 僅在正式策略已就緒後錄影；不啟動訓練、不上傳。
bash record_lift_v5.sh
# 可指定另一個 checkpoint 目錄以外的 MP4 路徑。
OUT=outputs/confirmed_v5.mp4 bash record_lift_v5.sh --seed 42
```

預設影片與 JSON 為 `docs/results/q1_leg_lift_v5.mp4`、`docs/results/q1_leg_lift_v5.json`。計畫為站立 1 秒、W 前進 2 秒、停止 1 秒、有限左右閃身 4.7 秒、站立 1.5 秒；共 **10.2 秒／510 幀／50 FPS／1280×720**。使用原生 LiftLegEnv、當前 gait/control 與完整腳部視角。過程不重置；失敗立即停止，另存唯一 `.candidate-<id>.mp4/.json`，保留既有正式影片。

正式影片還須符合該回合實際前進 >5 cm、左右閃身的前導與後隨腳均**卸載且離地 ≥2.5 cm**、規劃高擺腿的卸載成功率 ≥40%、最後站立速度 <0.07 m/s。四組方向／角色分別判斷；檢查工具會從逐幀量測重新計算，整段影片的最高值不能替代其中一腳失敗。單段影片的要求不替代原有完整評估。

## 給後續 artifact finalizer 的介面

將 `scripts/reinforcement_learning/rsl_rl` 加入 Python import path 後：

```python
from record_lift_v5 import preflight_recording, verify_artifact
gate = preflight_recording(project_root)  # 未就緒會 raise ValueError
report = verify_artifact(video_path, root=project_root)
```

`verify_artifact` 重新檢查正式 gate、checkpoint／contract／來源與影片 SHA256、錄影程式來源、完整影格數、實際解碼尺寸／FPS，以及扣除文字區後的畫面更新。wrapper 另外傳入每次獨立的 `recording_run_id`，防止 Kit 退出碼為零時誤認舊影片為本次成功結果。finalizer 重用相同且仍合格的既有影片時，可省略 run ID；正式發布應保留 `check_frames=True`。

JSON 核心欄位：

- `accepted`、`gate_passed`、`confirmation_stage=2`、`confirmation_episodes`、`gate_unchanged_after_recording`。
- `checkpoint_sha256`、`contract`、`source_sha256`、`recorder_sources`、`video_sha256`、`recording_run_id`。
- `frames`／`expected_frames`、`actual_duration_s`／`expected_duration_s`、`plan`、`mode_changes`、`root_positions`。
- `failure`、`technical_error`、`error`、`completed_plan`、`final_mode`、`final_sequence_active`。
- `reset_count_after_start`、`auto_resets_suppressed`、`unexpected_episode_reset`。
- `foot_observations`：每幀的時間、mode／gesture phase、原生 CAD 最低腳底高度、承重接觸、參考高度／接觸、腳連桿世界位置與重置計數。
- `foot_metrics`：雙腳最高實際卸載高度、規劃高擺腿成功率、世界位移與路徑。
- `directions.left/right`：各方向的 `lead/trailing_peak_clearance_m`（僅計卸載樣本）、`lift_success_rate`、`high_swing_samples`，驗證時由逐幀資料重新計算。
- `walk_forward_displacement_m`、`final_stand_speed_m_s`。

此入口尚無合格權重可作端到端錄影驗證；目前已通過 CPU gate、計畫、接觸量測 metadata、舊影片拒絕及失敗保留等測試。既有 V4 錄影入口與公開未合格診斷影片均不修改。
