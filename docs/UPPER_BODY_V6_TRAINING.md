# V6：上半身穩定與受控回手的 20,000 次新訓練

使用一個 command-conditioned **Isaac Lab DirectRLEnv + RSL-RL PPO**，共同處理站立、WASD／QE、左右閃身與恢復站姿。依 2026-10-23 的新授權，建立獨立 V6 run，最多 **20,000 次新 PPO 更新**，**10,240 envs × 24 steps/update**。13115 actor 只作初始化來源，另記 lineage；新 run 計數從 0 開始，舊 15000 run 不覆寫。

## 控制與學習目標

- 保留原生模型、23 actions／110 observations、腿部 PD 800／32 與上半身 PD 70／4。
- 使用[原生對照中 80／80 回合存活的腰臂幅度限制](UPPER_BODY_13115_REVIEW.md)，不採用造成腳踝碰地的加硬 PD。
- 完整保留 13115 actor MLP，包括輸出層；critic、Adam、探索 std 0.04 與課程驗收重新建立。
- 新增胸口傾角、胸口 roll／pitch 角速度、腰臂追蹤，以及揮手收回時的參考速度追蹤。胸口容許 4° 的原動作傾斜，yaw 不列入晃動懲罰。
- 保留所有原有腳底高度、卸載接觸、支撐滑動、速度、轉向與停止目標。沒有根部外力、固定機身或呈現姿態濾波。

先驗證原地抬腳與恢復，再進入有限左右閃身與回手，最後整合方向控制。各階段訓練回合皆為 **20 秒**，讓策略學到動作後的持續站穩。前置階段各須連續兩次 32 回合驗證通過；整合階段須連續三次，再以獨立種子的 **128 回合完整驗證**確認。

新增上半身門檻與原下半身門檻一起檢查：胸口 RMS ≤8°、單回合峰值 ≤25°、每情境回合峰值 P95 ≤20°；原地前置課程峰值與 P95 均 ≤15°。動態胸口 XY 角速度 RMS ≤0.7 rad/s。每次手臂最後 250 ms 檢查對移動參考的追蹤，動作結束後 **0.8–1.5 秒**檢查胸口、回手距離與收回速度，最後 18–20 秒再驗證站姿。每隻手均須達門檻，不以左右平均掩蓋單邊失敗。

早期回位胸口峰值 ≤8°、每手距離 RMS ≤2 cm、手臂速度 RMS ≤0.6 rad/s；晚期每手距離 ≤1.5 cm、胸口 XY 角速度 RMS ≤0.35 rad/s、手臂速度 RMS ≤0.4 rad/s。所有數值在訓練前宣告於 [控制與 reward 設定](../config/upper_body_control_v6.json)，不因失敗放寬。

## 保存、停滯與來源

前期及前置課程每 100 次驗證一次，整合後每 500 次；每段保留 immutable checkpoint、控制契約、來源 SHA256 及評估。模型停滯不等於成功：四次同階段驗證改善不足 1%，最多套用兩次預先宣告的 reward 加權修訂，增強胸口與受控回手目標，同時保留既有抬腳修訂，並降低 learning rate。上半身新增 reward 最大值依序為除數前 **5.75／6.75／7.5**。碰撞、控制、驗收門檻保持不變。

達上限而未通過即保留未合格候選。只有全部門檻通過的獨立確認才可發布到 `checkpoints/isaac/upper_body_v6`；訓練中記錄不會解鎖正式鍵盤控制。上半身 XY 與 XYZ 角速度分開記錄，不將兩種口徑混用作改善宣稱。

## 指令

```bash
cd ~/projects/q1_leg_urdf
bash start_upper_body_training.sh

# 查詢當前進度；不會啟動另一份訓練。
systemctl --user status q1-leg-upper-training --no-pager
.venv/bin/python tools/run_upper_body_training.py --status
tail -n 40 outputs/upper_body_training.log

# 需要重啟電腦時先停止；之後用相同 run 路徑接續。
systemctl --user stop q1-leg-upper-training
bash start_upper_body_training.sh --resume-run checkpoints/isaac/upper_body_v6/runs/<run-id>
```

原版與保守控制的物理回放仍可透過 `bash start_web.sh` 在本機檢視，不需要同時啟動 Isaac Sim。訓練期間不要另啟即時 Isaac player 搶占 GPU。

啟動前已通過 CPU 控制／預算／初始化／驗收測試及原生 actor 短測：**32／32 第一回合存活 20 秒**、上半身量測有限值、原生抬腳峰值左右平均約 **4.44／4.34 cm**。這是初始化檢查，**不是 V6 訓練完成或完整操作驗收通過**。實際更新數以 supervisor 已保存 checkpoint 與即時狀態為準。

[初始化原生量測與來源](results/upper_body_v6_initial_actor_smoke.json)。正式播放器已支援 V6，驗收未通過時維持鎖定；通過後 `bash start_native_web.sh` 才會優先選用 V6 及對應環境。
