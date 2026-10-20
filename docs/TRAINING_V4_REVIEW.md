# Command-driven walking and finite sidestep: v4 design review

此文件是程式與訓練設計審查。離線 IK 通過不代表 PPO 或物理模擬已通過。

## Existing behavior and chosen architecture

v3 使用 `DirectRLEnv`、RSL-RL PPO、23 個原生關節動作與 110 維觀測。既有 `walk` 固定追蹤前進 0.2 m/s 的時間查表；速度命令無法由使用者更改，根部獎勵也綁定從世界原點開始的固定路徑。既有閃身步是有限的七秒動作，末尾持續維持最後姿態。直接在網頁接上 WASD 不會產生可靠的方向控制或動作切換。

建議共用一個 command/phase-conditioned PPO，依序學習 locomotion，再加入左右斜前閃身及動作切換。保留同一個 actor、同一個機器人物理狀態和相同關節控制介面，容易把行走、站立、觸發手勢和恢復站立一起訓練。新參考動作和獎勵語意使用 `q1_leg_command_v4` 合約；v3 權重最多用於初始化，不能沿用 v3 的驗證結果。

分開訓練 locomotion/gesture PPO 有較少的獎勵干擾，也容易單獨修正模仿品質；代價是切換時兩個策略的動作分布、歷史 action 與隱含平衡狀態可能不一致。若採兩個策略，必須對切換狀態和 0.2–0.4 秒過渡進行訓練、測試。切換時重置模擬器只會掩蓋問題。整合策略必須保持足夠的 locomotion 樣本比例，並監控新增手勢後是否遺忘方向控制。

## User control and finite behavior

- WASD 產生機器人本體座標的前後、左右速度命令；轉向可另外使用 Q/E。速度透過約 0.3 秒的 ramp 改變，釋放按鍵後減速站立。失去視窗焦點、網路逾時或舊指令逾時時，命令應歸零。
- 按下閃身按鈕後，先減速站穩，再執行左前、右前各一次，最後回站姿。手勢執行期間暫存或忽略移動命令，完成後由新的有效按鍵命令恢復控制。
- `sidestep_left` 是右手向左前跨身、左腳向左前微抬落地、右腳維持地面支撐推進，再微抬收回；`sidestep_right` 對稱。所有左右均依機器人自身視角。
- 回站姿保留手勢造成的新位置。單側模板位移約前 8.5 cm、側 8.5 cm，因此左右一組的參考淨位移約向前 17 cm。過程不重置 root、heading、joint state 或 episode。

## Offline native-axis IK library

`tools/build_v4_gaits.py` 以 URDF 原生軸、原生 joint limits 和原始腳底 CAD 頂點建立 `config/v4_gaits.json`；不修改原本 MediaPipe JSON。足部位置使用帶界限的最小平方法，同時約束 foot-link origin 與方向。右、左關節符號由 URDF 讀取，沒有假設所有左右關節只差一個正負號。

每個模板有 61 個 phase 樣本、23 個關節；站立 root 高 0.86 m、支撐腳底約 3 mm。週期步態為 0.9 秒、62% stance、約 4 cm swing lift；前後基底速度 0.22 m/s、左右 0.18 m/s、左右 yaw 0.6 rad/s。斜前閃身每側 1.6 秒，起終關節均為同一個站姿。前腳先抬落地，後腳在 toe 仍支撐時微抬 heel，之後才離地收回；支撐腳的整塊平移不作為「推蹬」。

手臂先求一個可達的跨身 IK 姿態，再沿同一個關節分支平滑往返，避免冗餘肩部 IK 每幀切換解。跨身手在最大幅度時實際位於本體座標約前 22.9 cm、對側 8.0 cm。

| FK 檢查 | Walking | 單側 sidestep |
|---|---:|---:|
| 最高關節速度 | 3.20 rad/s | 5.59 rad/s |
| 最大 foot-link origin IK 誤差 | 0.60 mm | 1.18 mm |
| 左右腳實際最高離地 | 4.29 / 4.29 cm | 4.27 / 4.00 cm（右側對稱） |
| 起終關節最大差 | < 2e-11 rad | 0 |
| 原生角度範圍 | 通過 | 通過 |

資料介面：`plans[name]` 具有 `phase[T]`、`q[T,23]`、`root_offset_body[T,3]`、`foot_position_body[T,2,3]`、`foot_height[T,2]`、`contact[T,2]`、`duration_s`。`foot_position_body` 是足部 link 原點相對 pelvis 的位置。`root_offset_body` 的 z 是相對 0.86 m 的高度變化，xy 是本體起始 heading 下的參考平移／重心擺動。模板名為 `stand`、`forward`、`backward`、`left`、`right`、`yaw_left`、`yaw_right`、`sidestep_left`、`sidestep_right`。`v4_reference.py` 提供無需 Isaac Sim 的 numpy lookup；GPU 環境可一次載入 CUDA 後自行批次內插。

## Reward and independent acceptance criteria

獎勵應分開記錄速度、yaw、upright、height、雙腳位置、實際離地、接觸時序、支撐滑動、非足部碰撞、torque、action rate，以及手腳協調。原本整體姿態模仿與存活獎勵可能讓策略藉雙腳貼地滑動取得高分；新版必須讓兩腳各自的離地與時序成為必要條件。

支撐期懲罰接觸點的世界水平速度，擺盪期追蹤 CAD 腳底最低高度並懲罰負載擦地；禁止用最高單一腳高度替代雙腳判準。heel-rise 會讓 foot-link origin 移動，而 toe 仍能固定，因此滑動最好使用接觸點速度 `v + omega × r`，或明確保留足部原點速度的近似限制。推蹬力必須朝任務移動方向，並與後腳接觸、前腳落地的時間一致。

評估建議使用 deterministic actor 與新種子，逐方向／逐手勢列出結果。基本門檻至少包含：完整 episode 存活率 >= 95%、upright >= 0.94、高度誤差 <= 7 cm、每個命令方向確實移動且速度誤差 <= 0.10 m/s、停止後速度 <= 0.05 m/s；手勢的雙腳都在其指定 swing 峰值離地 >= 2.5 cm、成功率各 >= 80%、接觸時序一致率 >= 75%、負載支撐腳滑動 <= 0.08 m/s，且手勢後能保持站姿並恢復方向控制。命令幅度較小的內插步態需按實際目標高度訂門檻，避免把原地站立或半幅動作與全幅手勢混算。

完整驗證需包含連續流程：站立→前進→停止→左／右橫移→轉向→左右閃身一組→站立→再次 WASD；單獨重置後每個動作通過不能證明此流程可用。訓練用平均 reward 不作為完成標準。

## Requested budget, monitoring and early stopping

10240 envs × 24 rollout steps = 每個 PPO iteration 245760 transitions。15000 iterations 為每階段 **3,686,400,000 transitions**，相當於每 env 7200 秒、全部 env 合計 853.3 天模擬時間。若 locomotion 和整合手勢各使用此上限，總額為 7,372,800,000 transitions。實際 wall time 須由 full-size 基準測試估計。

建議每 500 iterations 保存完整 optimizer、PPO iteration 與 source manifest，使用不同新種子做分項評估。連續三次通過所有門檻且無退步，再以另一組至少 128 episodes 做確認，才允許提早完成與發布。若至少 1500 iterations 的驗證分數提升 < 1%，先保存最佳與最新模型：已過門檻則做獨立確認；未過門檻則檢視 reward 分項、接觸力、clearance 和失敗影片，必要時用新 run/reward revision 繼續。plateau 本身不代表成功。

wheel 專案的 `tools/native_training_queue.py` 已有分段訓練、optimizer/iteration resume、source hash、單 GPU 順序驗證、600 秒無輸出的 watchdog、重試上限與 atomic status。沒有現成的 plateau 判斷；這部分需新增。v3 leg runner 的 checkpoint load 會清除 optimizer 與 iteration，不能直接拿來當分段續訓而不作修改。

## GPU capacity

10240 份原生模型約有 307200 個 rigid bodies、296960 個原生關節，場景與 PhysX 接觸處理比小型 MLP 更可能先遇到容量限制。單份 float32 110 維 observation rollout 已約 103 MiB，另有 critic、action、return 和 minibatch activation。原有 6144 envs 約 5.4 GB 顯存的紀錄不能保證 10240 一定可行。

應先以完整 10240 場景執行少量物理步與 PPO updates，檢查顯存峰值、非有限狀態、PhysX contact/patch/aggregate overflow 和步速，再啟動全額訓練。訓練時關閉 live Isaac worker 和 RTX 錄影。contact stream 容量包含 pinned host memory，不能只看 VRAM；場景配置應保留 CPU RAM 餘裕。若 full-size 失敗，不應暗中降低 envs 後仍報稱使用 10240。

## Rebuild

```bash
.venv/bin/python tools/build_v4_gaits.py
```

完整 FK 診斷、URDF hash 與 builder hash 會記錄在 `config/v4_gaits.json`。這是參考動作驗證；最終動作品質仍以 native PhysX 的新種子 rollout、連續控制測試和錄影判斷。
