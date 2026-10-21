# Command-driven walking and finite sidestep: v4 design review

此文件記錄已實作的 v4 架構與訓練監控。10240 個原生環境的物理 smoke 和 PPO 更新已能執行；離線 IK、容量測試與數值檢查不代表新版動作已通過驗收。完整驗收門檻以 [V4_ACCEPTANCE.md](V4_ACCEPTANCE.md) 與 `interactive_evaluation.py` 為準。

## Existing behavior and chosen architecture

v3 使用 `DirectRLEnv`、RSL-RL PPO、23 個原生關節動作與 110 維觀測。既有 `walk` 固定追蹤前進 0.2 m/s 的時間查表；速度命令無法由使用者更改，根部獎勵也綁定從世界原點開始的固定路徑。既有閃身步是有限的七秒動作，末尾持續維持最後姿態。直接在網頁接上 WASD 不會產生可靠的方向控制或動作切換。

目前已實作單一 command/phase-conditioned PPO，同時學習站立、命令行走、左右斜前閃身及動作切換；整個 run 共用一個 15000 iterations 上限。控制環境 `InteractiveLegEnv` 繼承 `DirectRLEnv`，保留 23 個動作與 110 維觀測；以 mode、速度命令、相位與關節參考提供任務條件。物理為 200 Hz、控制為 50 Hz，動作是原生 IK 參考＋腳踝 IMU 回授＋0.35 rad PPO residual，最後受原生 joint limits 限制。

機器人／策略合約為 `q1_leg_interactive_v4`，離線 gait library 合約為 `q1_leg_command_v4`。v3 站立權重只作 bootstrap，不能沿用 v3 的驗收結果。分段訓練使用同一個 actor／critic／optimizer 與累計 iteration，沒有另外為 locomotion 和 gesture 各跑一份 15000 次預算。

分開訓練 locomotion／gesture PPO 能減少獎勵干擾，卻需要另外處理切換時的動作分布與平衡狀態。此次採單一 PPO，直接把行走→手勢→站立→再行走放進訓練；仍須以未 reset 的完整流程驗證，不能只因共用 actor 就宣稱交接成功。

## User control and finite behavior

- WASD 產生機器人本體座標的前後、左右速度命令，Q/E 為 yaw；組合命令依三軸正規化幅度限幅。環境速度命令使用 0.18 秒時間常數，mode 切換另用 0.2 秒關節參考餘弦過渡。釋放按鍵、失去焦點、斷線或後端 0.5 秒逾時時，命令歸零；實際停止表現仍由物理評估判斷。
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

低速行走已使用 forward／backward 模板平均作為原地踏步基底，抵銷步幅平移並保留約 4 cm 的抬腳。未使用的步幅權重填入此基底，只有接近零命令的最後 12% 幅度才逐步混回站立；因此一般低速命令不再把整個抬腳高度縮成幾毫米。這是參考目標，實際 PPO 腳底離地仍需量測。

訓練 episode 為 20 秒，走路窗口長度隨機為 3.5–5.5 秒，開始時間亦隨機。每個窗口中途更換命令，覆蓋轉向、反向與放鍵；第二個走路窗口使用相反方向。命令抽樣有 35% 取自 W／S／A／D／左 yaw／右 yaw 六個單軸全幅命令，其餘為平面方向與 yaw 組合；第二段另抽樣反向與歸零命令。混合 scenario 包含行走與手勢、純行走、純手勢及站立，避免只訓練單一固定流程。

## Reward and independent acceptance criteria

目前 reward 已包含平面速度與 yaw 追蹤、upright、高度、雙腳 body-frame 位置、原生 CAD 腳底淨空、接觸時序、負載滑動、擺盪擦地、torque、action rate、手臂姿態，以及後腳支撐且軀幹向斜前移動的推蹬代理項。高度追蹤使用 0.86 m 加上該相位 root offset 的 z。閃身步速度目標來自有限 root offset 的時間導數，沒有強制 root 位置或直接移動機器人。

支撐滑動目前使用 foot body 原點的世界水平速度；heel-rise 時即使 toe 固定，該原點仍會移動，因此它是滑移代理量，沒有宣稱是完整鞋底接觸點速度。推蹬項同樣是接觸負載與軀幹前進代理，沒有量測切向地面力積分。原生 CAD 最低頂點高度與卸載條件用來判斷真正抬腳；前腳及後腳回收、左右方向分開統計。

已實作的驗收包含每類存活率至少 0.875、upright 至少 0.97、最後站立速度低於 0.07 m/s、行走速度 RMSE 低於 0.12 m/s、指定方向位移至少 0.35 m、各腳淨空至少 2.5 cm 及指定峰值窗口卸載成功率至少 40%、支撐腳原點速度 RMS 低於 0.10 m/s；另有 yaw、途中改向、左右前踏落地、跨身手部、後腳回收、推蹬代理及動作後恢復行走等獨立門檻。沒有另外宣稱平均高度誤差或接觸吻合率具備合格門檻。完整數值、時序、平均方式與限制只以 [驗收文件](V4_ACCEPTANCE.md) 為準。

每份評估有八類各占相同數量的 20 秒回合，使用 deterministic actor 與一個新的批次 seed，保留每回合 env index。只統計第一個 episode，跌倒後的自動 reset 不算進原回合。行走測試包含途中垂直改向與放鍵；有限序列包含行走→預備→左右各一下→站立→再向前走。瀏覽器鍵盤／失焦／重複點擊另由 controller 與 web 測試驗證；不能將物理命令回放宣稱為瀏覽器測試。

目前模板的 lateral root sway 振幅仍為 4.5 cm、週期 0.9 秒。完全重現這個幾何 root 擺動會造成約 0.22 m/s 的側向速度 RMS，與直接追蹤使用者平均方向速度、RMSE <0.12 m/s 的要求存在張力。環境沒有強制跟隨 walk 的 root xy；足位追蹤是 soft reward（位置平方誤差尺度 0.008 m²），姿態模仿權重較低，PPO 可透過 residual 減少擺動。此 run 保留模板 SHA，不在訓練中改動 reference；後續評估須同時檢查側向振盪、指定方向位移、速度誤差、雙腳淨空和支撐滑動，不能只看 reward 上升或抬腳外觀，也不因參考擺動而放寬速度門檻。

## Requested budget, monitoring and early stopping

10240 envs × 24 rollout steps = 每個 PPO iteration 245760 transitions。**單一整合 PPO run 的總上限為 15000 iterations、3,686,400,000 transitions**，相當於每 env 7200 秒、全部 env 合計 853.3 天模擬時間。這是總預算，站立、走路與閃身步共享；不是每個階段各自 15000 次。接管同一 run 的先行 PPO 更新時，已完成次數會計入總額，最後一個 chunk 會截斷以避免超支。

`tools/run_interactive_training.py` 已實作每 500 iterations 分段訓練及新批次 seed 的 32 回合驗證，保留完整 optimizer、PPO iteration、checkpoint、source manifest 與逐回合結果。連續三次所有門檻通過後，再執行獨立 128 回合確認；只有完整確認通過才可提早完成、原子發布 `latest.pt`／contract／evaluation，並執行錄影、GIF、GoFile 與 GitHub finalizer。

Plateau 使用同一 reward revision 的最近四份驗證：gate 合格比例最高分相對第一份改善低於 1%，且最後一份仍未通過，才觸發停滯診斷。這個 score 是 gate 合格比例，不是任務成功率。最多啟用兩個預先定義的 reward revision，提高抬腳、速度追蹤與支撐滑動的權重，降低姿態模仿比重，並保留失敗項目及調整原因；驗收門檻不變。Plateau 不視為成功；到 15000 次仍無完整確認則標記 `budget_exhausted`。

新 worker `train_interactive.py` 與 supervisor 已使用完整 optimizer／iteration resume，避免沿用 v3 runner 會清除 optimizer／iteration 的行為。訓練與驗證依序使用同一張 GPU；檢查 child return code、結果檔更新時間、checkpoint ZIP、iteration 數量及 source hashes，避免 Isaac 啟動程式掩蓋錯誤後被誤報成功。訓練中修改固定來源會停止並保留檢查點。若模型合格但錄影／上傳／SSH 發布失敗，保留合格模型並標記 `artifact_error`。

## GPU capacity

10240 份原生模型約有 307200 個 rigid bodies、296960 個原生關節。完整規模 native PhysX smoke 已通過，量得約 6.4 GB GPU 使用量；實際 PPO 更新約 6.9 GB、初期約 2.4 秒／iteration。此結果確認目前 RTX 4090 Laptop 可載入並更新指定規模，沒有宣稱動作已合格。

以初期步速換算，15000 次純 PPO 更新約 10 小時；分段啟動 Isaac Sim、32／128 回合驗證及成果錄製另計，長時間溫度與系統負載也可能改變速度。單份 float32 110 維 observation rollout 約 103 MiB，另有 critic、action、return、activation 與 pinned host contact buffers；長跑仍監控 GPU／CPU RAM、非有限值和 PhysX buffer 警告。訓練時由外層流程停止 live Isaac viewer，錄製與完整驗證不和大型訓練重疊。失敗時不會暗中降低 envs 後仍宣稱使用 10240。

## Rebuild

```bash
.venv/bin/python tools/build_v4_gaits.py
```

完整 FK 診斷、URDF hash 與 builder hash 會記錄在 `config/v4_gaits.json`。這是參考動作驗證；最終動作品質仍以 native PhysX 的新種子 rollout、連續控制測試和錄影判斷。
