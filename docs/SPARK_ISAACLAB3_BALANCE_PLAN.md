# Spark 平衡強化與 Isaac Lab 3 驗證

2026-10-25。分支：`spark/isaaclab3-balance-6144-15000`。

本輪先在已驗證的原生 PhysX 上執行 **6,144 envs、15,000 次新 PPO updates**，
平衡 reward 使用原生全身 COM／速度、承重腳支撐代理、軀幹傾斜、失高、非腳底接觸及回站手位。
另於隔離環境驗證官方 Isaac Lab 3／Newton／OVPhysX，不升降級共用 Isaac 環境。
更換引擎是否更快、更接近實機，需要相同 Q1 模型與條件的動態比較，不能由版本名稱推定。

## 本輪訓練

| 設定 | 值 |
|---|---|
| 設定檔 | `config/spark_balance_6144_15000.json` |
| Backend | 現有 Isaac Sim 5.1／PhysX GPU |
| 環境／新更新預算 | 6,144／15,000，約 2,211,840,000 transitions |
| Rollout／epochs／minibatches | 24 steps per env／5／8 |
| 初始化 | 已提交的 V6 20,000-update 完整 actor；critic、Adam、std 重新初始化 |
| PPO learning rate／std／entropy | 5e-5 fixed／0.04／0.0005 |
| 課程 | 50% 四次閃身、20% 單側、20% 原方向控制、10% 站立；四步起始方向左右隨機 |
| 保存 | 先 10 updates，再到 250，其後每 250；最終恰為 15,000 |
| 驗測 | 訓練前、10 updates 後、每 500 與終點；64 envs、獨立 seed、20秒首次回合 |
| 完整驗收 | 四步真實雙腳卸載及存活達標後執行原完整 upper/lower gates；終點必做 |
| 狀態 | `outputs/spark_balance_status.json`，另有每個 run 的 status／events／source manifest |
| 訓練鎖 | `~/will/.spark-training-gpu.lock`；未知 GPU compute 先等待，不中止別的工作 |
| Worker timeout | 90 分鐘；無自動重跑可能已花費的 PPO updates |

歷史 V6 20,000 次與相容性 smoke 的 3 次均另列，**不冒充本輪成果，也不扣抵新的 15,000 次預算**。
既有 x86 supervisor、reward／控制來源、正式 checkpoint 不變；新 worker、subclass、設定與服務均使用 Spark 專名。

既有 V7 supervisor 不能直接在此 Spark 跑：它依賴未提交的 x86 狀態與 13,115 權重、
會永遠等待常駐 viewer，而且 12 分鐘 timeout 不足以完成 500 updates。
新 supervisor 自己凍結來源、原生資產、設定與 bootstrap，並從快照啟動 worker；
後續網站開發或分支工作不改動執行中的訓練來源。來源、控制、runtime、budget、checkpoint hash 變動皆拒絕續訓；每個 worker 前後也核對共用 Isaac Lab 原始碼指紋。

失敗後不清零重來。已提交的 completed counter 僅指原子保存的 `resume.pt`；
log 中的即時 iteration 顯示為未保存進度。若訓練 worker 被中斷且更新無法完整計數，
服務停在 `unsafe_to_auto_resume`，保留 log／權重供核帳，不自動重複消耗預算。
目前保留已驗證可共存的 wheel viewer，其餘不明 compute 工作先等候。

### 修正的驗測盲點

- LRLR／RLRL 均進入訓練，沒有把鏡射 transition 當作完全對稱的真實物理。
- `_get_dones` 在 auto-reset 前擷取 COM、腳底高度、負重、胸口角度與非腳底接觸；第一次 done 後不再累積成果。
- timeout 也算未完成，不以只記錄 terminated 的方式重建「存活」。
- `four_step_survival` 只是存活；另記 `completed_four_step_rate`：四次被命令的閃身，各自兩腳都曾真實卸載且腳底至少 1 cm，並完整存活。
- 1 cm 是四步診斷最低有效動作條件，不取代原本更嚴格的抬腳、滑動、速度、胸口與手臂完整驗收。
- 未自動上架或替換 live controller；`accepted=false` 保持到完整重複驗證與獨立確認、人工審查完成。

原非腳底接觸 25 N、root 高度與直立度失敗條件不放寬。前一步失敗的證據是腳踝接觸超標，
不能只憑該力值宣稱已證明整個身體倒地。COM support capsule 是 reward proxy，不是完整支撐多邊形證明。

### 操作

```bash
cd ~/will/q1_leg_urdf
.venv/bin/python tools/run_spark_balance.py --dry-run
bash start_spark_balance_training.sh

systemctl --user status q1-leg-spark-balance-15000 --no-pager
.venv/bin/python tools/run_spark_balance.py --status
tail -n 40 outputs/spark_balance_supervisor.log

# 明確停止；保留最後保存 chunk 與中斷資訊。
systemctl --user stop q1-leg-spark-balance-15000
# 僅在狀態允許、來源與預算核對完成後顯式 --resume；不會默認重新計數。
```

## 本次程式驗證

全套 CPU 回歸：**340 passed、1 skipped（選用 tensordict），27.00 秒**。
新 Spark worker 原生 PhysX preflight 已完成 32 envs × 2 PPO updates；
另做 8 回合、20 秒首次回合 audit，確認終止前接觸採樣與 timeout／四步完成統計可執行。
該極短 preflight 四步完成率仍為 0，只有站立案例存活，不能稱作訓練成果。
這些 preflight 不計入正式 15,000 次 run；正式 service 會先做來源 V6 基準再開始新更新。

## Isaac Lab 3 隔離實測

官方 [v3.0.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
與 [Spark 安裝條件](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)
支援 aarch64／CUDA 13。使用全新 Python 3.12 環境，保留既有 Python 3.11／Isaac Sim 5.1。
官方 Lab 3 不支援原 Isaac Sim 5.1，不能原地替換。

本機 pin 官方 tag `v3.0.0`、commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`。
該 tag 安裝後 package metadata 讀值是 `28.0.0`，因此以 tag／commit 辨識本次來源。
新環境為 Torch 2.11.0+cu130、Newton 1.6.1、Warp 1.17.0、RSL-RL 5.5.1。

| 實際驗證 | 結果與範圍 |
|---|---|
| Lab 3 Newton MJWarp，官方 Cartpole Direct，32 envs × 2 PPO | 通過，1,024 transitions，checkpoint 保存 |
| Lab 3 kitless OVPhysX，同任務／規模 | 通過，1,024 transitions，checkpoint 保存 |
| Newton 匯入同份 Q1 原生 URDF | 通過，30 bodies，總質量同為 42.205915473 kg；尚未鎖定全部非 action joints 或跑 Q1 控制 |
| 原 Q1 worker 直接套 Lab 3 | 不通過：`--headless` 參數移除；舊 AppLauncher 要求 Kit。需要正式遷移 |

[完整實測 JSON](results/isaaclab3_spark_validation.json)／[Q1 importer 報告](results/isaaclab3_q1_import.json)。
Cartpole 很小、包含不同冷啟動成本，不能拿它的 steps/s 推論 Q1 的效能與物理優劣。

### Q1 遷移與比較順序

1. 以 Lab 3 的 `launch_simulation` 與明確 backend preset 啟動；場景改成新的 declarative lifecycle。
2. 將 robot／sensor `ProxyArray` 轉成 `.torch`，quaternion 由 WXYZ 改 XYZW，寫入改 `*_index`／`*_mask`。
3. 逐名稱固定 23 action joints 與 110-D observation 順序，鎖定其餘 joints，保留 FK、mass、COM、慣量、碰撞比例與 limits。
4. 對齊 0.005 s physics／0.02 s control、PD、速度／扭矩限制、reference 和 normalization。
5. 新版 `net_forces_w` 含 normal＋friction；若要保留原失敗條件語意，要選明確 normal force 並做實測，不直接沿用同名欄位。
6. 先靜態重力／承重與單步落腳，再 32 episodes LRLR／RLRL，最後 128 episodes 確認；記錄第一次失敗時間、腳踝接觸、卸載／抬腳、滑動與軀幹晃動。
7. 同 Q1 設定量測啟動成本、6,144-env PPO transitions/s、記憶體及品質；完成 P→P、N→N、P→N、N→P 對照後再決定下一個實驗。

依據：[遷移指南](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/migrating.html)、
[ContactSensor](https://docs.nvidia.com/isaaclab/concepts/sensors/contact-sensor)、
[跨引擎策略驗證](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/transfer-policies-between-phys-x-and-newton)、
[solver 差異](https://docs.nvidia.com/isaaclab/concepts/solver-differences)。

真機可信度仍需碰撞、自碰撞、摩擦、質量／慣量、致動器延遲及感測器的校準。
現有 Q1 關閉 self-collision、固定摩擦 0.9，僅少量初始 joint noise；本輪不把未量測的任意 domain randomization 冒充校準。
後續建立持出摩擦／延遲／推擾測試與實機測量，再決定具物理依據的隨機範圍。

## 可擴充網站

參考使用者提供的 [Unitree UniStore](https://unistore.unitree.com)，
產品、影片→MediaPipe→retarget→PPO 與版本化成果的完整設計見 [Motion Store 設計](MOTION_STORE_PLAN.md)。
此網站設計與任意新影片 imitation environment 尚待分階段實作；不宣稱目前 V7 可直接學任意影片。
