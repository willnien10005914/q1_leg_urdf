# V6r1：胸口、接觸與動作保留的修正

2026-10-24。保留原 V6 run 與全部來源快照，於累計 **12,800 次**保存後建立新分支。**20,000 次總上限包含這 12,800 次**；恢復 actor 不會重置已花的預算，也不繼承任何驗收通過紀錄。

## 為何修改

原地抬腳胸口動態 XY 角速度在前 8,000 次的 80 批驗證中從未達到 0.7 rad/s。手臂停止後速度已有改善，但後期腳底滑動與抬腳過高增加；原目標對超過參考高度的抬腳沒有額外成本，點狀 exponential 角速度 reward 又會在短暫大幅晃動時飽和。

12,800 次 actor 的原生對照也揭露動作遺忘：三組阻尼在站立、原地與左閃可完成，但右閃與連續序列皆提前終止。繼續只訓練原地課程，不能當作閃身步已學會。[該保存點原生對照](results/v6_damping_probe_12800.json)。

## 阻尼選擇

使用原先已知會左右閃身的 **13115 actor**，保留原生關節、100 Nm／10 rad/s 限制、腿部800／32、上半身剛性70、肩肘阻尼4及相同腰臂 residual caps，只比較腰部阻尼4／8／12。兩個種子每組共60個20秒回合，平衡站立、原地、左閃、右閃及連續序列。

| 腰部阻尼 | 第一回合完整存活 | 原地胸口 XY 角速度 RMS 平均 |
|---|---:|---:|
| 4 | 59／60 | 約1.00 rad/s |
| **8** | **59／60** | **約0.69 rad/s** |
| 12 | 57／60 | 約0.56 rad/s |

採用8；12雖能進一步降低角速度，卻有更多提前終止。這是有限樣本的物理對照，**不是完整策略驗收**，失敗紀錄保留。[第一批](results/v6_damping_probe_teacher13115.json)、[第二批](results/v6_damping_probe_teacher13115_seed137.json)。沒有外力扶持、固定機身或姿態濾波。

## 新分支

- 恢復完整13115 actor，包括輸出層，重新建立 critic／Adam／探索std0.03；以舊V6的原始12800序列化checkpoint及SHA驗證已花預算。
- 胸口增加分段Huber角速度能量成本，超過0.7 rad/s後仍有連續改善方向；保留原先姿態、追蹤與回手 reward。
- 對超出參考腳底高度1 cm的部分增加成本，並加強實際有載支撐腳的滑動成本；不懲罰正常離地擺腿的前進速度。
- 前置課程保留30%有限左閃、右閃與連續閃身演練，避免單一原地課程再次遺忘動作。正式stage0／1／2驗收的案例、門檻及碰撞條件完全不變。
- 每次來源或控制修改均用獨立run，舊權重與來源快照保留；相同來源續訓則保留critic、Adam、std與計數。

新課程原生短測 **48／48第一回合存活20秒**，觀測涵蓋stand／walk／left／right／recover，左右卸載離地峰值平均約 **5.28／5.22 cm**。[初始化量測與預算繼承](results/v6r1_rehearsal_smoke.json)。這段沒有新PPO更新，不能代替訓練後驗收。

## 持續監看與發布

```bash
cd ~/projects/q1_leg_urdf
bash start_upper_body_training.sh
bash start_upper_body_monitor.sh --publish
```

fork設定下，預設啟動入口只會接續已初始化且設定相符的run；不會因漏寫 `--resume-run` 重新開一份從0計數的訓練。

`q1-leg-upper-progress-monitor` 每30秒檢查來源、保存權重、心跳、驗收及停滯旗標。每1000次保存、階段變動、恢復或終止事件，將公開進度與報告經SSH同步到GitHub；不提交權重、私有路徑或上傳憑證。[持續進度頁](UPPER_BODY_V6_PROGRESS.md)。

`q1-leg-upper-finalizer` 等待三次完整stage2驗證與獨立128回合確認，並等GPU與訓練鎖釋放後，才錄製原生影片。影片本身須再通過左右前導／後隨腳卸載離地、胸口與回手門檻，且沒有重置或提前終止；驗證成功後才上傳GoFile、產生GIF並更新README。上傳MD5、checkpoint SHA、控制與來源均會核對。尚未合格時只等待，不把診斷片段當完成成果。

```bash
systemctl --user status q1-leg-upper-training q1-leg-upper-progress-monitor q1-leg-upper-finalizer --no-pager
tail -n 30 outputs/upper_body_training.log
cat outputs/upper_body_monitor_status.json
cat outputs/upper_body_v6_finalization.json
```
