# V5.1 stage-zero recovery and stop-phase review

這是 CPU 與程式審查、主流程完成的原生診斷證據，以及尚待套用到新 run 的訓練課程修正。本次審查沒有修改 actual repo、既有檢查點、done predicate、PD、reward、觀測或驗收門檻；沒有自行啟動 GPU。

## Existing evidence

來源 run：`checkpoints/isaac/lift_v5/runs/20261021_1503_ff_mirror`。下表保留 `evaluation_1910_s0_172910.json` 的起始診斷；最新 2010 更新結果與原生失敗 capture 列於下一段。

| 1910 驗證 | Stand | In-place lift |
|---|---:|---:|
| 完整存活 | 16/16 | 0/16 |
| 平均持續時間 | 20.0 s | 10.096 s |
| 平均 upright | 0.99803 | 0.99800 |
| 左／右腳最高 clearance 平均 | — | 7.62 / 5.45 cm |
| 左／右腳指定窗口卸載抬腳比例 | — | 94.3% / 71.0% |
| 支撐滑移代理 RMS | 0.0295 m/s | 0.0657 m/s |

In-place 的終止時間集中在 10.02–10.24 秒，即 8 秒停止踏步後約兩秒。報告中的高平均 upright 不能證明最後一幀沒有歪斜，也不能證明是何種接觸；它只顯示此前大部分時間相當直立。沒有觀察到 18–20 秒停止窗口的回合，其 `final_stand_speed_m_s=1000000` 是缺資料 sentinel，不是真實速度。

程式的評估 horizon 是 20.5 秒，stand 回合可完整到 20 秒，因此此處不是全域固定十秒 timeout。實際失敗條件仍是 root 高度 <0.52 m、upright <0.65、任一非足部接觸力 >25 N，或非有限 root state；評估也會排除未回報 done 的隱藏 reset。1910 驗證沒有保存最後一幀的各 predicate，不能僅依該份彙總數據宣稱是 ankle contact。

## 2010 native pre-reset evidence

來源 2010 checkpoint SHA-256 為 `053abc84251e9a50ac37fd55d5575f82e888bf8515f2bd1ab2da400f2dd81824`。完整 32 回合評估 `evaluation_2010_s0_173010.json` 仍為 stand 16/16、in-place 0/16；in-place 平均持續時間改善至 13.1887 秒、平均 upright 0.99739，但尚未通過恢復與存活門檻。

主流程另以批次 seed 271100、8 個原生環境（4 stand／4 in-place）執行未改 done／collision 的 pre-reset capture。stand 全數存活，in-place 其中一回合存活、三回合失敗：

| Env | 終止時間 | mode／phase | 原生非足部接觸 | root height／upright |
|---|---:|---|---|---|
| 5 | 13.48 s | STAND／0.7777548 | right_ankle_pitch_link，62.16 N | 0.84076 m／0.98959 |
| 7 | 13.56 s | STAND／0.7777548 | right_ankle_pitch_link，82.91 N | 0.84064 m／0.98969 |
| 3 | 14.34 s | STAND／0.7777548 | right_ankle_pitch_link，86.92 N | 0.84067 m／0.98966 |

三個終止均為 `nonfoot_contact_over_25n=True`，height、upright 與 nonfinite predicates 全為 false，且不是 timeout。當時右 ankle-pitch 原始 CAD 最低點距地約 9–11 微米，符合接地證據；root 也比 0.86 m 參考低約 1.9 cm。這證明這三次的原生終止來源是恢復站立時踝部接地，沒有證明 phase mismatch 是唯一根因。資料及完整原始診斷的 hash 見 [精簡原生診斷](lift2010_recovery_failure_summary.json)。

## Training/evaluation phase mismatch

凍結版本的 stage-zero 訓練，每回合固定踏步 5.5 秒，訓練 horizon 為 8 秒。`gait_phase` 只在 WALK 增加，但 STAND 的 110 維觀測仍包含該相位的 sin/cos。

- 固定訓練停止相位：`(5.5 / 0.9) mod 1 ≈ 0.1111`，sin/cos 約 `[0.6428, 0.7660]`。
- 評估踏步 1–8 秒，共 7 秒；停止相位：`(7 / 0.9) mod 1 ≈ 0.7778`，sin/cos 約 `[-0.9848, 0.1736]`。
- 純 stand scenario 的 phase 為零。它能通過不代表策略看過 phase 0.7778 的停止恢復狀態。
- Mirror loss 只在 WALK 對 sin/cos 取負；STAND 相位保持原值，因此它不會自動補齊這個未見的停止條件。

CPU 對照使用不可變的 `chunks/0001910_s0_r1/resume.pt`，固定其他中立站立觀測，只換上述 phase。確定性 actor 輸出的平均絕對差為 0.1073，最大差 0.6312；以 stance residual 0.35 rad 計，目標角差可達 0.2209 rad。右 ankle Roll action 由訓練停止相位約 `+0.2875` 變為評估停止相位 `-0.3436`。

這是相位敏感度的反事實輸入測試，並非 native rollout，不能單獨證明終止原因。它支持「停止相位與長時間恢復不在訓練分布中」的診斷方向。完整來源 hash、輸入定義及輸出見 [CPU probe JSON](lift1910_stand_phase_cpu_probe.json)。

## Minimal proposed correction

Workspace 的 [patch](lift_v5_recovery20.patch) 只更動 `LiftLegEnv._training_schedule()` 的 stage-zero 分支：重用 base reset 已抽出的 `walk_duration∈[3.5,5.5]`，在該分支局部映射為 `lift_duration = 3.5 + (walk_duration - 3.5) * 2.5`，得到 3.5–8.5 秒的踏步長度。

不修改 reset、mode 切換控制、gait、觀測、推論、PD、reward、done 或其他 stage。新 profile `config/lift_v5_recovery20.json` 只把 stage-zero training episode 從 8 秒延長至 20 秒；其他 stage 仍為 8／20 秒。

開始時間原本為 0.3–0.7 秒，因此最長 8.5 秒踏步後仍有至少 10.8 秒站立。訓練若抽到評估的 7 秒踏步長度，則至少有 12.3 秒恢復，覆蓋評估停止後的十二秒。隨機長度讓停止相位覆蓋整個週期；它不要求改成新的 inference phase convention。

CPU 測試以真實 scheduling method 的 AST 建立 tensor harness，沒有載入 Isaac Sim。已檢查七秒 bout 恰為 350 個 control steps、停止相位與評估一致且具十二秒以上恢復；4096 組抽樣覆蓋 16 個 phase bins，每 bin 至少 148 個樣本，並涵蓋評估停止相位；stand-only、stage-one finite timing 與 stage-two delegation 維持既有行為。四項測試通過。

## Apply as an explicit new experiment

既有 run 已寫完 2010 checkpoint／32 回合評估並停止，主流程已保留完整 source snapshot；另完成上述 8-env pre-reset 診斷。保留舊 run 全部資料，不更動其 source snapshot。

確認要測此課程後，將 workspace patch 與新 profile 套用為新 source revision，建立新的 output run。使用 `--fork-policy` 從實際 serialized iteration=2010 的來源繼承完整 actor 與累計預算；此旗標會刻意重建 critic、Adam 和 exploration std，需如實記在 lineage，不能宣稱完整 optimizer resume。

```bash
bash train_lift_native.sh \
  --training-profile config/lift_v5_recovery20.json \
  --checkpoint SOURCE_2010/resume.pt \
  --completed 2010 --iterations 5 --fork-policy \
  --curriculum-stage 0 --num_envs 10240 \
  --output-dir NEW_RUN --headless
```

這個啟動檢查只新增五次更新，累計為 2015，15000 上限不變；它尚不足讓每個環境經過完整 20 秒訓練回合。後續由 supervisor 接管新 run，捕捉新 source manifest，按原門檻評估。兩個原地抬腳階段驗證通過才可前進到下一課程；最終仍需 stage-two 鍵盤／方向／有限動作驗收及獨立 128 回合確認。不得沿用舊 run 的合格次數或將 stage-zero 成果當作完整使用者控制已完成。
