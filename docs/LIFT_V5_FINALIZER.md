# V5 訓練完成後的媒體發布工具

`tools/finalize_lift_v5.py` 位於訓練來源凍結清單之外，獨立等待與處理成果；不修改訓練程式、不開始或恢復 PPO，也不啟動網頁播放器。

```bash
# 單次準備：無網路、不修改 README 或 Git index。
.venv/bin/python tools/finalize_lift_v5.py --once

# 等待訓練完成，再只準備本機影片/GIF。
.venv/bin/python tools/finalize_lift_v5.py --wait

# 執行已授權的 GoFile + GitHub SSH 發布。
# 應先將經審閱的工具程式提交並推送，讓本機 main 與遠端相同。
.venv/bin/python tools/finalize_lift_v5.py --wait --publish
```

可用 `--project-root` 指向部署專案。等待預設每 30 秒查詢一次，`--poll-seconds` 可設 1–60 秒；`--wait-timeout` 為零時不設總等待上限。`--once` 尚未就緒時回傳 2；成功回傳 0；執行失敗回傳 1。工具不自行建立服務，需由操作端安排背景執行。

錄影前必須同時符合：

- 全域及該 run 的 supervisor 狀態均為 `complete`、`ready=true`、Stage 2，且指向同一份發布權重與來源雜湊。
- 正式播放器的原生 128 回合 gate 完整通過；權重、URDF、gait、控制、評估來源與訓練設定全部一致。
- `q1-leg-lift-training`、`q1-leg-training` 已停止，兩個訓練 queue lock 可取得，且沒有 GPU compute／Isaac worker。Xorg 等顯示程序的常駐顯存不會被當成忙碌 GPU。

工具不搶用正在訓練的 GPU、不停止其他程序。若即時播放器仍持有 GPU，需由操作端先停止播放器，工具會繼續等待。

## 執行流程與保留範圍

預設 prepare 模式僅在 `outputs/lift_v5_artifacts/<run+checkpoint識別碼>/` 產生已驗證影片、GIF 與私有狀態。它呼叫 `record_lift_v5.sh`，每個 checkpoint 最多嘗試 seed 42、43、44；失敗影片與量測保留在同一私有目錄。缺少合格權重時完全不錄影。

`verify_artifact` 會檢查完整 510 幀／10.2 秒／50 FPS、扣除文字區後的畫面更新，以及四組「左／右閃身 × 前導／後隨腳」各自卸載 ≥2.5 cm、規劃高擺腿成功率 ≥40%。方向數值由逐幀量測重新計算；不能用整段最高值掩蓋貼地跨步。錄影成功後才轉換完整 GIF。

指定 `--publish` 才會執行：

1. 核對 main、指定的 GitHub SSH origin、既有 staged changes 與遠端 SHA。
2. 上傳已驗證的 MP4 到 GoFile，要求回應 MD5 等於本機 MP4 MD5。
3. 保護上傳回應為 mode 600；回應中的管理 token 不進入 log、README、公開評估或 Git。
4. 複製媒體與量測，替換 README 的 V5 marker 區塊；將先前 V5 狀態保留於歷史 details，V4 等其他區塊不變。
5. 只提交下列六個精確路徑，透過 SSH 推送指定 commit，並確認遠端 main 的 SHA 完全一致。

```text
README.md
docs/results/q1_leg_lift_v5.mp4
docs/results/q1_leg_lift_v5.json
docs/images/lift-v5.gif
docs/results/lift_v5_confirmed_evaluation.json
docs/LIFT_V5_ACCEPTED.md
```

**不提交任何 checkpoint、候選影片、訓練紀錄或 GoFile 私有回應。** V5 確認權重仍保留本機；新 clone 不會因為下載 README 影片就取得策略權重。

## 中斷後續接

狀態為 `outputs/lift_v5_finalization.json`，各 job 另有 mode 600 的 `state.json`；log 也為 mode 600，job 目錄為 700。以 run 和 checkpoint SHA 建立固定身份，同一份已驗證影片／GIF／上傳回應／commit 都可續用。

上傳前會先保存 intent。若中斷後存在相符 MD5 的私有回應，就重用該 URL；若已嘗試上傳卻沒有可驗證的回應，工具停止，避免盲目重複上傳，需先檢查實際上傳結果。已完成 commit 但尚未保存 commit 狀態的中斷，也可由固定訊息、parent 與精確路徑確認並接續。遠端已有其他提交時不強制推送。

上傳未驗證時不改 README；GitHub 只有在遠端 SHA 確認後才標示發布完成。若已完成本機 commit 但推送失敗，狀態保留該 commit 供續接，不宣稱 GitHub 已更新。

CPU 測試使用臨時 Git repository 及外部 subprocess 邊界替身，涵蓋準備模式、GPU／lock 等待、四組腳部驗證入口、MD5 與 token 保護、精確 allowlist、重跑與 commit 中斷恢復。尚無正式合格權重可作原生錄影、GoFile 與 GitHub 的端到端實測；測試不冒充訓練成功或真實上傳。
