# 本機執行狀態

快照時間：2026-10-20T20:15:01+08:00。最新狀態請看 http://127.0.0.1:8876/web/ 或執行 `.venv/bin/python tools/status.py`。

正式來源：`BU10_robot_Assets-main.zip`（SHA-256 `67dd0ee5b4129ad60d8edc64a430956e9ffce94c1a5fc24eb0af1d11d1a63ef8`）。30 個原生 leg mesh 齊全，缺件 0；23 個控制關節 / 110 維觀測。

MediaPipe：前七秒 306/306 幀，20 Hz 原生關節 IK。參考動畫可以在網頁選擇「閃身步」。

| 階段 | 評估狀態 | 存活率 | 平均時間 | 身高誤差 |
|---|---|---|---|---|
| 站立 | 通過原生 CAD 門檻 | 100% | 9.98s | 0.0019m |
| 行走 | 尚未通過 | 100% | 9.98s | 0.0077m |
| 閃身步 | 通過原生 CAD 門檻 | 97% | 7.67s | 0.0196m |

目前：`sidestep` / `passed`，後端 Isaac Lab，GPU 環境數：6144。

正式專案：`~/projects/q1_leg_urdf`。網頁服務：`q1-leg-web.service`；即時模擬：`q1-leg-live.service`；訓練服務：`q1-leg-training.service`。服務是本次工作階段的 systemd transient units，不是開機自動啟動。

```bash
systemctl --user status q1-leg-web q1-leg-live q1-leg-training
tail -f outputs/native_curriculum.log
systemctl --user stop q1-leg-training  # 停止整組訓練程序
```

每階段通過才會前進。未通過會保留權重與物理播放記錄；此時不會宣稱閃身步已訓練成功。原生 CAD 模擬不代表實機已驗證。

## 抬腳改善狀態

已完成新目標及 32 回合診斷，尚未啟動微調。舊策略通過原評估門檻，但右腳平均峰值離地僅約 0.4 cm，未通過新增的雙腳抬腳要求。詳見 [抬腳檢查報告](FOOT_LIFT_REVIEW.md)。
