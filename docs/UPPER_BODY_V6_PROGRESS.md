# V6 訓練監看

更新時間（UTC）：2026-10-24T11:44:04.062282+00:00。

已保存 **17305／20,000 次新 PPO 更新**，10,240 envs。
目前階段：**原地抬腳與回穩**；狀態：`training`；**尚未完成完整驗收**。

保存更新數與檢查項目通過比例均不是任務成功率。只有三次完整整合驗證與獨立 128 回合確認通過，才會開放 V6 正式控制及成果錄影。

最近驗證：17305 次保存點，32 回合，尚未通過本階段。

第一回合存活率：100.0%；下半身門檻：通過。

未達標項目：

- `upper_body.stand.tail.hand_0`
- `upper_body.stand.tail.hand_1`
- `upper_body.inplace.tail.hand_0`
- `upper_body.inplace.tail.hand_1`
- `upper_body.inplace.dynamic.angular_xy`

監看旗標：

- `saved_checkpoint_not_verified`
- `plateau_requires_review`

[機器可讀進度](results/upper_body_v6_progress.json) · [訓練架構與指令](UPPER_BODY_V6_TRAINING.md)

此監看程式只讀取進度與來源，不修改 reward、門檻、模型或訓練程序。初始化失敗由另一個有界的啟動防護處理；手動暫停不會被本監看恢復。
