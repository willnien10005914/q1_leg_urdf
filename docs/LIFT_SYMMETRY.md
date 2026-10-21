# V5 鏡射一致性：幾何可行範圍

實作為 [`lift_symmetry.py`](../source/leg_humanoid_lab/leg_humanoid_lab/lift_symmetry.py) 的 `augment_policy(env, obs=None, actions=None)`，對應本機 RSL-RL 5 PPO 的觀測、動作分次 callback。此模組只轉換訓練資料，不施加身體外力，也不修改驗收門檻。

原生 URDF 關節軸推導的轉換為：腰 Yaw／Roll 反號、Pitch 不變；左右腿及手臂互換且全部反號。隨機原生姿態的腳、手腕、夾爪均符合位置 `S p`、姿態 `S R S`，其中 `S=diag(1,-1,1)`。站姿保持不變，轉換兩次恢復原值。

110 維觀測保留原欄位順序。線速度與重力採極向量鏡射，角速度採軸向量鏡射；命令變成 `(vx,-vy,-yaw)`。只有 WALK 的 sin／cos 同時反號，代表半個步態週期；閃身、站立與恢復的 phase 不變。

**這不是全域精確的物理對稱。** 左右 Shoulder Pitch 限位同為 `[-1.1345,2.1817]`，反號後不能全域保持合法。原生左右質量、質心也有小差異，例如 Hip Yaw link 質量差約 0.0958 kg。因此只供低權重的一致性損失使用：`use_data_augmentation=False`、`use_mirror_loss=True`；不宣稱鏡射後的 PPO transition 具有完全相同的動力學。

每個樣本須同時符合：

- 實際關節位置與參考位置，以及各自的精確鏡射，皆在原生限位內。
- 兩側實際 Shoulder Pitch 絕對值不超過 1.1345 rad。
- 兩側參考 Shoulder Pitch 絕對值不超過 `1.1345−0.35=0.7845` rad，保留最大殘差空間。

不符合的樣本，第二份觀測和動作直接複製原樣；對確定性 actor 而言，該列一致性損失為零。合法鏡射不做裁切，以免破壞幾何證明。觀測 callback 快取可行遮罩，後續動作 callback 必須使用同一批遮罩；缺少快取或 batch 大小不符會拒絕執行。

測試包含隨機 FK 鏡射、限位例外與合法子域、站姿及 involution、觀測欄位／phase、無效樣本零損失、快取錯誤，以及 TensorDict batch metadata。這些檢查不替代原生 PhysX 的獨立動作驗收。
