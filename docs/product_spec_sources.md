# Motion Store 規格資料契約

產品頁採用動作系列與版本的呈現方式，但沒有把尚未驗證的參考、指令或影片誤稱為可保證的產品性能。

## API 欄位

`/api/store` 的 `catalog` 保留原有 `category=spark|history`，另以 `family=sidestep|stability|inplace` 做產品分類。
每個版本包含 `product_name`、`tagline`、`family_name`、`motion`、`scenes`、`highlights`、`specs` 及 `spec_scope`。

`specs` 的每個項目都使用：

```json
{"id":"battery","label":"續航時間","value":"待量測","unit":"","status":"unmeasured","source":null,"note":"尚無電池與實機量測。"}
```

狀態的意義：

| status | 顯示意義 |
|---|---|
| `source` | 原生模型或明確設定值，並非實機量測 |
| `command` | 控制／參考指令，不能當作實際最高速度 |
| `measured` | 指定歷史原生模擬片段的量測，非整體合格證明 |
| `unmeasured` | 沒有量測，不填造數字 |
| `unverified` | 尚未建立或通過可使用的驗證範圍 |

## 實際來源

- 模型重量：從 `urdf/q1_inertia/quanta_robot_legged.urdf` 的 inertial mass 總和得到 **42.205915473 kg**，是模型質量，不是實機秤重。
- 控制關節：`config/robot_contract.json` 的 ordered action joint names 為 **23 軸**；原生 URDF 有 **29 個可動 joints**，另外六軸在控制契約鎖定。API 同時檢查控制名稱確實存在於來源 URDF。
- `root_height` 是模型 root／骨盆的設定值，**不是機器人整體身高**；本次不將其放入產品外形尺寸。
- 速度：`config/v5_gaits_lift6.json` 的前進指令設定上限 **0.22 m/s**，原地抬腳為零平移指令。這不是閃身速度、平均實速或已驗證的最高速度。
- 歷史片段：V6 終點 2.74 s、V5 左右候選 7.20 s、原地抬腳 8.00 s，分別讀取相鄰 `docs/results/*diagnostic.json` 的 `actual_duration_s`。V6 提前終止，不以原計畫 7.20 s 代替。
- 腰臂修正：使用 `upper_body_13115_experimental_replay.json` 的原生錄影 trace；時間由最後 frame timestamp 取得，胸口 peak 取該 trace 的 metrics。
- 「錄影軌跡包絡」由同份片段 root positions 的 X、Y `max-min` 各自計算。它只涵蓋 root 參考點，**不包括手臂、腳底、全身掃掠範圍，也不是安全工作範圍**。
- 卸載抬腳峰值使用該單一片段 foot metrics 的 left／right unloaded sole clearance。整段的峰值不能證明每一步都成功。

## 尚未量測

電池容量、電量／每動作 Wh、電流、電壓、馬達效率、回生、實機續航與可保證速度均沒有可用量測。
產品 API 的 `battery_capacity_wh`、`endurance_minutes`、`maximum_measured_speed_m_s`、`verified_workspace_m` 均為 `null`，顯示待量測或待驗證。
不能從 GPU 用電、PPO 更新數或關節力矩推算機器人的電池續航。

產品棚、室內工作區、動作展示區是觀看 URDF 的示意場景；場景切換不構成新的碰撞、安全性或物理驗證。

## URDF 展示與產品圖片

`web/store/product-viewer.js` 直接載入同份 URDF／STL。`web/store/assets/*.jpg` 是該展示器的模型截圖，沒有另畫替代機器人。
模型姿態包圍盒由 mesh 幾何計算，UI 標示「目前姿態的模型外形高度」，不把 root 高度當成整機高度。
不同 scene 僅變更視覺地面、網格與展示環境；不提供場景物理保證。
歷史回放使用 x86 13,115-update trace，原地抬腳是 V5 IK 參考；Spark 的產品入口目前只顯示站姿，不會悄悄換成舊 PPO。
