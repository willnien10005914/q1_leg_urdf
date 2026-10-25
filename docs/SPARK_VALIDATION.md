# DGX Spark 執行驗證（2026-10-25）

`~/will` 原先沒有 `q1_leg_urdf`；已使用登入的 GitHub HTTPS 帳號 clone 到
`/home/spark/will/q1_leg_urdf`。此機預設 `git@github.com` SSH 驗證失敗，HTTPS 可讀寫。

驗證基準為 `main` 的 `aab1f950b5c0e89bd1f2707df3e2198c56503f5d`，工作分支為
`spark/arm64-compat-20261025`。本次不需要修改執行程式；既有 `scripts/isaac_env.sh`
已包含 Spark 的 pip Isaac Sim 探測與 ARM OpenMP preload。此分支僅保存本機實測與重現方式。

## 環境

| 項目 | 本機實測 |
|---|---|
| 系統 | Ubuntu 24.04.4、Linux aarch64 |
| GPU | NVIDIA GB10、統一記憶體約 128 GB |
| NVIDIA driver | 580.173.02 |
| Isaac Python | `~/isaac/env_isaaclab/bin/python`（Python 3.11） |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | `~/IsaacLab`，套件 metadata 0.54.4 |
| GPU PyTorch | 2.9.0+cu130 |
| RSL-RL | 5.0.1 |
| CPU 工具環境 | repo `.venv`，Python 3.10 aarch64 |
| CPU 工具套件 | PyTorch 2.14.1+cpu、MuJoCo 3.17.0、MediaPipe 1.1.1 |

`bash install.sh` 成功建立 repo 專用 CPU 環境；沒有升降級共用 Isaac 環境。
原生 30 個 mesh 完整、23 actions／110 observations。安裝重新產生的
`robot_contract.json` 只有 JSON key 順序不同，確認內容相同後還原，沒有資產修改。

## 結果

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`：
  **321 passed、1 skipped，30.87 秒**。跳過的是選用 `tensordict` 測試；
  CPU `.venv` 未安裝該套件，共用 Isaac 環境已有此套件。
- 原生 `IsaacLegEnv`：4 envs、24 steps、CUDA、`proxy=false`，觀測形狀 `[4, 110]`。
- V7 `BalanceLegEnv`：從已提交的 V6 20,000-update actor 初始化，
  **32 envs × 2 次 PPO 更新**成功，checkpoint 與迭代結果正常保存。
- V7 原設定 **6,144 envs × 2 次 PPO 更新**成功，共收集 294,912 transitions。
  訓練 loop 的兩次 iteration 為 4.66／3.90 秒，約 31,617／37,855 steps/s；
  這不包含 Isaac 啟動、資產匯入與環境建立時間，也不是長時間效能保證。
- **6,144 envs checkpoint 續訓**：載入 optimizer 與 `completed=2`，追加 1 次 PPO 更新成功，
  最終 `completed_iterations=3`、`accepted=false`，該次 loop 為 4.59 秒。
- 網頁測試在暫用的 `127.0.0.1:18876` 執行：`/web/`、
  `/urdf/q1_inertia/quanta_robot_legged.urdf`、`/api/status` 均回 HTTP 200，完成後停止測試伺服器。

GB10 的 CUDA capability 12.1 高於該 PyTorch wheel 宣告的原生架構上限 12.0，
本機因此出現警告；實際 CUDA 矩陣運算、原生 PhysX 與上述 PPO 更新均成功。
此處結論限於安裝、模擬與有界訓練流程可執行，沒有驗證長時間收斂或策略動作合格。

## 重現

```bash
cd ~/will/q1_leg_urdf
git switch spark/arm64-compat-20261025
bash install.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

# 原生 PhysX，不訓練、不替換已發布權重。
bash train_native.sh --skill stand --num_envs 4 --smoke-steps 24 --headless

# 使用既有 GPU 環境；不在此環境執行 pip 升降級。
source scripts/isaac_env.sh

# 新建本機測試 run；每次重做請換新目錄，以保留舊結果。
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train_balance_v7.py \
  --checkpoint checkpoints/isaac/upper_body_v6_20000/latest.pt \
  --output outputs/spark_v7_6144_smoke \
  --initialize --iterations 2 --envs 6144 --headless

# 載入剛保存的 optimizer 與計數，追加一次 PPO 更新。
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train_balance_v7.py \
  --checkpoint outputs/spark_v7_6144_smoke/resume.pt \
  --output outputs/spark_v7_6144_smoke \
  --completed 2 --iterations 1 --envs 6144 --headless

# CPU 網頁；預設 http://127.0.0.1:8876/web/
bash start_web.sh
```

## 分支與產物隔離

本次未修改 `main`、reward、控制、課程、訓練預算、x86 設定或已發布 checkpoint。
所有測試權重、TensorBoard 與 log 均留在已忽略的 `outputs/spark_*`，沒有提交為正式策略。
未啟動訓練 systemd 服務，也未中止 Spark 原有的 wheel 即時播放器。

後續 Spark 程式修正請繼續在 `spark/*` 分支提交；同步 x86 更新時先 `git fetch origin`，
再於停止本分支訓練後處理合併，勿將正在訓練中的來源直接切換或強推回 `main`。
本次僅完成有界檢查，沒有啟動長時間背景訓練。
