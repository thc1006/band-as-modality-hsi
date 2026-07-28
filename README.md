# Band-as-Modality — 物理化缺頻穩健性測試平台

> **Band-as-Modality Learning for Multispectral & Hyperspectral Segmentation**
> ＋可重用的物理化缺頻（missing-band）穩健性測試平台（`bandsim`）。
> 私有 repo：`PanNavi23/band-as-modality-hsi`。
> 目前訓練環境：NVIDIA NGC 容器(2×Tesla V100-SXM2-32GB，uv 管理、版本鎖死)。

---

## 專案結構

```
.
├── bandsim/            物理模擬器套件(SRF/大氣/卷雲/雜訊/分組/模型/hw/parallel/pipeline)
├── experiments/        各 phase 執行腳本(phase0–8)+ 驗證 harness(integrity/regression/adversarial)
├── tests/              單元測試(把關物理合理性)
├── configs/            YAML 實驗設定
├── scripts/            gpu_env.sh(容器 libcuda 驅動修正)
├── paper/              論文:main.tex/main.pdf + figs/ + tables/ + results_*.csv/tex
├── docs/               專案文件(見 docs/README.md 索引)
│   ├── guide/          開發路線圖與各階段計畫(00_ROADMAP 為主)
│   ├── submission/     ICAIMS 投稿作者說明(.md)
│   ├── status/         進程表:STATUS_REPORT、NEXTSTAGE_STATUS
│   ├── worklog/        工作日誌:DEV_LOG
│   └── review/         審查與決策:ADVERSARIAL_REVIEW、HARDWARE_AND_REVIEW、REVIEW_NOTES、NOVELTY_AUDIT
├── archive/            早期封存(provenance):早期報告、單檔版 experiment.py、v1 路線圖、研究傾印
├── data/               資料集(gitignored;Indian Pines/Pavia/Salinas/CloudSEN12,見 docs/guide/01_datasets.md)
├── pyproject.toml      打包 + uv 設定(cu124 torch 索引)
├── requirements-lock.txt   完全鎖定的環境(85 套件;torch 2.6.0+cu124 for V100/sm_70)
├── reproduce.sh        一鍵複現(缺 .venv 自動用 uv bootstrap)
└── README / LICENSE / LICENSES/ / REUSE.toml / CITATION.cff / ENVIRONMENT_SETUP.md
```

> **環境重建(uv 管理、版本鎖死)**:
> ```bash
> uv venv --python 3.12 .venv
> uv pip sync --python .venv/bin/python requirements-lock.txt \
>     --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
> ```
> 詳見 [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)(含容器 libcuda 坑與 uv 禁忌)。
>
> **文件導覽**:見 [`docs/README.md`](docs/README.md)(依工作日誌／進程表／審查分層)。
> **授權**:REUSE 三分法(`REUSE.toml`:MIT / CC-BY / 手稿版權保留)。

---

## 0. 這包是什麼、怎麼用

一句話:**不要用隨機丟頻,改用物理前向模型(SRF/大氣/卷雲/雜訊)生成「缺/壞頻」資料,壓測 band-as-modality 的穩健性,並加上「缺頻下的可靠性/棄答」評估。**

**閱讀順序**:
1. `README.md`(本檔)— 導覽 + 快速開始
2. `ENVIRONMENT_SETUP.md` — 用 uv 把環境裝好(含容器 libcuda 驅動坑、6S 選裝)
3. `docs/guide/00_ROADMAP.md` — **主路線圖**(4 設計、分階段計畫、論文圖表對應)← 最重要
4. `docs/guide/01_datasets.md` — 用哪個資料集、怎麼拿、陷阱
5. `docs/guide/02_methods_baselines.md` — 要比哪些 baseline、**novelty 怎麼誠實守住**
6. `docs/guide/03_physical_simulation.md` — 4 個物理模擬設計的詳細物理與工具
7. `docs/guide/04_tooling_plan.md` — 最小工具堆 + 分階段執行 + 明確 descope
8. `docs/guide/05_repo_decisions.md` — 歸檔/授權決策 + 公開前 checklist

**程式骨架**:
- `bandsim/` — 物理模擬器套件 + 自適應硬體層(`hw.py`/`parallel.py`:自動偵測 GPU/核數並行)
- `experiments/` — phase0–8 執行腳本(reproduce.sh 呼叫);`phase2_degradation.py` 為核心
- `tests/` — 單元測試(`pytest` 直接跑)
- `configs/` — YAML 實驗設定

---

## 1. 快速開始

```bash
# 1) 安裝 uv(若尚未安裝),用 lock 建立版本鎖死的隔離環境
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
source .venv/bin/activate                 # 自動載入 scripts/gpu_env.sh(libcuda 修正)

# 2) 跑通測試 + 驗證硬體
pytest                                     # 33 passed(3 skipped = 缺 6S 表)
python -c "from bandsim import hw; print(hw.info())"

# 3) 一鍵複現(自適應併行,自動用滿 GPU/CPU)
bash reproduce.sh
```

---

## 2. 核心優先序(別迷路)

```
Phase 0  合成 PoC + 5 seeds 誤差帶                 ← de-risk 起點
Phase 1  Indian Pines + SVM baseline → Table 1
Phase 2  Design A(SRF)缺頻退化曲線 ★核心
Phase 4R 可靠性 risk-coverage vs 丟頻數 ★誠實王牌
Phase 3  Design B(6S 大氣)穩健性曲線(需 6S 表)
Phase 4  Design C/D ablation
Phase 6  HSI 三件套泛化(Pavia/Salinas)
Phase 7  實測效率(FLOPs/latency/INT8 量化)
Phase 8  ★真實 Sentinel-2 雲遮(CloudSEN12)+ L1C→L2A 自然缺頻
```

## 3. 三個最重要的提醒
1. **Novelty 要誠實**:band-as-token／波長編碼／分組遮罩**都已被發表**(Panopticon、ChannelViT、SatMAE、DOFA、SEnSeI、HyperspectralMAE)。真正開放的角度是**缺頻下的可靠性/棄答評估**(見 `docs/review/NOVELTY_AUDIT.json` 與 `docs/guide/02`)。
2. **6S / Py6S 在 Linux 好裝**(Windows 難裝);Phase 3/5 需先產 6S 透射率表,否則自動 skip。
3. **防資料洩漏**:HSI 單場景**別**隨機像素抽樣 → 用 disjoint-region split;CloudSEN12 用官方 disjoint-ROI 切分。
