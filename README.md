# Band-as-Modality — 物理化缺頻穩健性測試平台

> **Band-as-Modality Learning for Multispectral & Hyperspectral Segmentation**
> ＋可重用的物理化缺頻（missing-band）穩健性測試平台（`bandsim`）。
> 公開 repo：`thc1006/band-as-modality-hsi`；論文發布版為 tag **`v1.1.0`**（論文原始檔：`paper/latex/main.tex`;
> repo 裡的 `paper/latex/main.pdf` 是較早的編譯,落後 release 的兩個修正 commit,見 `paper/latex/README.md`）。
> 論文 v1.1.0 的數字在 2×Tesla V100-SXM2-32GB(NGC 容器)上產生;目前的開發機是 4×Tesla P100-PCIE-16GB。
> 兩台的環境都由 `requirements-lock.txt` 建成(v1.1.0 之後 lock 只多釘了 setuptools;Python 的 patch 版本不同);
> 換機器時結果的差異見 `ENVIRONMENT_SETUP.md` §0。

---

## 專案結構

```
.
├── bandsim/            物理模擬器套件(SRF/大氣/卷雲/雜訊/分組/模型/hw/parallel/pipeline)
├── experiments/        各 phase 執行腳本(phase1–9 與 phase8R 系列;Phase 0 是 experiment_synthetic_multiseed.py)
│                       + 驗證 harness(integrity/regression/adversarial)
├── tests/              單元測試(把關物理合理性與實驗協定)
├── configs/            YAML 實驗設定
├── scripts/            gpu_env.sh(容器 libcuda 驅動修正)、doctor.py(repo 與機器健康檢查)、
│                       verify_h10.py、emit/(EMIT 下載與前檢)、2026-07-20 campaign 腳本
├── paper/              論文:latex/(main.tex、refs.bib)、figs/、tables/、results_*(結果檔與
│                       .provenance.json)、RESULT_MANIFEST.md(論文主要數字 → 腳本/紀錄/CSV/seed)、
│                       p100_rerun_2026-10-13/、p100_rerun_2026-10-14/(P100 上的重跑,方法學檢查的
│                       證據;不是論文的數字)
├── data/               資料集(gitignored;只附上 srf_cache/ 的 6S 透射率表;來源見 ENVIRONMENT_SETUP.md §4)
├── pyproject.toml      打包 + uv 設定(cu124 torch 索引)
├── requirements-lock.txt   完全鎖定的環境(88 套件;torch 2.6.0+cu124,含 sm_60/sm_70:P100/V100)
├── reproduce.sh        一鍵複現(分 tier;缺 .venv 自動用 uv bootstrap)
└── README / LICENSE / LICENSES/ / REUSE.toml / ENVIRONMENT_SETUP.md
```

> **環境重建(uv 管理、版本鎖死)**:
> ```bash
> uv venv --python 3.12 .venv
> uv pip sync --torch-backend cu124 --python .venv/bin/python requirements-lock.txt \
>     --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
> ```
> 詳見 [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)(含容器 libcuda 坑與 uv 禁忌)。
>
> **授權**:REUSE 三分法(`REUSE.toml`:程式碼 MIT / 作者文件 CC-BY-4.0 / `paper/` 手稿版權保留)。

---

## 0. 這包是什麼、怎麼用

一句話:**不要用隨機丟頻,改用物理前向模型(SRF/大氣/卷雲/雜訊)生成「缺/壞頻」資料,壓測 band-as-modality 的穩健性,並加上「缺頻下的可靠性/棄答」評估。**

**閱讀順序**:
1. `README.md`(本檔)— 導覽 + 快速開始
2. `ENVIRONMENT_SETUP.md` — 用 uv 把環境裝好(含容器 libcuda 驅動坑、6S 選裝、資料集來源)
3. `paper/latex/main.tex` — 論文 v1.1.0(Sentinel-2 雲遮在真實 L1C→L2A 轉換下的 conformal 可靠性;編譯方式見同目錄 README)
4. `paper/RESULT_MANIFEST.md` — 論文主要數字由哪支腳本、哪個紀錄或結果檔產生,以及與已提交結果比對後的已知落差
5. `reproduce.sh` — 各 tier 的正式重跑指令(開頭有 tier 說明)
6. `experiments/H10_TRAIN_ONCE_VERIFICATION.md` — 旗艦實驗與兩個部署軸(phase8R2)「每個模型 seed 只訓練一次」的驗證,以及 P100 與論文(V100)數字的比較
7. `experiments/PHASE8R2_NORM_STATS_CHECK.md`、`PHASE8R_TEST_PIXEL_SEED_CHECK.md` — 兩項方法學檢查(部署軸的標準化統計量、測試像素抽樣的不確定度),證據在 `paper/p100_rerun_2026-10-13/`;`PHASE8R10_QUANTILE_CONVENTION_CHECK.md`、`CLOUDSEN12_VAL_LABEL_VERSION_CHECK.md` — 分位數傳輸的 tie 慣例、驗證集的標註版本,證據在 `paper/p100_rerun_2026-10-14/`
8. `experiments/MODALITY_GENERALITY_NOTES.md`、`EMIT_CROSS_SENSOR_NOTES.md` — 發布後的探索性延伸(不在 v1.1.0 論文內)

**程式骨架**:
- `bandsim/` — 物理模擬器套件 + 自適應硬體層(`hw.py`/`parallel.py`:自動偵測 GPU/核數並行);各模組開頭的 docstring 寫明其物理與限制
- `experiments/` — 各 phase 執行腳本(由 `reproduce.sh` 呼叫);論文主結果是 `phase8R_reliability.py`(旗艦)與 `phase8R2_landcover_reliability.py`(兩個部署軸)
- `tests/` — 單元測試(`python -m pytest` 直接跑)
- `configs/` — YAML 實驗設定

---

## 1. 快速開始

```bash
# 1) 安裝 uv(若尚未安裝),用 lock 建立版本鎖死的隔離環境
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip sync --torch-backend cu124 --python .venv/bin/python requirements-lock.txt \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
source .venv/bin/activate
source scripts/gpu_env.sh                  # 只有 NGC 這類容器需要(libcuda 修正);其他機器上不做事

# 2) 跑通測試 + 驗證硬體
python -m pytest                           # 缺資料集的測試會自動 skip
python -c "from bandsim import hw; print(hw.info())"

# 3) 一鍵複現(自適應併行,自動用滿 GPU/CPU;可指定 tier,見 reproduce.sh 開頭)
bash reproduce.sh
```

---

## 2. 核心優先序(別迷路)

```
Phase 0  合成 PoC + 5 seeds 誤差帶                 ← de-risk 起點
Phase 1  Indian Pines + SVM baseline → Table 1
Phase 2  Design A(SRF)缺頻退化曲線 ★核心
Phase 4R 可靠性 risk-coverage vs 丟頻數 ★誠實王牌
Phase 3  Design B(6S 大氣)穩健性曲線(用 repo 附的 6S 表)
Phase 4  Design C/D ablation
Phase 6  HSI 三件套泛化(Pavia/Salinas)
Phase 7  實測效率(FLOPs/latency/INT8 量化)
Phase 8  ★真實 Sentinel-2 雲遮(CloudSEN12)+ L1C→L2A 自然缺頻
Phase 8R ★論文主結果:CloudSEN12 上 conformal 風險控制在 L1C→L2A 下的無聲失效(+ 8R2 兩個部署軸,
         與 8R3–8R19 的各項控制與分析)
Phase 9  缺的是「哪些」頻段重不重要(物理選頻 vs 隨機)
```

## 3. 三個最重要的提醒
1. **Novelty 要誠實**:band-as-token／波長編碼／分組遮罩**都已被發表**(Panopticon、ChannelViT、SatMAE、DOFA、SEnSeI、HyperspectralMAE;論文的相關研究一節引用其中的 Panopticon、ChannelViT、DOFA、SEnSeI)。真正開放的角度是**缺頻下的可靠性/棄答評估**。
2. **6S / Py6S 在 Linux 好裝**(Windows 難裝);不過 repo 已附上論文所用的 6S 透射率表,只有要重新產生它時才需要 6S(`ENVIRONMENT_SETUP.md` §3)。
3. **防資料洩漏**:HSI 單場景**別**隨機像素抽樣 → 用 disjoint-region split。CloudSEN12 的官方 train/test 切分以
   ROI 為單位,但不是場景互斥(test 中有 12 個 Sentinel-2 產品各跨兩個 test ROI,另有 4 個產品同時出現在 train
   與 test):旗艦把 test 中共用產品的 ROI 合併成 184 個場景連通單元,並丟掉 test 中與 train 共用產品的 4 個 patch。
