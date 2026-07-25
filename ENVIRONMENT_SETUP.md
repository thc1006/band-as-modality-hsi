# 環境設定

> 用 uv 把整套跑起來(§0 = 目前實際訓練環境)。分「核心(必裝,§0/§1)」與「Design B 的 6S(選裝、有坑,§3)」。

---

## 0. GPU 容器（2×V100）一鍵建置 ★目前實際訓練環境

> 這台是 **NVIDIA NGC PyTorch 容器**：2×Tesla V100-SXM2-32GB（NVLink，P2P ~24 GB/s）、
> cgroup 限定 **8 CPU 核**（cpus 18–25）、754 GB RAM、host 驅動 535。訓練已**自適應化**
> （`bandsim/hw.py` + `bandsim/parallel.py`）：自動偵測 GPU/核數，把各 seed 的獨立工作
> 併行灑到兩張 V100 + CPU 核上。實測 phase2 五 seed 100.8s→31.7s（**3.2×**），且與序列版
> **逐位元相同**（見 `docs/review/HARDWARE_AND_REVIEW.md`）。

> **本專案用 [uv](https://docs.astral.sh/uv/) 管理環境**(快、可重現)。環境為**完全隔離**的
> venv,所有版本**鎖死於 `requirements-lock.txt`**(torch 2.6.0+cu124 才支援 V100/sm_70)。
> ⚠️ 容器內建的 NGC torch 2.12 丟棄了 sm_70(Volta)→ V100 跑不動,所以必須用官方 cu124 輪子。

```bash
# 0) 安裝 uv (若尚未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh          # 或 pip install uv

# 1) 用 uv 從 lock 建立「隔離 + 版本鎖死」環境 (torch 由 cu124 索引取得)
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
source .venv/bin/activate                                # 自動載入 scripts/gpu_env.sh (libcuda 修正)

# 2) 驗證 (應印出 2 GPU、kernel 可跑)
python -c "from bandsim import hw; print(hw.info())"
pytest                                                   # 33 passed (3 skipped = 缺6S表)

# 3) 一鍵複現 (自適應併行；缺 .venv 會自動用 uv bootstrap)
bash reproduce.sh
```

> ⚠️ **uv 禁忌**:絕對不要 bare `uv pip install <pkg>` —— uv 會**重解整個相依**,曾把 torch
> 2.6→2.13、numpy 2.1→2.5 升上去,直接弄壞 V100 驅動相容。要加套件請**先在
> `requirements-lock.txt` 釘好版本再 `uv pip sync`**。

**⚠️ libcuda 驅動坑（此容器必踩）**：容器把 `LD_LIBRARY_PATH` 指向 CUDA-13 forward-compat
的 `libcuda.595`，但 host kernel 是 535 → 建 CUDA context 直接 `device(s) busy or unavailable`
（NGC/官方 torch 都一樣）。修法已寫進 **`scripts/gpu_env.sh`**（`source .venv/bin/activate`
與 `reproduce.sh` 都會自動載入）：偵測到壞掉的 compat 就改用 host 的 libcuda 535。健康的機器
（如一般健康的 GPU 機器,無壞掉的 CUDA compat）此腳本為 no-op。手動可 `source scripts/gpu_env.sh`。

**榨硬體的旋鈕**（環境變數或各 phase 的 `--jobs/--device`）：
- `BANDSIM_DEVICE=cpu|cuda|auto`（預設 auto→cuda）
- `BANDSIM_WORKERS=N`（併行 seed 工作數；`=1` 為序列參考路徑）
- `BANDSIM_GPU_OVERSUB=K`（每張 GPU 疊幾個 worker，預設 2；模型極小、顯存幾乎沒用，可調高）
- `BANDSIM_THREADS=T`（每 worker 的 BLAS/intra-op 執行緒；固定它可得逐位元重現）

**NVLink**：兩張 V100 直連 1 條 NVLink（`nvidia-smi topo -m` 顯示 NV1，實測 P2P 24.3 GB/s）。
目前設計是「一個 seed 一張 GPU」的工作級並行，**不走 NVLink**（對 <100k 參數小模型，獨立工作
併行遠勝把單一小模型 DDP 拆兩卡）。NVLink 只在未來把**單一大工作**跨卡（DDP/model-parallel，
如 WHU-Hi 270 頻大面板）時才會用到梯度 all-reduce。

## 1. 核心環境

**用 §0 的 uv 流程即可** —— 所有相依已鎖定於 `requirements-lock.txt`(torch 2.6.0+cu124、numpy/scipy/scikit-learn/matplotlib/pandas/pyyaml/pytest、pyspectral、fvcore、tacoreader/rasterio 等,共 85 套件)。
抽象相依宣告於 `pyproject.toml`。**不要用 conda 或 bare `pip/uv pip install`**——版本必須鎖死(見 §0 的 uv 禁忌:曾把 torch 升到不相容版而弄壞 V100 驅動)。

## 2. Design A（SRF 卷積）所需

```bash
pip install pyspectral      # 內建 Sentinel-2 / Landsat RSR
# 首次使用會下載 RSR 資料；或手動放 ESA S2-SRF v4.0 / USGS OLI RSR
```
備援 SRF 來源（若 pyspectral 取用不便）：
- ESA Sentinel-2 SRF v4.0（xlsx）
- USGS Spectral Characteristics Viewer（Landsat-8/9 OLI RSR）
- Zenodo RSR dataset：https://zenodo.org/records/3381083
- `jgomezdans/sentinel_SRF`（Py6S 格式）

## 3. ⚠️ Design B 的 6S / Py6S（最容易卡的一步）

Py6S 需要 **6S Fortran 執行檔**。**在 Linux 上比 Windows 好裝很多**。三種方式，擇一：

**方式 A（推薦，最省事）— conda 直接裝 6S 二進位：**
```bash
conda install -c conda-forge sixs py6s -y
python -c "from Py6S import SixS; SixS.test()"   # 應印出 6S version
```

**方式 B — 自行編譯 6S：**
```bash
# 需 gfortran
sudo apt-get install gfortran        # 或 module load gfortran（在 HPC）
# 下載 6SV1.1 原始碼，make，並把 sixsV1.1 執行檔加入 PATH
pip install Py6S
```

**方式 C（無法裝 6S 時的退路）— 預製表透射率：**
- 在能裝 6S 的機器上，對參數網格（CWV/AOD/幾何）跑出各波長透射率，存成 `.npz`；
- 本機只讀表、逐頻相乘（`bandsim/atmosphere.py` 的 `load_cached_transmittance()`）。

> 若 Phase 3 卡在 6S，**先跳過**，用 Phase 4R（可靠性）與 Phase 2 撐起論文；6S 是加分不是必需。

## 4. 資料放置慣例

```
data/                      # 皆 gitignored
├── indian_pines/          # Indian_pines_corrected.mat + gt(Phase 1/2/4R)
├── pavia/  salinas/       # PaviaU / Salinas .mat + gt(Phase 6 泛化)
├── cloudsen12/            # 真實 Sentinel-2 雲遮 .dat(train/val/test;Phase 8 ★)
└── srf_cache/             # 6S 透射率表 T_6s_grid.npz(Phase 3/5,選裝)
```
下載連結與授權見 `docs/guide/01_datasets.md`。

## 5. 備註

repo 為 **private**;論文授權為 rights-reserved,勿公開散布 `paper/`(見 `REUSE.toml`)。
