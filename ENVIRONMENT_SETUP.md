# 環境設定

> 用 uv 把整套跑起來。§0–§1 = 核心(必裝);§2 = Design A 的 SRF(已含在 lock);§3 = Design B 的 6S
> (選裝、有坑);§4 = 資料;§5 = 授權。

---

## 0. 一鍵建置(GPU)

> **論文數字的硬體 vs 目前的開發機。** 論文 v1.1.0 的數字是在 **2×Tesla V100-SXM2-32GB**(NVIDIA NGC
> PyTorch 容器、cgroup 限 8 CPU 核、host 驅動 535)上產生的;目前的開發機是 **4×Tesla P100-PCIE-16GB**
> (VM、8 CPU、host 驅動 535.230.02、PCIe、無 NVLink)。兩台的環境都由 `requirements-lock.txt` 建成(結果
> sidecar 記錄的 torch、numpy、scipy、matplotlib、h5py、pyspectral 版本兩台相同;lock 在 v1.1.0 之後只多釘了
> 一個 setuptools;Python 本身的 patch 版本不同:3.12.3 與 3.12.15)。從 V100 機換到 P100 機,每個訓練出的模型
> 都不同(只比較過這一對機器;兩台的 GPU、CPU 指令集、驅動與作業系統都不一樣,差異來自哪一項並未分離):在
> P100 上重跑旗艦 phase8R 的 10×10,headline(naive L2A joint risk)是 28.707 ± 0.815,論文是 27.849 ± 0.786;
> phase8R2 兩軸每一臂的 joint risk 與 coverage 平均,與論文的差距都在論文標準誤的 0.69 倍以內。細節見
> `experiments/H10_TRAIN_ONCE_VERIFICATION.md`。**一個 campaign 要在同一台機器上整批重跑**,不要混用兩台
> 機器的格子。
>
> 訓練已**自適應化**(`bandsim/hw.py` + `bandsim/parallel.py`):自動偵測 GPU/核數,把各 seed 的獨立工作
> 併行灑到所有 GPU 與 CPU 核上。什麼條件下結果逐位元相同(實測過的範圍)見 `bandsim/parallel.py` 的
> Determinism 一節。

> **本專案用 [uv](https://docs.astral.sh/uv/) 管理環境**(快、可重現)。環境為**完全隔離**的 venv,所有版本
> **鎖死於 `requirements-lock.txt`**。torch 必須是 lock 裡的 2.6.0+cu124:它的輪子含 sm_60(P100)與
> sm_70(V100),較新的建置已陸續拿掉舊架構(例如 NGC 容器內建的 torch 2.12 就不含 sm_70,V100 跑不動)。

```bash
# 0) 安裝 uv (若尚未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh          # 或 pip install uv

# 1) 用 uv 從 lock 建立「隔離 + 版本鎖死」環境 (torch 由 cu124 索引取得;--torch-backend 需 uv >= 0.6.9)
uv venv --python 3.12 .venv
uv pip sync --torch-backend cu124 --python .venv/bin/python requirements-lock.txt \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
source .venv/bin/activate
source scripts/gpu_env.sh                                # 只有 NGC 這類容器需要 (見下方 libcuda 坑);其他機器上不做事

# 2) 驗證 (應印出本機的 GPU 與核數,例如 P100 機: device=cuda (4x Tesla P100-PCIE-16GB 17GB) | 8 cores, ...)
python -c "from bandsim import hw; print(hw.info())"
python -m pytest                                         # 缺資料集的測試會自動 skip

# 3) 一鍵複現 (自適應併行;缺 .venv 會自動用 uv bootstrap,並自動載入 scripts/gpu_env.sh)
bash reproduce.sh
```

> ⚠️ **uv 禁忌**:絕對不要 bare `uv pip install <pkg>` —— 它解析相依時不受 lock 約束,為了滿足新套件的需求
> 會**升級已裝的套件**,曾因此把 torch 2.6→2.13、numpy 2.1→2.5 升上去,直接弄壞 V100 驅動相容。要加套件請
> **先在 `requirements-lock.txt` 釘好版本再 `uv pip sync`**;只是臨時裝一個 lock 以外的套件(如 §3 的 Py6S),用
> `uv pip install --no-deps <pkg>==<版本>`。另外 `uv pip sync` 會**移除** lock 以外的套件,sync 之後要重裝。

**⚠️ libcuda 驅動坑(NGC 這類容器)**:V100 機的容器把 `LD_LIBRARY_PATH` 指向 CUDA-13 forward-compat
的 `libcuda.595`,但 host kernel 是 535 → 建 CUDA context 直接 `device(s) busy or unavailable`
(NGC/官方 torch 都一樣)。修法在 **`scripts/gpu_env.sh`**(`reproduce.sh` 會自動載入;互動式使用時在
activate 之後手動 `source`):偵測到壞掉的 compat 就改用 host 的 libcuda。它只在容器回報 compat 狀態異常
(`_CUDA_COMPAT_STATUS`)或設了 `BANDSIM_FORCE_SYSDRIVER=1` 時才動作,所以在 P100 VM 這類沒有 compat 層的
機器上不做事。

**榨硬體的旋鈕**（環境變數或各 phase 的 `--jobs/--device`）：
- `BANDSIM_DEVICE=cpu|cuda|auto`（預設 auto→cuda）
- `BANDSIM_WORKERS=N`（併行 seed 工作數；`=1` 為序列參考路徑）
- `BANDSIM_GPU_OVERSUB=K`（強制每張 GPU 疊 K 個 worker；預設依最小那張卡的顯存與每 worker 預算自動決定，預算 8 GB 時 16 GiB P100 與 32 GB V100 皆為 2。實測單一旗艦工作峰值 5,478 MiB、phase8R2 評估 9,090 MiB，調高可能 OOM）
- `BANDSIM_PER_WORKER_GB=G`（上述自動決定所用的每 worker 顯存預算，十進位 GB，預設 8；腳本可自帶較大預算，如 phase8R2 評估階段用 11。設定後兩者都被取代，設太小可能 OOM；inf = 每卡 1 個；NaN、0、負數會被忽略）
- `BANDSIM_THREADS=T`（每 worker 的 torch intra-op 執行緒；spawn 出的 worker 裡 numpy 的 OpenBLAS 執行緒數沿用父程序的環境（父程序的 hw.setup 會把未設定的 OMP_NUM_THREADS 設為匯出的 BANDSIM_THREADS 或全部核心數），不是每個 worker 的預算，見 `bandsim/parallel.py`。CPU 結果隨執行緒數改變，固定它才可比較。穩定的逐位元重現只在 CUDA（實測）或 CPU 每 worker 1 執行緒（小型合成實驗實測）時觀察到；2 執行緒時偶有不一致，見 `bandsim/parallel.py` 的 Determinism）

**GPU 互連**:並行是工作級的(每個工作只用一張 GPU,一張卡可同時跑幾個工作),GPU 之間不傳資料,所以不需要 NVLink —— V100 機
的 NVLink 沒有用到,P100 機只有 PCIe 也不受影響(對 <100k 參數的小模型,獨立工作併行遠勝把單一小模型用
DDP 拆到多卡)。只有未來把**單一大工作**拆到多卡(DDP/model-parallel)時,互連頻寬才重要。

## 1. 核心環境

**用 §0 的 uv 流程即可** —— 所有相依已鎖定於 `requirements-lock.txt`(torch 2.6.0+cu124、numpy/scipy/scikit-learn/matplotlib/pandas/pyyaml/pytest、pyspectral、fvcore、tacoreader/rasterio 等,共 88 套件)。
抽象相依宣告於 `pyproject.toml`。**不要用 conda 或 bare `pip/uv pip install`**——版本必須鎖死(見 §0 的 uv 禁忌:曾把 torch 升到不相容版而弄壞 V100 驅動)。

## 2. Design A（SRF 卷積）所需

`pyspectral`(0.14.3)已在 lock 內,§0 裝好就有;第一次用 `bandsim.srf.pyspectral_srf` 時,它會下載
自己的 RSR 資料庫(ESA Sentinel-2 / USGS Landsat 等的實測光譜響應,需要網路)。
備援 SRF 來源（若 pyspectral 取用不便）：
- ESA Sentinel-2 SRF v4.0（xlsx）
- USGS Spectral Characteristics Viewer（Landsat-8/9 OLI RSR）
- Zenodo RSR dataset：https://zenodo.org/records/3381083
- `jgomezdans/sentinel_SRF`（Py6S 格式）

## 3. ⚠️ Design B 的 6S / Py6S(選裝)

**多數情況用不到 6S。** repo 附有三個透射率表:`data/srf_cache/T_6s_grid.npz`(Indian Pines)正是論文
phase3/phase5/phase9 所用的那份(sha256 與它們 provenance 記錄的 `lut_sha256` 相同);
`T_6s_grid_pavia.npz`、`T_6s_grid_salinas.npz` 是發布後的探索性 HSI 6S 研究(phase8H)用的。其他程式只讀
這些表;只有要**重新產生或稽核**它們時,才需要 6S 本身(`experiments/precompute_6s_table.py`、
`precompute_6s_dataset.py`)。論文的主結果(CloudSEN12)完全不需要 6S。

6S 與 Py6S **都不在 lock 裡**。Py6S 的文件首推 conda;本專案不用 conda,而是走它的手動路線,在 Linux 上
從原始碼編 6SV1.1:

```bash
sudo apt-get install gfortran make                       # 需要 gfortran
mkdir -p ~/.local/src ~/.local/bin
( cd ~/.local/src \
  && curl -fsSL -o 6SV-1.1.tar https://rtwilson.com/downloads/6SV-1.1.tar && tar xf 6SV-1.1.tar && cd 6SV1.1 \
  && sed -i 's/^FC *= *g77 \$(FFLAGS)/FC      = gfortran -std=legacy -ffixed-line-length-none -ffpe-summary=none $(FFLAGS)/' Makefile \
  && make && cp sixsV1.1 ~/.local/bin/sixs )             # 在子 shell 裡編,做完仍在 repo 目錄
# Py6S 在 PATH 上找 `sixs`:~/.local/bin 要在 PATH 裡
# ⚠️ 不要另外加 -O2:在 V100 機上曾觀察到能編過、但每次模擬都以 OutputParsingError + IEEE_INVALID_FLAG 失敗
uv pip install --python .venv/bin/python --no-deps Py6S==1.9.2   # --no-deps 必加 (見 §0 的 uv 禁忌)
python -c "from Py6S import SixS; SixS.test()"          # 應印出 "Results agree, Py6S is working correctly"
```

兩支預算腳本不會覆寫已存在的表(要覆寫須加 `--force`):稽核時用 `--out` 寫到別的檔名,例如
`python experiments/precompute_6s_table.py --out data/srf_cache/T_6s_grid_regen.npz`,再和附上的表比對。

**編完要做物理檢查**(只「能跑」不代表編對):重新產生的表應重現 repo 附的那份,例如 Indian Pines 表在
936.8 nm(水氣吸收帶)的氣體透射率,於 CWV 0.5 / 2.0 / 4.0 g/cm² 時為 0.526 / 0.265 / 0.151,而 1596 nm
(大氣窗口)為 1.000 / 0.999 / 0.999。吸收帶不隨 CWV 下降、或窗口波段會動,就是編壞了。

## 4. 資料

全部放在 `data/` 下(gitignored;只有 `data/srf_cache/` 的三個 6S 表隨 repo 附上):

```
data/
├── indian_pines/   Indian_pines_corrected.mat + Indian_pines_gt.mat        (core 與 physics 層的 HSI 實驗)
├── pavia/          PaviaU.mat + PaviaU_gt.mat                              (Phase 6)
├── salinas/        Salinas_corrected.mat + Salinas_gt.mat                  (Phase 6)
├── cloudsen12/     train/ val/ test/ 各含 L1C_B*.dat、LABEL_manual_hq.dat、metadata.csv;
│                   test/ 另含 L2A_B*.dat(真實 L1C→L2A 轉換)                (flagship 與 reliability 層 ★)
├── cloudsen12_val_secondbench.npz   驗證集第二基準的像素樣本(reliability 層;見下)
├── emit_<區域>/    EMIT L2A RFL / RFLUNCERT / MASK granules (.nc)          (emit 層)
├── emit/           emit_sample.npz:單一 EMIT granule 的像素擷取(見下)
└── srf_cache/      6S 透射率表(隨 repo 附上,§3)                         (physics 層)
```

| 資料 | 來源 | 授權 / 條款 |
|---|---|---|
| Indian Pines、Pavia University、Salinas | GIC (UPV/EHU)「Hyperspectral Remote Sensing Scenes」:https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes (檔名即上表) | 來源頁未載明授權條款 |
| CloudSEN12 high | Hugging Face `csaybar/CloudSEN12-high`,**revision `d97728dd61858a21c7d6bc61385687d328433cba`**(論文所用;各 split 資料夾內的 `.dat` 與 `metadata.csv` 原樣下載,不需轉檔) | CC-BY-NC-4.0(非商業) |
| EMIT | NASA LP DAAC 的 `EMITL2ARFL` V001 granules:先跑 `scripts/emit/dl_emit_ndvi.py`(RFL + RFLUNCERT),再跑 `dl_mask_backfill.py`(MASK);發布後的探索性跨感測器研究另需 `EMITL1BRAD`(`experiments/emit_fetch_l1b.py --go`,含 OBS 約 30 GB)。需 Earthdata 帳號:`dl_emit_ndvi.py` 接受 `EARTHDATA_TOKEN` 或 `~/.netrc`,其餘只讀 `~/.netrc`。這些腳本用 `earthaccess`,它不在 lock 裡:`uv pip install --python .venv/bin/python earthaccess -c requirements-lock.txt`(`-c` 讓 lock 內的套件維持原版本;之後的 `uv pip sync` 會把它移除) | NASA 開放資料(產品 DOI 見 `experiments/phase8F_emit.py` 開頭) |

⚠️ **train/test 的 CloudSEN12 只能用上面那個 revision。** CloudSEN12+ 是後來的修訂版,tacofoundation 以 TACO
格式發布的 `tacofoundation:cloudsen12-l1c/-l2a` 就是它(Hugging Face `tacofoundation/cloudsen12` 的 dataset
card:CloudSEN12+ v1.1.0 把舊版標註全部重新整理修訂過,GeoTIFF 以 `discard_lsb=2` 儲存,反射率因此量化成
4 DN 的倍數;`experiments/phase8_cloudsen12.py` 開頭也有說明)。它不是同一份資料換格式;用它建出的 train/test 檔
跑出的數字不能和論文比較。CloudSEN12 `train` 很大(上述 revision 的整個 train 資料夾約 209 GB),只需下載
程式實際讀的檔案(`L1C_B*.dat`、`LABEL_manual_hq.dat`、`metadata.csv`;`test/` 另加 `L2A_B*.dat`;只有不在
`reproduce.sh` 裡的 `phase8R_scenedump_rich.py --val` 另需 `val/` 的 `L2A_B*.dat`)。

唯一的例外是驗證集的第二基準 `data/cloudsen12_val_secondbench.npz`:論文的驗證集重現(`phase8R5_*`)本來就是
用 `experiments/phase8R5_val_loader.py` 從 tacofoundation 的版本抽樣建成的(它需要 `tacoreader<1.0` +
`rasterio`,要另建 venv,不是這個環境;tacofoundation 版的授權是 CC0-1.0,建檔時用的 revision 沒有紀錄)。所以
它的標註是 CloudSEN12+ 的修訂版:在抽樣的像素上和 CloudSEN12-high 比對,535 個 patch 中有 23 個的標註不同
(214,000 個抽樣像素中的 2,815 個,1.3 %;所以至少 23 個 patch 的標註修訂過),見
`experiments/CLOUDSEN12_VAL_LABEL_VERSION_CHECK.md`。

其他要另外取得的輸入:
- `data/emit/emit_sample.npz`:`phase8F_emit.py` 與 `phase8G_emit_reliability.py` 讀的單一 granule 像素擷取
  (5 萬像素);repo 裡沒有產生它的腳本。沒有它時 integrity harness 跳過這兩支,`reproduce.sh` 跳過
  `phase8G_emit_reliability.py`(`phase8F_emit.py` 不在 `reproduce.sh` 裡)。
- DOFA 的程式與權重:`phase8E_dofa.py` 與 `phase8E2_*.py` 第一次執行時,經 torch.hub 從 GitHub
  (`zhu-xlab/DOFA` 的固定 commit)取得程式、從 Hugging Face(`earthflow/DOFA` 的固定 revision)下載權重,
  需要網路。
- 衍生檔由對應腳本產生(各腳本開頭有說明),例如 `data/cloudsen12/test/ACOLITE_*.dat`(第二個大氣校正處理器,
  `experiments/acolite_batch.py` + `acolite_crops_to_dat.py`)。它們需要另裝 ACOLITE:論文用的是
  https://github.com/acolite/acolite 的 commit `d5e4cb0`(LUT `ACOLITE-LUT-202110`,`acolite_batch.py` 選它的
  MOD2 氣膠模型),clone 到 `data/acolite_stage/acolite/`(`acolite_batch.py` 從那裡匯入)。ACOLITE 自己的相依
  套件中,GDAL 的 Python 綁定、netCDF4、pyproj 等(見其 `environment.yml`)本專案的 .venv 沒有:`acolite_batch.py`
  要在另一個環境跑,那個環境還要有裁切步驟用的 `rasterio`(論文當時用系統的 GDAL 3.8.4 與 numpy<2);
  `acolite_crops_to_dat.py` 只用 numpy,在本專案的 .venv 跑即可。

## 5. 授權

本 repo 公開,授權依路徑而定(機器可讀的對照在 `REUSE.toml`):程式碼與設定為 MIT;作者自有文件(本檔、
`README.md`)為 CC-BY-4.0;`paper/` 下的手稿、圖與結果檔為**版權保留**(`LICENSES/LicenseRef-Author-Preprint.txt`),
可以閱讀,但未經作者同意不得再散布或改作。
