# TokenSim 通訊頻寬實驗發布範圍

程式、設定、Trace 與選定實驗的報告目錄放在 [tokensim-llama7b-tp8-bandwidth-s0 分支](https://github.com/milin-chang/MigrationTax_extend_experiment/tree/tokensim-llama7b-tp8-bandwidth-s0)。最初三批 TP8 實驗的原始 log、來源快照與完整備份放在 [GitHub Release](https://github.com/milin-chang/MigrationTax_extend_experiment/releases/tag/tokensim-llama7b-tp8-bandwidth-s0-2026-11-11)。後續 TP 實驗的分支收錄範圍見下節。

## 收錄的實驗

| 實驗 | 輸入／通訊條件 | 發布時狀態 |
|---|---|---|
| `servegen7b-tp8-bw-s0-200-01` | ServeGen 容量篩選子集前 200 筆；50／100／200 GB/s 各三方法，加 Serial reference | 10／10 組完成並通過稽核 |
| `servegen7b-tp8-bw-s0-8000-01` | 同一 ServeGen 子集 8,000 筆；相同十組條件 | 10／10 組完成並通過稽核；保留暫停、復原與中斷嘗試紀錄 |
| `five-stage-llama7b-h100-tp8-s0-01` | Mooncake shifted v1 全集與 ServeGen primary v1 全集；450／128／50 GB/s 與 reference | Mooncake 12／12 組完成；另有 ServeGen 1 組完成、1 組失敗、8 組未執行，共 13／22 組完成 |

五階段的 ServeGen 高頻寬 reference 因控制器偵測 `host_memory_reserve_pressure` 而中止；其部分輸出僅保留供診斷。完成的 ServeGen nvl4／Serial 與未完成的 reference 不形成完整方法比較，因此主要比較表使用已完成的 Mooncake。

三批皆為 seed 0、llama-7b／FP16、H100 SXM、TP8／PP1、4P＋4D，共 64 張模擬 GPU、64-token KV blocks。改變的是 KV 通訊頻寬；HBM 固定 3,352 GB/s，TP 群組內 collective 仍使用固定 NVLink 設定。FP16 operator 缺少相符實測資料時使用 TokenSim 原生 analytical fallback。

## 分支與 Release 的分工

- 分支：TokenSim 與客製化程式、共用方法與執行工具、最初三批 TP8 的設定與報告，以及下節列出的 Mooncake TP 實驗設定與報告。
- Release：各實驗的 `native_research/runs/<實驗名稱>/` 與 `native_research/reports/<實驗名稱>/`，包含原始 log、收據、原生結果、逐請求時間、執行時來源快照、稽核與圖表。
- 五階段封裝另收錄 `native_research/reports/mooncake-four-part-s0-01/`，內有 Times New Roman 字體的 PNG／SVG 圖與圖表 ZIP。
- 上述指定範圍以外的 runs／reports、所有 probes、已被取代的 `config/archive/`、本機依賴與暫存檔不納入分支。本機檔案仍保留。

`configs/` 根目錄的舊基礎設定與 `data_*` catalog 同時被這三批的 frozen plan 納入檔案校驗，因此保留為重現相依資料；這些檔案不代表本次額外發布了 TP1 實驗結果。

## 新增 Mooncake TP 實驗與圖表

以下每批皆有 10 組完成並通過稽核的 Mooncake shifted v1 seed 0 正式條件：Serial inf reference、450／128／50 GB/s 各三方法。各自的 report 內以 `figure.png`／`figure.svg` 提供一張整合圖。

| Run／report 名稱 | 設定資料夾，位於 `config/` | 模擬 GPU 總數 |
|---|---|---:|
| `five-stage-llama7b-h100-tp1-s0-01` | `five-stage-llama7b-h100-sxm80gb-tp1-s0` | 8 |
| `five-stage-llama7b-h100-tp2-s0-01` | `five-stage-llama7b-h100-sxm80gb-tp2-s0` | 16 |
| `five-stage-llama7b-h100-tp4-s0-01` | `five-stage-llama7b-h100-sxm80gb-tp4-s0` | 32 |
| `five-stage-llama7b-h100-tp1-s0-02` | `five-stage-llama7b-h100-sxm80gb-tp1-s0-rerun02` | 8 |

- 以上四批的設定、實際輸入、report Markdown／CSV、圖表與執行／製圖腳本收錄於分支；原始 `runs/` 與來源快照仍保留本機，本次未新增到 Release。
- TP8 原 report 同樣新增 `figure.png`／`figure.svg`，只呈現已完成的 Mooncake 正式條件。
- [CUSTOMIZATION_SPEC.md](CUSTOMIZATION_SPEC.md) 說明客製化範圍、TP／KV 計量、准入與路由、計時及統計定義。
- [TP1／2／4／8 比較圖表](native_research/reports/mooncake-tp1-2-4-8-s0-01/FIGURES.md) 另收錄 11 張 PNG／SVG、數值 CSV 與圖表 ZIP。
- [TTFT 三部分組成圖](native_research/reports/mooncake-ttft-decomposition-tp1-2-4-8-s0-01/TTFT_DECOMPOSITION.md) 收錄 TP1／2／4／8、三方法、450／128／50 GB/s 的時間占比總覽；各 TP report 另附 `ttft-decomposition.png`／`.svg`、數值 CSV、計算說明與驗證紀錄，TP1 重跑亦有獨立一份。TTFT 沿用 `native_prefill_latency_s`，拆為前輪等待、KV 准入等待、准入後至 client 首 token；占比為各段平均秒數除以平均 TTFT。
- 各 TP 的 GPU 總數與 KV 容量不同；affected share 各自使用該 TP 的 Serial inf baseline。TP1 同設定重跑的 120,310 筆配對 request records 相同，未當作新 seed 彙整。

製圖腳本會驗證原始 `runs/` 的逐筆事件與稽核收據。分支中的 CSV 與圖表可直接閱讀；重新執行驗證式製圖還需要對應的本機原始紀錄。既有 Release 僅包含最初三批 TP8 備份，未因本次分支更新而改寫。

## 新增 P／D 時間觀測報告（2026-11-13）

修正 MigrationTax 專用准入分支的 P／D 預算時間戳後，Mooncake shifted v1、seed 0、TP1／4／8 各十條件皆已完成，共 30／30。
本次分支新增三批完整設定、觀測功能的程式與測試、分析／製圖工具與下列報告；既有 Release 保持原有附件。

- [TP1／4／8 圖表與欄位說明總覽](native_research/reports/mooncake-pd-timeline-tp1-4-8-s0-02/PD_TIMELINE.md)
- [TP1：8 GPU](native_research/reports/mooncake7b-h100-tp1-pd-timeline-s0-02/PD_TIMELINE.md)
- [TP4：32 GPU](native_research/reports/mooncake7b-h100-tp4-pd-timeline-s0-02/PD_TIMELINE.md)
- [TP8：64 GPU](native_research/reports/mooncake7b-h100-tp8-pd-timeline-s0-02/PD_TIMELINE.md)

各批包含 `figure.png/svg`（TTFT 與 P／D 等待）、`worker-steps.png/svg`（每步載入、運算與 batch）、`matched-request-timeline.png/svg`（同一請求三方法時間線）、`ttft-decomposition.png/svg`（TTFT 占比），以及統計 CSV、欄位公式、驗證紀錄。
每批的 `request-timelines.zip` 收錄十條件完整逐請求 P／D 時間戳，供比對；只把指定請求的少量逐步／傳輸樣本納入 Git 報告，完整的逐步與傳輸 CSV 保留於本機 `runs/`。
這 30 條件的 `requests.jsonl` 與加觀測前對應實驗逐位元組相同；校驗值見各報告的 `pd-integrity.json`。

TP1 的長 TTFT 主要出現在前輪依賴與 adapter 的 KV 准入等待；短的引擎內 D queue 不足以證明 D 沒有壓力。
TP 改變時 GPU 總數與 KV 容量也改變；本次沒有調低 arrival rate，也沒有重新校準 analytical 計時。
觀測功能以 `--record-timeline` 開啟，輸出欄位定義見 [TokenSim/docs/native-timeline.md](TokenSim/docs/native-timeline.md)。

## Release 資產

| 資產 | 內容 |
|---|---|
| `servegen7b-tp8-bw-s0-200-01.tar.gz` | 200 筆實驗的完整 runs 與 reports |
| `servegen7b-tp8-bw-s0-8000-01.tar.gz` | 8,000 筆實驗的完整 runs 與 reports |
| `five-stage-llama7b-h100-tp8-s0-01.tar.gz` | 五階段現有完整紀錄、報告與 Mooncake 圖表 |
| `servegen7b-tp8-bw-s0-200-01-results.md` | 200 筆結果表，可單獨下載 |
| `servegen7b-tp8-bw-s0-8000-01-results.md` | 8,000 筆結果表，可單獨下載 |
| `five-stage-mooncake-results.md` | 已完成的 Mooncake 比較表，可單獨下載 |
| `mooncake-figures.zip` | Mooncake PNG／SVG 圖，可單獨下載 |
| `MANIFEST.json` | 封裝清單、每個檔案的 SHA-256、稽核清單與來源檔案差異 |
| `VERIFICATION.json` | 發布前重新核對請求數、輸出量、來源快照與封裝內容的結果 |
| `SHA256SUMS` | Release 各附件的 SHA-256 |
| `RESTORE.md`、`RELEASE_NOTES.md` | 下載還原方式與結果說明 |

## 下載與還原

先取得分支，再將附件放在不納入 Git 的目錄。以下命令適用於全新 checkout：

```bash
git clone --branch tokensim-llama7b-tp8-bandwidth-s0 \
  https://github.com/milin-chang/MigrationTax_extend_experiment.git
cd MigrationTax_extend_experiment

mkdir -p native_research/release-assets/download
gh release download tokensim-llama7b-tp8-bandwidth-s0-2026-11-11 \
  --repo milin-chang/MigrationTax_extend_experiment \
  --dir native_research/release-assets/download

cd native_research/release-assets/download
sha256sum -c SHA256SUMS
cd ../../..

tar --skip-old-files -xzf native_research/release-assets/download/servegen7b-tp8-bw-s0-200-01.tar.gz
tar --skip-old-files -xzf native_research/release-assets/download/servegen7b-tp8-bw-s0-8000-01.tar.gz
tar --skip-old-files -xzf native_research/release-assets/download/five-stage-llama7b-h100-tp8-s0-01.tar.gz
```

封裝保留原本的相對目錄；還原後可按照 `Readme.md` 的路徑查看。目前分支已包含三批報告，`--skip-old-files` 會略過 checkout 已存在的檔案，繼續還原原始執行紀錄。Release 內的舊版 `RESTORE.md` 保留發布當時內容；目前分支的還原步驟以本節為準。

## 歷史來源與重新執行

每組結果的 `source-snapshot/` 保存當時實際載入的來源；`condition.json`、controller 的 `plan.snapshot.json` 與 `audit.json` 保留其校驗及稽核證據。原始檔案不因本次發布而改寫。

目前工作目錄的 `migrationtax/reference/metrics.py` 已在實驗後增加統計功能，與歷史 frozen plan 的檔案雜湊不同。Release manifest 會明列這個差異；應依各組來源快照核對當時程式，不能宣稱目前分支逐檔等同於每一組執行時的版本。

原始計畫包含當時的 Python 路徑、工作 ID 與輸出位置。重新執行時，需針對新環境與新輸出目錄建立新計畫，並重新產生來源校驗。既有計畫與輸出會拒絕覆寫；單改 `--run-root` 不足以重跑。
