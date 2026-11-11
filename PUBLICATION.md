# TokenSim 通訊頻寬實驗發布範圍

程式、設定、Trace 與三批實驗的報告目錄放在 [tokensim-llama7b-tp8-bandwidth-s0 分支](https://github.com/milin-chang/MigrationTax_extend_experiment/tree/tokensim-llama7b-tp8-bandwidth-s0)。原始 log、逐請求結果、來源快照、圖與報告的完整備份放在 [GitHub Release](https://github.com/milin-chang/MigrationTax_extend_experiment/releases/tag/tokensim-llama7b-tp8-bandwidth-s0-2026-11-11)。

## 收錄的實驗

| 實驗 | 輸入／通訊條件 | 發布時狀態 |
|---|---|---|
| `servegen7b-tp8-bw-s0-200-01` | ServeGen 容量篩選子集前 200 筆；50／100／200 GB/s 各三方法，加 Serial reference | 10／10 組完成並通過稽核 |
| `servegen7b-tp8-bw-s0-8000-01` | 同一 ServeGen 子集 8,000 筆；相同十組條件 | 10／10 組完成並通過稽核；保留暫停、復原與中斷嘗試紀錄 |
| `five-stage-llama7b-h100-tp8-s0-01` | Mooncake shifted v1 全集與 ServeGen primary v1 全集；450／128／50 GB/s 與 reference | Mooncake 12／12 組完成；另有 ServeGen 1 組完成、1 組失敗、8 組未執行，共 13／22 組完成 |

五階段的 ServeGen 高頻寬 reference 因控制器偵測 `host_memory_reserve_pressure` 而中止；其部分輸出僅保留供診斷。完成的 ServeGen nvl4／Serial 與未完成的 reference 不形成完整方法比較，因此主要比較表使用已完成的 Mooncake。

三批皆為 seed 0、llama-7b／FP16、H100 SXM、TP8／PP1、4P＋4D，共 64 張模擬 GPU、64-token KV blocks。改變的是 KV 通訊頻寬；HBM 固定 3,352 GB/s，TP 群組內 collective 仍使用固定 NVLink 設定。FP16 operator 缺少相符實測資料時使用 TokenSim 原生 analytical fallback。

## 分支與 Release 的分工

- 分支：TokenSim 與客製化程式、共用方法與執行工具、三批實驗設定、兩份 ServeGen 執行計畫、五階段執行計畫、實際輸入和來源資訊，以及上述三批各自的 `native_research/reports/<實驗名稱>/`。
- Release：各實驗的 `native_research/runs/<實驗名稱>/` 與 `native_research/reports/<實驗名稱>/`，包含原始 log、收據、原生結果、逐請求時間、執行時來源快照、稽核與圖表。
- 五階段封裝另收錄 `native_research/reports/mooncake-four-part-s0-01/`，內有 Times New Roman 字體的 PNG／SVG 圖與圖表 ZIP。
- 其他批次的 runs／reports、所有 probes、已被取代的 `config/archive/`、本機依賴與暫存檔不納入本次發布。本機檔案仍保留。

`configs/` 根目錄的舊基礎設定與 `data_*` catalog 同時被這三批的 frozen plan 納入檔案校驗，因此保留為重現相依資料；這些檔案不代表本次額外發布了 TP1 實驗結果。

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
