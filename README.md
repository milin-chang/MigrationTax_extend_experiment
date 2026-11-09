# MigrationTax 延伸模擬實驗

本 repo 公開本次已完成的 TP8、seed 0 實驗結果與結果彙整 CLI。完整原始紀錄保存在 [Release](https://github.com/milin-chang/MigrationTax_extend_experiment/releases/tag/simulator-tp8-seed0-2026-11-10)。

## 實驗範圍

| 項目 | 規格 |
|---|---|
| 模擬器 | SplitwiseSim、TokenSim |
| Trace | Mooncake shifted v1、ServeGen primary v1 |
| 方法 | Serial、adapted PPD、MigrationTax |
| Seed | 0；共 12 組完成結果 |
| 模型 | Llama2-70B；80 層；FP16 權重與 KV |
| GPU | H100 SXM 80GB；TP8／PP1；4P＋4D；共 64 GPU |

結果表：[results.md](native_research/reports/tp8-aligned-s0-shifted-v1-01/results.md)。另提供完整精度 JSON 與 CSV。

## 下載與還原原始紀錄

先 clone 本 repo。在 repo 根目錄使用 GitHub CLI 下載 Release 全部附件：

```bash
gh release download simulator-tp8-seed0-2026-11-10 --repo milin-chang/MigrationTax_extend_experiment --dir release-assets
```

在 Linux／WSL 驗證所有附件，並將兩組封存檔各自按分片編號串流解壓至 repo 根目錄：

```bash
(cd release-assets && sha256sum -c SHA256SUMS)
cat release-assets/simulator-tp8-seed0-2026-11-10-run-part-01.tar.gz.chunk-* | tar -xzf -
cat release-assets/simulator-tp8-seed0-2026-11-10-run-part-02.tar.gz.chunk-* | tar -xzf -
```

還原後為 `native_research/runs/tp8-aligned-s0-shifted-v1-01/`，包含完整原生結果、逐請求紀錄、事件串流、source snapshots、有效設定與稽核收據。manifest 保存全部原始檔案的 SHA-256。

## 重新產生結果表

在 repo 根目錄執行，只需要 Python 3 標準函式庫：

```bash
python3 native_research/report_native_run_markdown.py
```

CLI 重新讀取並核對已完成的 12 組結果，於終端顯示 Markdown，並寫入：

```text
native_research/reports/tp8-aligned-s0-shifted-v1-01/results.md
native_research/reports/tp8-aligned-s0-shifted-v1-01/results.json
native_research/reports/tp8-aligned-s0-shifted-v1-01/results.csv
```

指標包含平均／中位數／P95 TTFT、平均完整生命週期（含寫回）、整批輸出吞吐量，以及平均 TPOT。報告不使用 Emoji，粗體表示同列最佳值；單位與統計分母列於報告。

此 CLI 計算既有紀錄的統計，不啟動模擬。原始封存內保留執行當時的 source snapshots；本 repo 主分支提供彙整工具與結果。

## 統計口徑

TTFT 從 Trace 原始到達至首 token，包含前輪相依與准入等待；生命週期至最後寫回完成。TPOT 為各請求首 token 後平均 token 間隔的算術平均，排除寫回，單輸出請求記為 0。整批輸出吞吐量的觀察窗從時間 0 至最後寫回完成，包含到達空檔。

僅使用 seed 0，沒有跨 seed 彙整。完成與資料完整性核對不代表實機計時模型已校準；兩套模擬器仍保有各自的計時、排程與准入假設。
