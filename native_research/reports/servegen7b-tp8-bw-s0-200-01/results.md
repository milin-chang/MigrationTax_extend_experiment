# TokenSim：ServeGen 200 筆通訊頻寬實驗

Seed 0；llama-7b fp16；H100 SXM；TP8／PP1；4P＋4D 共 64 GPU；64-token KV block。

已完成並通過稽核：10/10 組。以下只納入完成且稽核通過的條件。

通訊頻寬使用十進位 GB/s、單向；HBM 頻寬固定 3,352 GB/s。高頻寬 Serial 參考為 1e15 bytes/s。

## Affected share：第一個真正由 D 採樣的輸出

| 通訊頻寬 GB/s | Serial % | PPD % | MigrationTax % |
|---:|---:|---:|---:|
| 50 | 67.500000 | 67.500000 | 17.500000 |
| 100 | 7.500000 | 6.250000 | 2.500000 |
| 200 | 1.250000 | 1.250000 | 1.250000 |

## Affected share：使用者首 token

| 通訊頻寬 GB/s | Serial % | PPD % | MigrationTax % |
|---:|---:|---:|---:|
| 50 | 67.500000 | 67.500000 | 17.500000 |
| 100 | 7.500000 | 6.250000 | 2.500000 |
| 200 | 1.250000 | 1.250000 | 1.250000 |

## Decode miss share：原始到達時的 GPU resident prefix

| 通訊頻寬 GB/s | Serial % | PPD % | MigrationTax % |
|---:|---:|---:|---:|
| 50 | 98.750000 | 98.750000 | 98.750000 |
| 100 | 98.750000 | 98.750000 | 98.750000 |
| 200 | 98.750000 | 98.750000 | 98.750000 |

## 請求與整批指標

| 頻寬 GB/s | 方法 | Avg client TTFT s | TTFT P95 s | Avg D-first s | Avg lifecycle s | Avg TPOT s/token | Output tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | Serial | 0.113181 | 0.568615 | 0.113181 | 2.284501 | 0.003092 | 182.659174 |
| 50 | PPD | 0.112877 | 0.568615 | 0.112877 | 2.284777 | 0.003093 | 182.659174 |
| 50 | MigrationTax | 0.080464 | 0.398485 | 0.080464 | 2.249511 | 0.003088 | 182.669054 |
| 100 | Serial | 0.077496 | 0.377019 | 0.077496 | 2.243089 | 0.003089 | 182.672001 |
| 100 | PPD | 0.077333 | 0.377019 | 0.077333 | 2.243230 | 0.003089 | 182.672001 |
| 100 | MigrationTax | 0.060027 | 0.283884 | 0.060027 | 2.224491 | 0.003087 | 182.677601 |
| 200 | Serial | 0.059611 | 0.281221 | 0.059611 | 2.222346 | 0.003087 | 182.678415 |
| 200 | PPD | 0.059519 | 0.281221 | 0.059519 | 2.222419 | 0.003088 | 182.678415 |
| 200 | MigrationTax | 0.050925 | 0.234569 | 0.050925 | 2.213107 | 0.003086 | 182.681223 |
| reference | Serial | 0.041968 | 0.185442 | 0.041968 | 2.201840 | 0.003086 | 182.684828 |

## 統計口徑

- 200 筆請求含 80 個 returning turns；同一份 ServeGen seed 0 容量篩選子集的前 200 筆；未再排除請求、截短 token 或重縮到達時間；保留前輪相依。
- Affected 使用同一個 source request 對照 Serial 高頻寬參考；延遲嚴格大於參考的兩倍才算 affected，分母為全部 returning turns。
- D-first 主指標採真正的第一個 D worker 輸出事件，從原始到達計算。本子集輸出至少 4 tokens，全部請求都有 D 採樣。HF 的前兩個輸出可在 P 產生，因此第一個 D 輸出可能是整段回覆的第三個 token。KV 交接／ownership 時間另作 D availability proxy 保存，不取代 D 採樣事件。
- Client TTFT 保留原生使用者首輸出時間，HF 可在 P 先輸出。兩種 endpoint 分開報告。
- Decode miss 來自原始到達時對每個 D 的快取快照，再對應實際目的 D；不是 store load 或 admission 後的命中。到達時候選 D 與最後目的 D 的差異也保留於 CSV。
- 上表 decode miss 採整段 inherited prefix 未完整命中，包含部分 prefix 命中。comparison.csv 與 results.json 另列零覆蓋與部分覆蓋比例，保留 block 粒度造成的差異。
- Cache 狀態、路由與排隊由各條件獨立演化，不假設 miss share 在方法或頻寬間相同。
- 原生完整生命週期含寫回；TPOT 為每請求平均 token 間隔的算術平均；吞吐量從時間 0 至最後完整完成，包含到達空檔。
- 指定 Llama-7B FP16 在此固定 H100 vLLM 資料包缺少 FP16 量測；缺少的 operator 使用原生 analytical fallback。每組的實際 match_type 計數保存在 results.json 與原生結果。
- 使用既有 fork 的 cache-aware P 路由、D affinity 與完整寫回；P2P 使用可提供明確數值延遲的 MooncakeConnector。與提供 HTML 的原版問題情境有差異，預期百分比不作通過門檻。
