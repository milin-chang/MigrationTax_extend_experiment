# TokenSim：五階段連線實驗

llama-7b／FP16、H100 SXM 80GB、TP8、4P＋4D，共 64 張模擬 GPU；seed 0。

已完成並通過稽核：12／22 組。

| 階段 | 已完成 | 規劃組數 |
|---:|---:|---:|
| 0 | 2 | 2 |
| 1 | 4 | 4 |
| 2 | 0 | 4 |
| 3 | 3 | 6 |
| 4 | 3 | 6 |

## mooncake：Affected share

D-first 嚴格大於同請求 Serial inf 參考兩倍；分母為 returning turns。

| 連線／GB/s | Serial % | PPD % | MigrationTax % |
|---|---:|---:|---:|
| nvl4／450 | 23.263450 | 23.541298 | 13.260924 |
| pcie6／128 | 45.213438 | 44.733519 | 35.615054 |
| ib400／50 | 68.047487 | 68.072746 | 58.044961 |

## servegen：Affected share

D-first 嚴格大於同請求 Serial inf 參考兩倍；分母為 returning turns。

| 連線／GB/s | Serial % | PPD % | MigrationTax % |
|---|---:|---:|---:|
| nvl4／450 | pending | pending | pending |
| pcie6／128 | pending | pending | pending |
| ib400／50 | pending | pending | pending |

## Decode miss share

原始到達時，實際目的 D 的 inherited prefix 未完整命中，包含部分覆蓋。

| Trace | 連線 | 方法 | Returning turns | Decode miss % |
|---|---|---|---:|---:|
| mooncake | nvl4 | Serial | 3959 | 99.191715 |
| mooncake | nvl4 | PPD | 3959 | 99.267492 |
| mooncake | nvl4 | MigrationTax | 3959 | 99.267492 |
| mooncake | pcie6 | Serial | 3959 | 99.242233 |
| mooncake | pcie6 | PPD | 3959 | 99.267492 |
| mooncake | pcie6 | MigrationTax | 3959 | 99.267492 |
| mooncake | ib400 | Serial | 3959 | 99.292751 |
| mooncake | ib400 | PPD | 3959 | 99.267492 |
| mooncake | ib400 | MigrationTax | 3959 | 99.267492 |

## 原生請求指標

| 階段 | Trace | 連線 | 方法 | Avg TTFT s | P95 TTFT s | Avg lifecycle s | Avg TPOT s/token |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | mooncake | nvl4 | Serial | 0.172193 | 0.607182 | 1.259209 | 0.003299 |
| 0 | mooncake | ib400 | Serial | 0.390754 | 1.267998 | 1.536059 | 0.003450 |
| 1 | mooncake | inf | Serial | 0.100285 | 0.350582 | 1.264779 | 0.003363 |
| 1 | mooncake | nvl4 | Serial | 0.124063 | 0.425083 | 1.297766 | 0.003386 |
| 1 | mooncake | nvl4 | PPD | 0.124021 | 0.424666 | 1.299151 | 0.003390 |
| 1 | mooncake | nvl4 | MigrationTax | 0.110534 | 0.387922 | 1.277258 | 0.003367 |
| 3 | mooncake | pcie6 | Serial | 0.187284 | 0.650399 | 1.386552 | 0.003479 |
| 3 | mooncake | pcie6 | PPD | 0.187007 | 0.646915 | 1.386546 | 0.003468 |
| 3 | mooncake | pcie6 | MigrationTax | 0.142788 | 0.486419 | 1.312398 | 0.003372 |
| 4 | mooncake | ib400 | Serial | 0.330853 | 1.135618 | 1.596989 | 0.003703 |
| 4 | mooncake | ib400 | PPD | 0.329760 | 1.133006 | 1.596502 | 0.003705 |
| 4 | mooncake | ib400 | MigrationTax | 0.221673 | 0.766417 | 1.400054 | 0.003383 |

## 統計與模型口徑

- Mooncake 為 shifted v1 全量 12,031 筆；ServeGen 為 primary v1 全量 39,698 筆，不排除對話；sanity 為 Mooncake 前 200 筆。
- 各 Trace 共用一份全量 Serial inf 參考，依同一 source_id 配對；sanity 不混入正式統計。
- D-first 為原始到達至第一個實際 D 採樣事件；client TTFT 另列。
- Cache miss 使用原始到達快照，對應最後目的 D，不使用 admission 後的 d_admit_hit。
- 連線服務使用峰值 bytes/s 與固定延遲；目前 Mooncake KV 路徑不套用 catalog 的 ramp 或 collective efficiency。
- TP 群組內維持 nvlink4；GPU HBM 維持 3,352 GB/s。
- 所有方法的 compute 實際 match type 另列 JSON；FP16 缺少量測時使用原生 analytical 模型，未完成實機校準。
- 對照圖表預期百分比不作驗收門檻。

[CSV 索引](csv-index.json)／[比較表](comparison.csv)／[完整數值](results.json)。
