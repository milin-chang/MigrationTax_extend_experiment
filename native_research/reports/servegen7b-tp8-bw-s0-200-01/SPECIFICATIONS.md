# TokenSim／ServeGen 200 筆 TP8 實驗規格

本輪以 CPU 執行 TokenSim，GPU 為模擬資源，不載入模型權重。
本文件配合本次 200 筆結果使用。

## 模型與硬體

| 項目 | 設定 |
|---|---|
| 模型／權重／KV 精度 | LLaMA 7B（`llama-7b`）／FP16／FP16 |
| 參數量／層數 | 6,738,149,376／32 |
| GPU | H100 SXM，catalog 為 80 GiB／rank |
| P／D 配置 | 4 個 P TP 群組＋4 個 D TP 群組 |
| Tensor／Pipeline Parallel | TP8／PP1 |
| 模擬 GPU 總數 | 64；每群組 8 ranks |
| 網路拓撲 | 8 個邏輯網路域，每域 8 ranks；群組內使用 NVLink |
| 模型權重／GPU rank | 1,684,537,344 bytes |
| 執行環境記憶體預留／GPU rank | 4,169,138,176 bytes |
| KV 大小／token | 524,288 bytes／完整 TP 群組；65,536 bytes／rank |
| KV block | 64 tokens；4 MiB／rank，32 MiB／完整群組 |
| KV blocks／rank | 19,084 |
| KV token 容量／完整群組 | 1,221,376；各 rank 是相同 token 的分片，不再乘以 8 |
| 原生准入 | 保守預留完整 context／output 需求，加 2 個 COW／boundary blocks；不折抵 cache sharing |
| Batching | paged-attn；64 sequences、8,192 tokens／batch |

## 通訊設定

| 路徑 | 設定 |
|---|---|
| GPU HBM | 固定 3,352 GB/s／GPU |
| TP 群組內 collective | 固定 `nvlink4`，單向 catalog 頻寬 450 GB/s；原生模型另計 efficiency、launch 與同步成本 |
| P→D KV | 50／100／200 GB/s，十進位、單向 |
| Store↔群組 | 同一條件的 50／100／200 GB/s host-link 服務 |
| KV／Store 固定傳輸延遲 | 0，隔離頻寬變化；TP 原生啟動成本仍保留 |
| 高頻寬參考 | Serial，KV／Store `1e15 bytes/s`；TP 群組內仍固定 NVLink |
| 頻寬計量範圍 | 完整 TP KV bundle 共用群組端點頻寬，沒有乘成 8 倍 |
| Connector | MultiConnector：MooncakeConnector＋embedded MooncakeStoreConnector |
| Store | 全域 10,000,000 個完整 KV blocks，約 305.176 TiB 的模擬邏輯容量 |
| Buffer | 每個活躍 TP 群組端點 1 GiB，8 個端點共 8 GiB 的模擬配置 |
| Offload／async／全域 overlap | 關閉；MigrationTax 方法自己的事件相依與重疊仍保留 |

Store 與 buffer 容量不是宿主機程序實際配置的 RAM。
TP 採既有對稱群組模型，以 leader 排程與 canonical KV ledger 表示完整群組；未模擬獨立 rank NIC 或非對稱 rank 進度。

## 輸入與方法

| 項目 | 設定 |
|---|---|
| Trace／Seed | ServeGen primary v1／0 |
| 取樣 | 既有容量篩選後 8,000 筆子集，依原順序取前 200 筆；沒有新增排除或截短 token |
| 請求／對話／returning turns | 200／120／80 |
| 每組輸出 tokens | 137,909 |
| 原始到達時間範圍 | 0.000000～752.901239 模擬秒 |
| 最大 context＋output−1 | 98,975 tokens |
| 方法 | Serial、adapted PPD、MigrationTax |
| 正式條件數 | 3 頻寬×3 方法＋1 組 Serial reference＝10 組 |
| P／D 路由 | 既有 cache-aware P、returning D affinity 與准入預算路由 |
| 前輪相依 | 等上一輪完整完成、含寫回，再提交；原始到達時間保留 |

子集 SHA-256：`d053822266c1529fc9fc6b44317a679f1878ac389a2e378aaa6dc5cac57166e1`。
[取樣 manifest](../../../inputs/servegen-s0-first200-tp8.manifest.json) 記錄直接來源與上游 8,000 筆的篩選來源。

## 計時與指標

本次使用固定 H100 vLLM operator 資料包；FP16 查詢缺少對應量測時，使用原生 analytical fallback。
各組實際 match type 與 TP collective 統計以原生 JSON 為準。
模型 catalog 上下文為 4,096 tokens，較長請求僅作計時與容量外插。
歷史 `splitwise_*` profile 欄位不供應本次 TokenSim 時間。

| 指標 | 定義／單位 |
|---|---|
| Client TTFT | 原始到達至第一個使用者輸出，含前輪、准入、排隊、計算與傳輸等待；秒 |
| D-first | 原始到達至第一個 D worker 採樣輸出；HF 可能先在 P 產生兩個 token；秒 |
| 生命週期 | 原始到達至完整完成，含寫回；秒 |
| Avg TPOT | 各請求的原生 decode time／(output tokens−1)，再取算術平均；秒／token |
| 整批吞吐 | 全部輸出 tokens／時間 0 至最後寫回完成；tokens/s，包含到達空檔 |
| Affected share | returning 請求的 D-first 嚴格大於同 request 的 Serial reference 兩倍；分母 80，百分比 |
| Decode miss share | 原始到達時快照，對應最後目的 D 的 inherited prefix 未完整命中；分母 80，百分比 |

每個 returning request 對 affected share 的影響為 1.25 個百分點。
高頻寬參考仍有排程與等待，不代表每筆無排隊的理想延遲。
200 筆與之前 8,000 筆同時改變了輸入範圍與 TP 配置，平均值差異不能單獨歸因於 TP。

## 原始設定與操作

- [模型與 GPU profile](../../../configs/servegen7b-tp8-200/hardware.json)
- [Cluster](../../../configs/servegen7b-tp8-200/cluster.json)
- [50 GB/s 設定](../../../configs/servegen7b-tp8-200/study_bw50.json)
- [100 GB/s 設定](../../../configs/servegen7b-tp8-200/study_bw100.json)
- [200 GB/s 設定](../../../configs/servegen7b-tp8-200/study_bw200.json)
- [Reference 設定](../../../configs/servegen7b-tp8-200/study_reference.json)
- [凍結執行計畫](../../plans/servegen7b-tp8-bandwidth-s0-200.json)
- [執行命令與方法說明](../../TP8_BANDWIDTH_200.md)
- [結果 Markdown](results.md)
- [原始結果與程序收據](../../runs/servegen7b-tp8-bw-s0-200-01/)

每工作使用 1 個邏輯 CPU，上限 8 GiB；控制器保留至少 2 GiB 可用 RAM，依資源動態決定並行數。
原生模擬引擎與三方法實作沿用既有程式，本輪新增配置與測試，並讓執行、稽核與報告工具支援不同輸入筆數及 TP ranks。
舊實驗的輸入、原始結果與 source snapshots 均保留。

## 執行與驗收結果

十組均完成 200 筆，無拒絕或未完成請求，全部程序與原始結果稽核通過。
651 項測試通過；九份 paired CSV 的 affected 判斷逐筆重新計算一致。
正式矩陣實際經過 75.96 秒，約 1 分 16 秒；各組約 16.8～20.2 秒。
最高單工作 peak RSS 約 0.523 GiB。
所有條件均執行 TP collective；本次 compute match type 全部為 analytical。

[TTFT 等待拆分](TTFT_DECOMPOSITION.md) 顯示，本批每筆請求的前輪與准入預算等待均為 0。
[verification.json](verification.json) 保存重算、測試與檔案雜湊紀錄。
