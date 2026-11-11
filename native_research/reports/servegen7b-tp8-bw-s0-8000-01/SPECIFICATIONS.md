# TokenSim／ServeGen 8,000 筆 TP8 實驗規格

本輪以 CPU 執行 TokenSim，GPU 是模擬資源，不載入模型權重。
正式條件為 seed 0，三種頻寬各跑三個方法，再加一組 Serial 高頻寬參考，共十組。

## 模型、GPU 與 KV

| 項目 | 設定 |
|---|---|
| 模型／權重／KV 精度 | LLaMA 7B（`llama-7b`）／FP16／FP16 |
| 參數量／層數 | 6,738,149,376／32 |
| GPU | H100 SXM，catalog 為 80 GiB／rank |
| P／D 配置 | 4 個 P TP 群組＋4 個 D TP 群組 |
| Tensor／Pipeline Parallel | TP8／PP1 |
| 模擬 GPU 總數 | 64；每群組 8 ranks |
| 拓撲 | 8 個邏輯網路域，每域 8 ranks；群組內使用 NVLink |
| 模型權重／GPU rank | 1,684,537,344 bytes |
| 執行環境預留／GPU rank | 4,169,138,176 bytes |
| KV 大小／token | 524,288 bytes／完整 TP 群組；65,536 bytes／rank |
| KV block | 64 tokens；4 MiB／rank，32 MiB／完整群組 |
| KV blocks／rank | 19,084 |
| KV token 容量／完整群組 | 1,221,376；各 rank 是同一批 token 的分片，不再乘以 8 |
| 原生准入 | 保守預留完整 context／output 需求，加 2 個 COW／boundary blocks；不折抵 cache sharing |
| Batching | paged-attn；64 sequences、8,192 tokens／batch |

本次沿用 TP8 200 筆驗證所用的模擬引擎、三方法、硬體 profile、模型 catalog 與 operator 資料包。
新的硬體、cluster、transfer 與 catalog 檔案逐 byte 比對原 TP8 200 筆配置，內容相同。
Study JSON 僅更新資源路徑與輸入描述。

## 通訊與 Store

| 路徑 | 設定 |
|---|---|
| GPU HBM | 固定 3,352 GB/s／GPU |
| TP 群組內 collective | 固定 `nvlink4`，單向 catalog 頻寬 450 GB/s；另計原生 efficiency、launch 與同步成本 |
| P→D KV | 50／100／200 GB/s，十進位、單向 |
| Store↔群組 | 同一條件的 50／100／200 GB/s host-link 服務 |
| KV／Store 固定傳輸延遲 | 0；TP 原生啟動成本仍保留 |
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
| 輸入 | `inputs/servegen-s0-first8000.jsonl`，與先前 TP1 實驗完全相同 |
| 取樣來源 | 依原 TP1 容量規則排除超量對話後，按到達順序取前 8,000 筆；本輪沒有再排除、截短 token 或重縮時間 |
| 請求／對話／returning turns | 8,000／2,680／5,320 |
| 每組輸出 tokens | 7,663,048 |
| 原始到達時間範圍 | 0～16,457.611273 模擬秒 |
| Prompt／Output 範圍 | 15～122,178／4～9,629 tokens |
| 最大 context＋output−1 | 124,092 tokens |
| 方法 | Serial、adapted PPD、MigrationTax |
| P／D 路由 | 既有 cache-aware P、returning D affinity 與准入預算路由 |
| 前輪相依 | 等上一輪完整完成、含寫回，再提交；原始到達時間保留 |

輸入 SHA-256：`66d767ac2b84181eb7cf8fb94245b12779280eaa098a9eed0845ea2b9883caf5`。
本輪沒有將 TP8 額外可容納的對話補回，以保留與原 TP1 結果相同的輸入。
模型 catalog 上下文為 4,096 tokens，較長請求僅作計時與容量外插。
固定 H100 vLLM 資料包缺少對應 FP16 operator rows，使用原生 analytical fallback；實際 match type 以每組原生 JSON 為準。
歷史 `splitwise_*` profile 欄位不供應本次 TokenSim 時間。

## 指標與比較口徑

| 指標 | 定義／單位 |
|---|---|
| Client TTFT | 原始到達至第一個使用者輸出，含前輪、准入、排隊、計算與傳輸等待；秒 |
| D-first | 原始到達至第一個 D worker 採樣輸出；HF 可先在 P 產生兩個 token；秒 |
| 生命週期 | 原始到達至完整完成，含寫回；秒 |
| Avg TPOT | 各請求的原生 decode time／(output tokens−1)，再取算術平均；秒／token |
| 整批吞吐 | 全部輸出 tokens／時間 0 至最後寫回完成；tokens/s，包含到達空檔 |
| Affected share | returning 請求的 D-first 嚴格大於同 request 的 Serial reference 兩倍；分母 5,320，百分比 |
| Decode miss share | 原始到達時快照，對應最後目的 D 的 inherited prefix 未完整命中；分母 5,320，百分比 |

高頻寬參考仍有排程與等待，不代表每筆無排隊的理想延遲。
相同 8,000 筆的 TP1 與 TP8 可用於部署比較；GPU 總數也從 8 張增加至 64 張，因此不屬於相同 GPU 預算的比較。
TP8 200 筆與本輪同配置但輸入範圍不同，不能只看平均值推論全量負載表現。

## 原始設定與執行

- [輸入與原始篩選 manifest](../../../inputs/subset-manifest.json)
- [硬體 profile](../../../configs/servegen7b-tp8-8000/hardware.json)
- [Cluster](../../../configs/servegen7b-tp8-8000/cluster.json)
- [50 GB/s](../../../configs/servegen7b-tp8-8000/study_bw50.json)
- [100 GB/s](../../../configs/servegen7b-tp8-8000/study_bw100.json)
- [200 GB/s](../../../configs/servegen7b-tp8-8000/study_bw200.json)
- [Reference](../../../configs/servegen7b-tp8-8000/study_reference.json)
- [凍結執行計畫](../../plans/servegen7b-tp8-bandwidth-s0-8000.json)
- [原始結果與程序收據](../../runs/servegen7b-tp8-bw-s0-8000-01/)
- [完整指標](results.md)

每工作使用 1 個邏輯 CPU，上限 8 GiB；控制器保留至少 2 GiB 可用 RAM，依資源動態決定並行數。
開始執行時固定程式、設定、catalog 與輸入雜湊，並保存每組 source snapshots。
舊 TP1 8,000 筆與 TP8 200 筆的原始輸入、結果和來源快照保留。

## 執行與驗收結果

十組均完成 8,000 筆，全部稽核通過；九份 paired CSV 的 affected 判斷逐筆重新計算一致。
正式執行經過 359.94 分鐘，最高單工作 peak RSS 為 1.413 GiB。
[TTFT 等待拆分](TTFT_DECOMPOSITION.md)／[相同輸入的 TP1 與 TP8 對照](TP1_VS_TP8.md)／[稽核與重算紀錄](verification.json)。
