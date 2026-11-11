# 五階段連線實驗規格

## 模型與 GPU

| 項目 | 設定 |
|---|---|
| 模擬器 | TokenSim，本研究 native method adapter |
| 模型 | llama-7b，6,738,149,376 parameters，32 layers |
| 模型／KV 精度 | FP16／FP16 |
| GPU 型號 | H100 SXM 80GB，device_id＝h100_sxm |
| Catalog 記憶體 | 85,899,345,920 bytes／GPU，即 80 GiB；保留型號名稱與容量數值兩種口徑 |
| P／D 配置 | 4 個 P TP 群組＋4 個 D TP 群組 |
| 每群組 GPU／TP／PP | 8／TP8／PP1 |
| 模擬 GPU 總數 | 64 |
| GPU HBM 頻寬 | 全部條件固定 3,352 GB/s／GPU |
| TP 群組內通訊 | 固定 nvlink4；catalog 單向 450 GB/s，另計原生 collective efficiency 與啟動成本 |
| KV block | 64 tokens |
| KV 大小 | 524,288 bytes/token／完整 TP 群組；65,536 bytes/token／rank |
| KV 容量 | 19,084 blocks／rank；1,221,376 tokens／完整 TP 群組，不再乘以 8 |
| Batching | paged-attn，max_num_seqs＝64，max_num_batched_tokens＝8,192 |
| Store | embedded，共享 10,000,000 個完整 KV blocks，1 GiB buffer／活躍 TP 群組端點 |
| 方法 | Serial、adapted PPD、MigrationTax（Ours） |
| Seed | 0 |

模型與 GPU 是模擬設定，程式在 CPU 執行。
Store 約 305.176 TiB 是模擬邏輯容量，不是宿主機實際配置的 RAM。
固定 H100 vLLM operator 資料包缺少對應 FP16 rows，compute 會使用原生 analytical fallback；實際 match counts 隨每組結果保存。
保留 device_id 可選擇 H100 資料包，不能單憑此欄位宣稱本次命中量測。
模型 catalog 上下文為 4,096 tokens；較長請求屬計時與容量外插。
Hardware profile 內歷史 splitwise_* 欄位不供應本次 TokenSim 計時。

## 執行優先順序：先完成全部 Mooncake

原五個階段保留作為實驗分組，排程拆成下列七批，資料、方法與總組數不變。

| 執行批次 | 原階段 | 內容 | 組數 |
|---:|---|---|---:|
| 0 | 0 | Mooncake 200 筆 sanity：Serial 450／50 GB/s | 2 |
| 1 | 1 | Mooncake 全集：Serial inf＋450 GB/s 三方法 | 4 |
| 2 | 3 的 Mooncake | Mooncake 全集：128 GB/s 三方法 | 3 |
| 3 | 4 的 Mooncake | Mooncake 全集：50 GB/s 三方法 | 3 |
| 4 | 2 | ServeGen 全集：Serial inf＋450 GB/s 三方法 | 4 |
| 5 | 3 的 ServeGen | ServeGen 全集：128 GB/s 三方法 | 3 |
| 6 | 4 的 ServeGen | ServeGen 全集：50 GB/s 三方法 | 3 |

前四批累計 12 組，含 2 組 sanity 與 10 組 Mooncake 正式條件。
Mooncake 全部完成後，先交付它的 CSV、三個連線的 affected share、TTFT 與其他原生指標，再開始 ServeGen 全集十組。
[execution-batches.json](execution-batches.json) 是實際執行順序；下表保留原五階段定義。

## 五階段與 22 組執行清單

| 階段 | Trace／請求數 | 執行條件 | 組數 |
|---:|---|---|---:|
| 0 | Mooncake shifted v1 前 200 筆 | Serial，nvl4 與 ib400 | 2 |
| 1 | Mooncake shifted v1 全量 12,031 筆 | Serial inf reference＋nvl4 下三方法 | 4 |
| 2 | ServeGen primary v1 全集 39,698 筆 | Serial inf reference＋nvl4 下三方法 | 4 |
| 3 | 兩種 Trace，沿用各自正式輸入 | pcie6 下三方法 | 6 |
| 4 | 兩種 Trace，沿用各自正式輸入 | ib400 下三方法 | 6 |
| 合計 | 2 組 sanity＋20 組正式實驗 | 兩種 Trace 各一份完整 Serial inf 參考 | 22 |

Mooncake 正式 returning turns 為 3,959；ServeGen 為 28,482。
Sanity 的 200 筆 Mooncake 有 10 個 returning turns，僅用於前置檢查，不混入正式統計。
原始到達時間、token 長度與前輪相依均保留。
ServeGen 使用原始 normalized 全集 39,698 筆、11,216 段對話，合計 42,344,145 output tokens；不排除對話、不抽樣、不截短 token。
最大 context＋output−1 為 265,207 tokens，符合本次 TP8 每群組約 1,221,376-token 的 KV 容量。

## 連線服務

| Data copy | Store host_link | 完整群組 KV 單向頻寬 | 固定 KV link 延遲 |
|---|---|---:|---:|
| data_nvl4 | nvlink4 | 450 GB/s | 1 µs |
| data_pcie6 | pcie6_x16 | 128 GB/s | 3 µs |
| data_ib400 | ib_ndr400 | 50 GB/s | 3 µs |
| data_inf | inf_link | 1e15 bytes/s | 0 µs |

每個條件保存一份完整 data folder，device_id 維持 h100_sxm。
TP8 的 scale_up_link 全部固定 nvlink4；P→D 由顯式 connector 設定選擇條件頻寬，Store 使用上表 host_link。
因此這裡將 TP 群組內通訊與群組間 KV 服務分開，沒有把 TP collective 也換成 PCIe、IB 或 inf。
跨群組以連線名稱命名的條件是邏輯傳輸服務，不宣稱多台 host 具有實際跨 host NVLink。

目前 uniform Mooncake KV connector 的服務公式為固定延遲加完整 TP bundle bytes／頻寬。
它不使用 catalog 的 half-bandwidth ramp 或 collective efficiency；這些欄位仍保留原始值，且 TP collective 另走固定 NVLink 模型。
Native P→D engine 對 cross_node 延遲乘 1.5，因此 connector 中設定 link latency／1.5，使實際 P→D 固定延遲恰為上表值。
Embedded Store 直接讀取 host_link 的固定延遲與峰值頻寬。
同樣的 native 函式已用 0、1 MiB、256 MiB 的 payload 驗證服務時間，見 validation.json。
完整 TP bundle 共用一個群組端點頻寬，沒有乘成每 rank 各一份頻寬。

這輪與正在跑的 50／100／200 GB/s、固定 KV 延遲 0 的實驗分開保存，舊結果不直接改名沿用。

## 指標與交付

Affected 使用 source_turn＞0，D-first 為原始到達至第一個 D 採樣事件。
延遲嚴格大於同 request 的 Serial inf 參考兩倍才算 affected；等於兩倍不算。
分母為全部 returning turns，三方法與三個有限連線共用該 Trace 的一份參考。
Client TTFT、D availability／ownership 交接時間分開匯出。

Decode miss 使用原始到達時、尚未等待 parent 或 admission 的 GPU 快照，再對應最後實際目的 D。
完整 inherited prefix 未命中即為 miss，包含部分覆蓋；零覆蓋與部分覆蓋另列。
不用 d_admit_hit 替代到達時快取狀態。

Mooncake 有輸出 1～2 tokens 的請求；HF 若在 P 完成全部輸出，可能沒有 D 採樣事件。
這類請求保留於 CSV 與 returning 分母，不用 client 或交接時間冒充 D 採樣。
若發生缺值，D-first affected share 顯示已知下界與可能上界，另列缺值筆數；client affected 仍獨立計算。
所有原生事件都保留，不按預期百分比調整結果。

階段 0 檢查兩個相同 200 筆輸入的平均 D-first 是否在低頻寬條件增加；個別請求可能受排程與快取影響，不強制每筆單調。
各階段都要程序、來源、輸出 token 數與資源釋放稽核通過，才進入下一階段。
Mooncake 完成後，先在 delivery-mooncake-complete 保存累計 12 組的 CSV 與比較表。
ServeGen 的 reference 與 450 GB/s 三方法完成後，在 delivery-after-phase-2 另存累計 16 組的交付快照；其中包含原階段 0～2 的全部條件。
所有 22 組完成後，更新完整 CSV、表格與趨勢圖。

## 執行資源與路徑

最多使用 12 個邏輯 CPU，每個 native 工作 1 CPU，上限 8 GiB。
控制器保留至少 2 GiB 可用 RAM，依工作估計與觀察 peak 動態准入；Mooncake 初估 2 GiB、全量 ServeGen 的 Serial／PPD 初估 3 GiB、MigrationTax 初估 5 GiB。
使用者已暫停 servegen7b-tp8-bw-s0-8000-01 的最後一組；這 22 組立即從 Mooncake sanity 開始，不等待該組完成。
暫停的 8,000 筆工作保留進度，等待使用者後續指示恢復。
執行狀態寫在 runs，靜態 config 不用來宣稱工作是否完成。

- [整體規格](experiment.json)
- [硬體](hardware.json)／[Cluster](cluster.json)
- [連線服務](links.json)／[指標公式](metrics.json)
- [來源與輸入雜湊](input-manifest.json)／[設定驗證](validation.json)
- [階段 0](phases/00-sanity.json)
- [階段 1](phases/01-mooncake-fast.json)
- [階段 2](phases/02-servegen-fast.json)
- [階段 3](phases/03-both-pcie6.json)
- [階段 4](phases/04-both-ib400.json)
- [22 組工作清單](jobs.json)
- [執行狀態](../../native_research/runs/five-stage-llama7b-h100-tp8-s0-01/status.json)
