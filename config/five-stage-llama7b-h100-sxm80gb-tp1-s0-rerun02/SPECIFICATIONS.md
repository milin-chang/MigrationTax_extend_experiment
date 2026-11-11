# TokenSim／Mooncake：TP1 連線實驗

沿用 five-stage-llama7b-h100-tp8-s0-01 的完整 Mooncake shifted v1 seed 0，只執行階段 1、3、4；不執行 sanity、ServeGen 或估時微型模擬。

| 項目 | 設定 |
|---|---|
| 模型／精度 | llama-7b／FP16，32 層 |
| GPU | H100 SXM 80GB，catalog 80 GiB |
| P／D | 4P＋4D，每個 worker 1 張，共 8 張 GPU |
| TP／PP | TP1／PP1 |
| 輸入 | 12,031 requests、3,959 returning turns、4,122,048 output tokens；與 TP8 輸入逐 byte 相同 |
| KV block／batching | 64 tokens；paged-attn；max_num_seqs＝64；max_num_batched_tokens＝8,192 |
| 單 worker KV 容量 | 2,034 blocks＝130,176 tokens |
| 最大請求 live tokens | 126,526；全部請求可容納，不需刪減 |
| 連線 | nvl4＝450 GB/s；pcie6＝128 GB/s；ib400＝50 GB/s；inf＝1e15 bytes/s |
| 通訊設定 | 沿用 TP8 各連線的 Store／P→D 頻寬與固定延遲；不改 HBM |
| HBM | 固定 3,352 GB/s |
| 原生計時 | FP16 operator analytical fallback；TP1 無跨 rank 的 TP collective |
| CPU／記憶體 | 最多 12 邏輯 CPU；每工作 1 CPU、上限 8 GiB；動態並行並保留至少 2 GiB host 空間 |

| 階段 | 條件 | 組數 |
|---:|---|---:|
| 1 | Serial inf reference＋nvl4 下 Serial／PPD／MigrationTax | 4 |
| 3 | pcie6 下 Serial／PPD／MigrationTax | 3 |
| 4 | ib400 下 Serial／PPD／MigrationTax | 3 |

依階段順序執行，階段內由現有 resource-admitted controller 動態安排並行。共 10 組，不是 16 組。TP1 共 8 GPU，原 TP8 共 64 GPU；兩者為不同 GPU 預算的部署比較。

完成後自動輸出 results.md、results.json、native-metrics.csv、comparison.csv、csv-index.json 與十組逐請求 CSV；保留原始 audit、job receipts 與 source-snapshot。

啟動：

```bash
python -m native_research.run_mooncake_link_study \
  --design config/five-stage-llama7b-h100-sxm80gb-tp1-s0-rerun02 \
  --run-root native_research/runs/five-stage-llama7b-h100-tp1-s0-02 \
  --report-dir native_research/reports/five-stage-llama7b-h100-tp1-s0-02
```

既有計畫、工作 ID 與輸出不覆寫。原 TP8 結果不修改。

## 本次重跑

原 TP1 的 seed、Trace、模型、GPU、KV 容量、頻寬、延遲與實際載入的 Python 來源均已逐檔核對相同。只更換工作 ID、設定與輸出路徑；本次等待 TP4 完成後才啟動。

原始對照：`native_research/runs/five-stage-llama7b-h100-tp1-s0-01`；新紀錄：`native_research/runs/five-stage-llama7b-h100-tp1-s0-02`；新報告：`native_research/reports/five-stage-llama7b-h100-tp1-s0-02`。

啟動時加上 `--after-run native_research/runs/five-stage-llama7b-h100-tp4-s0-01`。
