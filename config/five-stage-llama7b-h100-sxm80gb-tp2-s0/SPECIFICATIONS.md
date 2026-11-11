# TokenSim／Mooncake：TP2 連線實驗

| 項目 | 規格 |
|---|---|
| 模型／精度 | llama-7b／FP16，32 層 |
| GPU | H100 SXM 80GB；catalog 80 GiB／GPU |
| P／D | 4P＋4D，每群組 2 GPU；共 16 張模擬 GPU |
| 平行方式 | TP2／PP1 |
| Trace | Mooncake shifted v1、seed 0、全量 12,031 筆；與 TP1／TP8 輸入逐 byte 相同 |
| 輸出／returning | 4,122,048 output tokens；3,959 returning turns |
| KV block／容量 | 64 tokens；4,469 blocks＝286,016 tokens／完整 TP 群組 |
| Batching | paged-attn；max_num_seqs＝64；max_num_batched_tokens＝8,192 |
| 通訊連線 | nvl4＝450 GB/s；pcie6＝128 GB/s；ib400＝50 GB/s；inf＝1e15 bytes/s |
| HBM／群組內連線 | HBM 固定 3,352 GB/s；TP collective 使用固定 NVLink |
| KV 頻寬計量 | 完整 TP 群組端點，不額外乘上 TP rank 數 |
| 計時 | 原生 FP16 operator analytical fallback＋原生 TP collective／connector 計時 |
| CPU／記憶體 | 最多 12 邏輯 CPU；每工作 1 CPU、上限 8 GiB；動態並行並保留至少 2 GiB host 空間 |

| 階段 | 條件 | 組數 |
|---:|---|---:|
| 1 | Serial inf reference＋nvl4 下 Serial／PPD／MigrationTax | 4 |
| 3 | pcie6 下 Serial／PPD／MigrationTax | 3 |
| 4 | ib400 下 Serial／PPD／MigrationTax | 3 |

共 10 組；不執行 sanity、ServeGen 或估時微型模擬。每個 TP 使用自己的 Serial inf 參考。
TP2 與 TP4 分成兩批獨立結果；TP4 等 TP2 成功完成後自動開始，避免兩個控制器同時預留記憶體。

啟動指令：

```bash
python -m native_research.run_mooncake_link_study \
  --design config/five-stage-llama7b-h100-sxm80gb-tp2-s0 \
  --run-root native_research/runs/five-stage-llama7b-h100-tp2-s0-01 \
  --report-dir native_research/reports/five-stage-llama7b-h100-tp2-s0-01
```

TP4 排隊時另加 `--after-run native_research/runs/five-stage-llama7b-h100-tp2-s0-01`。
結果自動輸出 Markdown 與逐請求 CSV；原始 log、程序收據、audit 與 source-snapshot 分開保留。
既有計畫與輸出不覆寫。各 TP 的 GPU 總數不同，屬不同資源量的部署比較。
