# Mooncake TP1／2／4／8 比較圖

Times New Roman；llama-7b／FP16；H100 SXM；4P＋4D；seed 0。

每個 TP 納入 10 組已稽核的 Mooncake 正式條件；另驗證 TP1 同設定重跑 10 組。TP8 採已完成的 Mooncake delivery，未納入 ServeGen 或 sanity。

TP1／2／4／8 分別為 8／16／32／64 張模擬 GPU，KV 容量也隨 TP 改變。這是部署設定比較。

TTFT 比較圖使用對數縱軸，所有數字仍為秒；下表保留六位小數，粗體綠底為同 TP、同頻寬下最佳方法。Affected 各自使用該 TP 的 Serial inf reference。

## 圖片

- [tp1-three-stages.png](tp1-three-stages.png)
- [tp2-three-stages.png](tp2-three-stages.png)
- [tp4-three-stages.png](tp4-three-stages.png)
- [tp8-three-stages.png](tp8-three-stages.png)
- [tp-comparison-mean-ttft-s.png](tp-comparison-mean-ttft-s.png)
- [tp-comparison-p50-ttft-s.png](tp-comparison-p50-ttft-s.png)
- [tp-comparison-p95-ttft-s.png](tp-comparison-p95-ttft-s.png)
- [tp-comparison-max-ttft-s.png](tp-comparison-max-ttft-s.png)
- [tp-comparison-affected-d-percent.png](tp-comparison-affected-d-percent.png)
- [tp-comparison-mean-tpot-s.png](tp-comparison-mean-tpot-s.png)
- [tp1-repeatability.png](tp1-repeatability.png)

## Avg client TTFT（秒）

| TP | GPU 數 | 通訊 GB/s | Serial | PPD | MigrationTax |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 450 | 221.693914 | 186.824738 | **147.961256** |
| 1 | 8 | 128 | 248.368308 | 257.098716 | **161.200284** |
| 1 | 8 | 50 | 345.318255 | 370.660338 | **200.430121** |
| 2 | 16 | 450 | 0.267218 | 0.267215 | **0.228690** |
| 2 | 16 | 128 | 0.350549 | 0.350547 | **0.262806** |
| 2 | 16 | 50 | 0.541637 | 0.541632 | **0.341858** |
| 4 | 32 | 450 | 0.172246 | 0.171946 | **0.152494** |
| 4 | 32 | 128 | 0.243551 | 0.243046 | **0.185630** |
| 4 | 32 | 50 | 0.407814 | 0.407936 | **0.262465** |
| 8 | 64 | 450 | 0.124063 | 0.124021 | **0.110534** |
| 8 | 64 | 128 | 0.187284 | 0.187007 | **0.142788** |
| 8 | 64 | 50 | 0.330853 | 0.329760 | **0.221673** |

## TP1 重現性

逐筆比較 120,310 筆配對紀錄，完整 request record 不同者為 0 筆。
同 seed 重跑不視為新 seed，不加入 TP1 平均；細節見 `tp1-repeatability.csv`。

## 附件與重畫指令

- `request-metrics.csv`：40 組正式條件的 TTFT、生命週期、TPOT 與吞吐。
- `affected-share.csv`：36 組有限頻寬條件的 affected 與 decode miss。
- `verification.json`：50 組原始逐筆重算、完成稽核、來源 SHA-256 與字體證據。
- 每張 PNG 另附同名 SVG；`mooncake-tp-comparison-figures.zip` 包含全部圖表與彙整。

在專案既有 Python／依賴環境下：

```bash
python3 -m native_research.plot_mooncake_tp_sweep
```

此命令重新讀取既有結果、覆寫本圖表資料夾，不啟動模擬或修改原始 run。
