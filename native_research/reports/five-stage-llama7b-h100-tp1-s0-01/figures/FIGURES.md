# Mooncake TP1 圖表

Times New Roman；seed 0；llama-7b／FP16；H100 SXM；TP1／PP1；4P＋4D，共 8 GPU。

本次只跑 Mooncake 階段 1、3、4，共 10 組，沒有 sanity。圖中秒數保留原始紀錄的單位，並未做毫秒轉換。

TP1／TP8 對照的 GPU 總數不同（8／64）；affected share 各自使用該部署的 Serial inf 參考。這是部署比較。

- [01-nvl4-450.png](01-nvl4-450.png)
- [03-pcie6-128.png](03-pcie6-128.png)
- [04-ib400-50.png](04-ib400-50.png)
- [mooncake-tp1-three-stages.png](mooncake-tp1-three-stages.png)
- [affected-share.png](affected-share.png)
- [ttft-p50-comparison.png](ttft-p50-comparison.png)
- [ttft-p95-comparison.png](ttft-p95-comparison.png)
- [ttft-avg-max-comparison.png](ttft-avg-max-comparison.png)
- [tp1-vs-tp8-affected-share.png](tp1-vs-tp8-affected-share.png)
- [tp1-vs-tp8-avg-ttft.png](tp1-vs-tp8-avg-ttft.png)

每張圖另有同名 SVG。數字表見 ttft-summary.csv、affected-share.csv、tp1-vs-tp8.csv；原始事件重算與字體證據見 verification.json。
