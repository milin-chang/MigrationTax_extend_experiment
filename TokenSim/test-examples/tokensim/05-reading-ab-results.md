# 閱讀 200 筆 A/B 結果

這份案例直接讀取已保存的四個正式執行，不啟動新模擬。
固定使用 `*-fullwb-*` 的本輪資料，避免與先前的中間版本混用。

## 完成筆數與穩定 ID 先通過

```python
import json
from pathlib import Path

root = Path("trace-study/results/tokensim-flow-v3")

def read_turns(run):
    path = root / run / "scenario" / "turns.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 200
    assert len({r["request_id"] for r in rows}) == 200
    return {r["request_id"]: r for r in rows}

a = read_turns("moon200-fullwb-a")
b = read_turns("moon200-fullwb-b")
assert a.keys() == b.keys()
```

request ID 對應來源資料，可能有跳號；沒有要求它剛好是 `0..199`。
較早停在 197 筆的執行無法通過筆數檢查，因此不能把它的部分資料當成完整結果。
還應查看各 run 的 `run.json`、`scenario/summary.json`，確認成功退出與完成狀態。

## 比較每筆狀態，不只比較總命中率

```python
same_fields = (
    "arrival_s", "input_tokens", "output_tokens", "prefix_tokens",
    "prefill_worker", "decode_worker",
    "p_local_tokens", "p_store_tokens", "d_local_tokens",
    "computed_prompt_tokens", "cell", "p2p_bytes",
    "store_fetch_bytes", "store_writeback_bytes",
)

for rid in a:
    assert all(a[rid][field] == b[rid][field] for field in same_fields)
```

總 HH 比例相同仍可能是不同請求互換狀態，無法證明逐輪 A/B 語意相同。
時間本身預期可不同；完整配對器也會核對輸入雜湊、設定與載入的核心版本。

## 三條 KV 路徑各自核對

本輪 7B MHA 使用精確 524288 bytes/token：

```python
for row in a.values():
    C = row["input_tokens"]
    N = row["output_tokens"]

    assert row["generated_tokens"] == N
    assert row["materialized_kv_tokens"] == C + N - 1
    assert row["store_fetch_bytes"] == row["p_store_tokens"] * 524288
    assert row["p2p_bytes"] == (C - row["d_local_tokens"]) * 524288
    assert row["store_writeback_bytes"] == (C + N - 1) * 524288
```

store→P 是取回缺少的 prefix；P→D 扣除 D 已有的 prefix；D→store 是輪末完整一次寫回。
將三者合併成一個 cache-hit count，會看不出重複抓取或寫回少算 bytes 的問題。

## 首 token 直接配對，不再額外加 ST

```python
paired = []
for rid in a:
    ta = a[rid]["timing"]["source_first_token_s"]
    tb = b[rid]["timing"]["source_first_token_s"]
    assert ta > 0 and tb > 0
    paired.append({"request_id": rid, "turn": a[rid]["turn"], "slowdown": ta / tb})

returning = [r for r in paired if r["turn"] > 0]
assert len(returning) == 8
over_2 = sum(r["slowdown"] > 2 for r in returning)
assert over_2 == 0
assert over_2 / len(returning) == 0.0
```

這些 `assert` 對應保存的 Mooncake 子集，表示 0／8，不是完整 Mooncake trace 的零風險結論。
此模式的首 token 已經由 D step 產生，再加 ST 會重複計費。
相對 solo 的解析式 gap 則是另一個指標。

## 分母與範圍

已保存的本輪結果為：

| 子集 | 全部請求 | returning turns | returning 六狀態筆數 |
| --- | ---: | ---: | --- |
| Mooncake | 200 | 8 | HH＝5、FF＝3，其餘＝0 |
| ServeGen | 200 | 125 | HH＝125，其餘＝0 |

這些是已核對 chronological subset 的前 200 筆，來源種子為 0。
Mooncake returning 樣本少，ServeGen 此段也沒有自然產生六種狀態。
R 的功能可由獨立診斷驗證，但不能拿診斷案例填入正式 trace 的占比。
目前資料不支持完整 trace、四 seed、學姊原始數值或真實 GPU 效能已對齊的結論。

## 對照位置

- [本輪總表](../../../trace-study/results/tokensim-flow-v3/summary.json)
- [Mooncake 逐輪 A/B](../../../trace-study/results/tokensim-flow-v3/moon200-fullwb-pair/turns.jsonl)
- [ServeGen 逐輪 A/B](../../../trace-study/results/tokensim-flow-v3/serve200-fullwb-pair/turns.jsonl)
- [正式配對與檢查邏輯](../../../trace-study/compare_tokensim_flow_v3.py)
- [驗收與守恆核對](../../../trace-study/finalize_tokensim_flow_v3.py)
