# 到達時鐘、路由與容量

這裡把規格推導與既有執行記錄分開。
到達 FIFO、逐對話 LRU、最低 ID ties 都是目前共同情境模式的明訂規則。

## P 優先找 holder，D 維持 affinity

規格案例：繼承 32-token prefix，P0 有完整副本，P1 較空但沒有副本。

```python
prefix = 32
prefill_workers = [
    {"id": 0, "prefix_tokens": 32, "resident_conversations": 3},
    {"id": 1, "prefix_tokens": 0,  "resident_conversations": 1},
]

holders = [w for w in prefill_workers if w["prefix_tokens"] >= prefix]
chosen_p = min(
    holders or prefill_workers,
    key=lambda w: (w["resident_conversations"], w["id"]),
)
previous_d = 5
expected_route = {"P": chosen_p["id"], "D": previous_d}

assert expected_route == {"P": 0, "D": 5}
```

P 的 holder 優先級高於空閒程度。
D 沿用前輪裝置，即使別的 D 當下有較少 resident conversations。
若有多個完整 P holders，才依 resident conversation 數與 ID 排序。
路由在到達時決定一次；後面 cache 改變不會重新選一組 P／D。

## 既有觀測：改頻寬改時間，arrival 狀態仍相同

壓力案例只有一組 P／D，容量 64 tokens：

| 到達時間 | 請求 | input | 繼承 prefix |
| --- | --- | ---: | ---: |
| 0 ms | A 的第 0 輪 | 32 | 0 |
| 2 ms | B 的第 0 輪 | 48 | 0 |
| 3.5 ms | A 的第 1 輪 | 48 | 32 |

P 的 arrival LRU 在放入 B 時需要淘汰 A，因為 32＋48 超過 64。
A 返回時在 P 缺少 prefix；store 的 A 版本若尚未完成寫回，實際執行仍須等它。

```python
import json
from pathlib import Path

observed = json.loads(Path(
    "trace-study/results/tokensim-flow-v3/acceptance/observations.json"
).read_text())

slow, fast = observed["slow"], observed["fast"]
assert slow["routing"] == fast["routing"]

returning = slow["routing"][-1]
assert returning["p_local_tokens"] == 0
assert returning["p_store_tokens"] == 32
assert returning["d_local_tokens"] == 0
assert slow["turns"][-1]["first_token_s"] > fast["turns"][-1]["first_token_s"]
```

這段讀取已保存的觀測，不重跑模擬。
兩次各自運行 cache 規則；沒有把其中一次的六狀態抄給另一次。

## 邏輯上有版本，執行仍須等待 producer

另一個已驗證情境：A 第 0 輪在 0 到達，第 1 輪在 0.2 ms 到達，繼承 32 tokens。
第 1 輪可以捕捉到 planned resident 的 P／D 版本，但 producer 還沒生成的 KV 不可提早讀取。

以下是假設的事件時序，用來手算 eligibility；數值不是實測：

```python
arrival = 0.0002
ready_p_version = 0.001
ready_d_version = 0.003
previous_p_prefill_done = 0.001

# 無額外固定 overhead；尚未考慮其他容量等待。
p_eligible = max(arrival, ready_p_version, ready_d_version, previous_p_prefill_done)
assert p_eligible == 0.003
assert p_eligible > arrival

new_prefill_done = p_eligible + 0.001
previous_d_completed = 0.0032
d_eligible = max(new_prefill_done, previous_d_completed)
assert d_eligible == 0.004
```

到達決策、KV ready、admission 是不同事件。
維持原始到達時間，能讓這段 producer wait 出現在延遲裡。
eligibility 表示依這些前提最早可以進入；容量或其他等待仍可能把實際 admission 延後。

## 較晚、較短的 prefill 不可超越同一 D 的早到請求

規格案例對應已驗證的兩個 P、一個指定 D：A 先到，B 後到；兩者的 P 可以平行。
診斷 backend 的 prefill 成本為每 token 0.1 ms，因此 60-token A 的 prefill 較慢。

```python
arrival_order = ["A", "B"]
prefill_done_s = {"A": 0.006, "B": 0.0011}

assert sorted(prefill_done_s, key=prefill_done_s.get) == ["B", "A"]
expected_d_order = arrival_order
assert expected_d_order == ["A", "B"]
```

D 票序在到達時已固定，B 要等 A 的 D 輪次完成。
若按 prefill 完成順序交給 D，後到者可能先把 cache 填滿，阻擋較早的大請求。
這正是早期 200 筆執行停在 197 筆的修正原因。

## 完整輪次預留與容量邊界

直接引擎的小案例使用 input 32、output 16、容量 47、1-token block：

```python
needed = 32 + 16 - 1
capacity = 47
assert needed == capacity
assert capacity - needed == 0
assert needed > 46
```

D 在 admission 就預留 47 個 slots；整輪的後續輸出不必再搶新 slot，因此預期零 preemption。
這個邊界在 `test_decode_reserves_complete_turn_without_preemption` 直接驗證。

**目前另一層檢查較保守：**正式 driver 會先呼叫 `LLMEngine.validate_request_capacity()`，其 D 檢查使用 input＋output，也就是此例的 48。
因此直接引擎接受 47 的案例，不能推定經過這個前置檢查的 driver 也會接受 47。
正式 200 筆使用的容量遠大於單筆需要，沒有碰到這個一格差異。

## 只 pin 要讀的 prefix

前輪 input 10、output 100，D 已有 109 個 materialized tokens。
新輪 input 100、output 2，只繼承最前面的 10，容量為 128。

```python
old_context = 10 + 100 - 1
needed_prefix = 10
new_positions = 100 - needed_prefix
new_output_kv = 2 - 1

assert old_context == 109
assert needed_prefix + new_positions + new_output_kv == 101 <= 128
assert old_context + new_positions + new_output_kv == 200 > 128
```

舊物件退出 resident 集合後，只需為 captured reader 保留那 10 個位置。
整份舊 output suffix 都繼續鎖住會造成不必要的容量阻塞。
取消或完成時也要釋放剩餘 snapshot references，取消案例已有自動化驗證。

## 對照位置

- [實作：arrival LRU、版本 ready、範圍 pin 與 FIFO](../../TokenSim/placement/handover_clock.py)
- [實作：admission 的完整輪次配置](../../TokenSim/block/block_manager.py)
- [實作：driver 使用的容量前置檢查](../../TokenSim/llm/engine/core.py)
- [自動化測試：路由、等待與 FIFO](../../tests/test_handover_arrival.py)
- [自動化測試：容量、prefix pin 與取消](../../tests/test_handover_arrival_states.py)
