# 首 token 與完成時間

這份案例同時保留既有觀測與手算，兩者的模型尺度分開標示。

## 既有觀測：首 token 要晚於 P→D 與一次 D step

下面摘自 [六狀態診斷的 HH 記錄](../../../trace-study/results/tokensim-flow-v3/acceptance/observations.json)。
時間是相對於該輪到達的秒數；微型模型每 token 4 bytes，固定 D step 為 1 ms。

```python
from math import isclose

prefill_done = 0.001
handoff_ready = 0.0010120023841857914
first_output = 0.0020120023841857914
first_d_step = 0.001

assert prefill_done < handoff_ready < first_output
assert isclose(first_output, handoff_ready + first_d_step, abs_tol=1e-12)
```

Prefill 完成後，KV 還在搬移；D 收齊後仍需執行產生第一個輸出的 query。
`source_first_token_s` 在這個選用模式記錄 D 的首次 sample。
保存的 `prefill_completion_after_arrival_s` 是另一個終點；它也不等於另外重跑未修改原生模式所得的 TTFT，因為排程已不同。

## 只輸出一個 token 也要走完這條路

```python
input_tokens = 32
output_tokens = 1
d_prefix_tokens = 0
bytes_per_token = 4

expected_first_output_role = "decode"
expected_p2p_bytes = (input_tokens - d_prefix_tokens) * bytes_per_token
expected_materialized_tokens = input_tokens + output_tokens - 1

assert expected_first_output_role == "decode"
assert expected_p2p_bytes == 128
assert expected_materialized_tokens == 32
```

這是完整情境模式的契約。
P 只完成 prompt construction，沒有先 sample 輸出；即使 output length 為 1，仍必須有 handoff 與 D step。
原生預設模式會在 P sample 第一個 token，其限制反例繼續保留在原自動化測試中。

## 手算：共同成本模式下的冷啟動 1k

前提：7B MHA、524288 bytes/token、A 頻寬 100 GB/s；單一請求、沒有 queue、fetch、offload 或固定鏈路延遲。
這是公式推導，不是真實 GPU 量測。

```python
from math import isclose
from TokenSim.latency.handover import prefill_seconds, decode_seconds

C = 1000
fixed_overhead = 0.001
pf = prefill_seconds(C)
st = decode_seconds(C)
p2p = C * 524288 / 100_000_000_000

assert isclose(pf, 0.05180166666666667)
assert isclose(st, 0.003117)
assert isclose(p2p, 0.00524288)

first_token = fixed_overhead + pf + p2p + st
assert isclose(first_token, 0.06116154666666667)
```

公式中的 1 ms 與 ST 各出現一次。
完整模式輸出的 first-token 時間已包含這些事件，不再把 ST 加到輸出欄位上。
有 queue 或 producer wait 時，應由實際事件增加等待時間，不靠這個無等待手算式補猜。

## output 完成後還有寫回

```python
# 延續上面的無等待 1k 例子，這次只輸出 1 token。
materialized_tokens = C + 1 - 1
writeback = materialized_tokens * 524288 / 100_000_000_000
completed = first_token + writeback

assert completed > first_token
assert isclose(completed, 0.06640442666666667)
```

此時第一個輸出也是最後一個輸出，但 store 尚須收到完整 materialized KV。
完成事件等寫回可見後才發出，P 的輪內保留也維持到這個終點。
本模式的 KV 範圍為 input＋output－1；最後剛 sample 的 token 尚未被 forward，不捏造它的 KV。

## 對照位置

- [實作：Prefill 不 sample、D 首步與完成事件](../../TokenSim/llm/handover_execution.py)
- [實作：寫回完成後才完成請求](../../TokenSim/llm/handover_writeback.py)
- [自動化測試：單輸出 token](../../tests/test_handover_flow.py)：`test_single_output_waits_for_transfer_and_one_d_step`。
- [自動化測試：四個 context 與 chunk 成本](../../tests/test_handover_cost.py)。
