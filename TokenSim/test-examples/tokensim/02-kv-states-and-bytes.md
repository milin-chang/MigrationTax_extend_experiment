# 六狀態與 KV 搬移量

推導前提：第二輪輸入 64 tokens，繼承 prefix 32 tokens，新增 input 32 tokens；`block_size=1`、每 token 4 bytes。
下列六列也有 [既有小案例觀測](../../../trace-study/results/tokensim-flow-v3/acceptance/observations.json) 可核對。

## 六狀態由實際覆蓋量判斷

P 的 H／F／R 分別表示 local hit、store fetch、重算。
D 的 H／F 表示有完整 prefix、缺少 prefix。

```python
from TokenSim.llm.handover_metrics import classify_prefix

# cell: P local、P store、D local、P 計算量、P→D bytes
cases = {
    "HH": (32,  0, 32, 32, 128),
    "FH": ( 0, 32, 32, 32, 128),
    "RH": ( 0,  0, 32, 64, 128),
    "HF": (32,  0,  0, 32, 256),
    "FF": ( 0, 32,  0, 32, 256),
    "RF": ( 0,  0,  0, 64, 256),
}

for expected, (p_local, p_store, d_local, computed, p2p_bytes) in cases.items():
    assert classify_prefix(32, p_local, p_store, d_local)["cell"] == expected
    assert computed == 64 - p_local - p_store
    assert p2p_bytes == (64 - d_local) * 4
```

D 只持有舊 prefix 時，還需接收 32 個新位置，因此 HH 的搬移量仍是 128 bytes。
FH 的 P 取回 32 tokens，D 已有這段 prefix；store→P 與 P→D 必須分開計量。
RH／RF 的診斷案例獨立停用 store admission，確認重算分支；主情境的無限 store 不需要自然出現 R。
六列是分支診斷，沒有用它們推估 trace 的狀態占比。

## 部分命中保留 counts

第一輪 input 32、output 18，實際 materialized context 有 49 tokens。
第二輪繼承其中 37 個位置，再加入 7 個新 input，總長度 44。

```python
prefix = 37
input_tokens = 44
p_local = 32
p_store = 5
d_local = 37

assert p_local + p_store == prefix
assert input_tokens - p_local - p_store == 7
assert (input_tokens - d_local) * 4 == 28
assert classify_prefix(prefix, p_local, p_store, d_local)["cell"] is None
```

P 同時用了 local 與 fetch，單一 H 或 F 會遺失覆蓋資訊。
預期保存四個 counts 與未分類原因。
1-token block 讓第 33～37 個位置可以精確取回，不把它們向 16-token 邊界取整。

## 輪末完整傳一次，包含已存在的 prefix

這個推導對應自動化測試中的兩輪：第一輪 input 32／output 2；第二輪 input 48／output 4，繼承 32。

```python
bytes_per_token = 4
materialized = [32 + 2 - 1, 48 + 4 - 1]
writeback_bytes = [tokens * bytes_per_token for tokens in materialized]

assert materialized == [33, 51]
assert writeback_bytes == [132, 204]
assert sum(writeback_bytes) == 336
```

第二輪的 204 bytes 包含已存在的 32-token prefix。
完整情境採輪末完整 write-back，因此每輪一次；decode step 數增加不會變成每一步都重傳 prompt。
每輪完成前都要等待這次寫回。

假設每個新 input／output 身分皆不同，store 會持續保留舊版本：

```python
unique_store_tokens = 33 + (48 - 32) + (4 - 1)
assert unique_store_tokens == 52
assert unique_store_tokens * 4 == 208
assert sum(writeback_bytes) == 336
```

store 的 unique 儲存量是 208 bytes，累計傳輸量是 336 bytes。
前輪的一個 output KV 仍留在無限 store；它沒有成為第二輪的 input prefix，所以 unique 物件數可以大於第二輪的 51-token context。
兩個數字回答不同問題，不應拿其中一個代替另一個。

## 第一輪沒有繼承 prefix

```python
assert classify_prefix(0, 0, 0, 0)["cell"] is None
```

第一輪的 cold prefill 需要計算，但六狀態比較以有繼承 prefix 的返回情況為核心。
預期保留「no inherited prefix」原因，避免把第一輪直接混進 RF 的比例。

## 對照位置

- [實作：六狀態分類](../../TokenSim/llm/handover_metrics.py)
- [實作：輪末寫回](../../TokenSim/llm/handover_writeback.py)
- [自動化測試](../../tests/test_handover_arrival_states.py)：`test_six_cells_and_exact_payloads`、`test_prefix_crosses_prompt_output_boundary_without_rounding`、`test_turn_end_sends_full_context_once_including_existing_prefix`。
