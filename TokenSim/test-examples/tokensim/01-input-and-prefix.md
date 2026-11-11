# 輸入與 prefix 身分

這份案例處理輸入契約，不啟動模擬器。
完整資料在 [data/two-turns.json](data/two-turns.json)，欄位採閱讀用命名，並非 TokenSim loader 的直接輸入格式。

## 同一對話的第二輪

```python
import json
from pathlib import Path

example = json.loads(
    Path("test-examples/tokensim/data/two-turns.json").read_text()
)
previous, current = example["turns"]

assert current["conversation_id"] == previous["conversation_id"]
assert current["turn"] == previous["turn"] + 1
assert current["prefix_tokens"] == 7 - 3 == 4
assert len(current["input_identities"]) == current["input_tokens"] == 7

prefix = current["prefix_tokens"]
previous_kv_identities = (
    previous["input_identities"]
    + previous["materialized_output_identities"]
)
assert current["input_identities"][:prefix] == previous_kv_identities[:prefix]
assert current["input_identities"][prefix:] == ["x", "y", "z"]
```

預期可繼承四個位置，原因是身分序列的前四個位置相同。
只有「長度剛好等於四」不足以證明可以重用 KV。
完整前輪關係可涉及已生成的 output KV；這個簡例只繼承前輪 input，跨 input/output 的案例見下一份文件。

對應到 `TokenSim.llm.llm_request.Request` 時，閱讀資料的 `input_tokens`、`output_tokens`、`conversation_id` 分別對應 `prefill_len`、`decode_len`、`chat_id`。
在完整 arrival 模式的 1-token block 設定下，`hash_ids` 每個元素代表一個輸入位置的身分。

## 到達時間保留原始相對差

```python
original_arrivals_s = [100.0, 100.0002]
origin = original_arrivals_s[0]
relative_arrivals_s = [t - origin for t in original_arrivals_s]

assert relative_arrivals_s[0] == 0.0
assert abs(relative_arrivals_s[1] - 0.0002) < 1e-12
```

移除共同時間原點保留了間隔。
第二輪的到達時間仍是 0.2 ms；前輪若尚未完成，等待會進入模擬時序，不把輸入時間直接改成前輪完成時間。

## 錯誤的前輪關係應被拒絕

```python
parent = ["a", "b", "c", "d"]
incoming = ["a", "b", "wrong", "d", "new"]
claimed_prefix = 4

assert incoming[:claimed_prefix] != parent[:claimed_prefix]
# 預期：ConfigurationError，不能沿用這四個位置的 KV。
```

這是宣稱的「前一輪」不相符，不能靠輸入／輸出長度補猜 token 身分。

## 某個裝置的舊副本分歧，仍可有部分命中

```python
latest_parent = ["a", "b", "x", "y"]
incoming = ["a", "b", "x", "y", "new"]
older_device_copy = ["a", "b", "c", "d"]
claimed_prefix = 4

assert incoming[:claimed_prefix] == latest_parent

common = 0
for cached, needed in zip(older_device_copy, incoming[:claimed_prefix]):
    if cached != needed:
        break
    common += 1

assert common == 2
```

整份 trace 的前輪關係有效；這個裝置只是持有較舊版本。
預期回報兩個 local tokens，讓路由繼續找完整 holder，或為缺少的部分取回／重算。
這兩種情況的區別，決定我們應拒絕輸入，還是保留可用的部分命中。

## 對照位置

- [實作：arrival 版本、前輪驗證與共同 prefix](../../TokenSim/placement/handover_clock.py)
- [自動化測試：舊副本分歧與錯誤前輪](../../tests/test_handover_arrival_states.py)：`test_older_divergent_copy_reports_only_its_common_prefix`、`test_invalid_preceding_turn_is_rejected`。
- [正式 trace 輸入轉換](../../../trace-study/run_tokensim_flow_v3.py)：只處理已確認的來源記錄，使用可追溯的合成身分。
