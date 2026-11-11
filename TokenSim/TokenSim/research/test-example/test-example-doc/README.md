# Reading the arrival-study FH path

This example replays the first 243 normalized public Mooncake requests at seed zero through tokensim.
The input contains lengths and timestamps, without cache states or device answers.
The local data manifest records both the full normalized input hash and this prefix hash.

Open [the executable case](../test-example-code/fh_walkthrough.py) beside [the input](../test-example-code/data/mooncake-first243.jsonl).
Assertions inspect the result after execution: prefill device 3 lacks the inherited 7,103-token prefix, while decode device 0 retains 7,151 tokens.
The resulting FH serial path moves 7,103 tokens from the store and 87 tokens to decode.
Its 0.04763106494285714-second TTFT exceeds twice the same-row reference.
Finite high bandwidth remains distinct from zero movement.

The project study owns its cache decisions and analytical costs.
Native version headers represent whole extents; their count is not a GPU-token allocation count.
This case does not execute native serving queues or establish author-state equivalence.

Run this file with the bundle's Python environment from any working directory.
The shared [reading index](../../../../../test-examples-docs/README.md) links this case to its L0–L10 guide.
