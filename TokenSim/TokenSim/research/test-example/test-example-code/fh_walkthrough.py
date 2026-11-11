"""Replay real public input, then inspect the observed FH at row 242."""
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("reading_example_tokensim", HERE / "migrationtax.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
study = module.Study(seed=0)
with (Path(__file__).parent / "data/mooncake-first243.jsonl").open() as inputs:
    for line in inputs:
        request = json.loads(line)
        assert not any(k in request for k in ("cell", "P", "D", "expected_cell"))
        observed = study.step(request)

# Expected results are checked after independent decisions, never supplied to Study.
assert observed["row_index"] == 242
assert observed["cell"] == "FH"
assert (observed["P"], observed["D"]) == (3, 0)
assert observed["required_prefix_tokens"] == 7103
assert observed["p_version_tokens_before"] is None
assert observed["d_version_tokens_before"] == 7151
assert observed["footprints"]["D_tokens"] == 7216
serial = observed["methods"]["Serial"]
assert (serial["store_to_p_tokens"], serial["p_to_d_tokens"]) == (7103, 87)
assert math.isclose(serial["ttft_s"], 0.04763106494285714, abs_tol=1e-14)
assert serial["affected"]
assert not observed["methods"]["PPD"]["affected"]
assert observed["high_bandwidth"]["Serial"]["ttft_s"] > serial["reference_s"]
print(json.dumps(dict(observation=observed, native_audit=study.audit()), indent=2))
