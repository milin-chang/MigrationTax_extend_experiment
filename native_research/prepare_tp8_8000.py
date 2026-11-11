"""Reuse the validated TP8 configuration with the original 8,000-request input."""

from hashlib import sha256
import json
from pathlib import Path
import shutil

from native_methods.inputs import load_requests

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "configs/servegen7b-tp8-200"
CONFIG = ROOT / "configs/servegen7b-tp8-8000"


def main():
    if CONFIG.exists():
        raise FileExistsError("Refusing to overwrite the TP8 8000 configurations")
    source = ROOT / "inputs/servegen-s0-first8000.jsonl"
    manifest = json.loads((ROOT / "inputs/subset-manifest.json").read_text())
    if sha256(source.read_bytes()).hexdigest() != manifest["subset_sha256"]:
        raise ValueError("The 8000-request input differs from its original manifest")
    rows = load_requests(source, "servegen")
    if len(rows) != 8000 or sum(row["output_tokens"] for row in rows) != 7663048:
        raise ValueError("Unexpected ServeGen population or output token count")
    shutil.copytree(PREVIOUS, CONFIG)
    for label in ("reference", "bw50", "bw100", "bw200"):
        path = CONFIG / f"study_{label}.json"
        study = json.loads(path.read_text())
        for key in ("data_root", "cluster_path", "transfer_path"):
            study[key] = study[key].replace(
                "configs/servegen7b-tp8-200/", "configs/servegen7b-tp8-8000/", 1
            )
            if not (ROOT / study[key]).exists():
                raise ValueError("Missing copied study resource: " + key)
        study["input_manifest"] = "inputs/subset-manifest.json"
        study["input_selection"] = (
            "沿用 TP1 實驗同一份 ServeGen seed 0 的 8,000 筆容量篩選子集；"
            "先依原 TP1 容量規則排除整段對話，再按原始到達順序取樣；"
            "本輪沒有新增排除、截短 token 或重縮到達時間；保留前輪相依。"
        )
        path.write_text(json.dumps(study, indent=2) + "\n")
    for source_file in PREVIOUS.rglob("*"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(PREVIOUS)
        if relative.name.startswith("study_"):
            continue
        if source_file.read_bytes() != (CONFIG / relative).read_bytes():
            raise ValueError("The copied TP8 deployment changed: " + str(relative))
    print(
        json.dumps(
            dict(
                config=str(CONFIG),
                requests=len(rows),
                returning=sum(row["turn"] > 0 for row in rows),
                input_sha256=manifest["subset_sha256"],
                prior_tp8_runtime_and_catalogs_unchanged=True,
            )
        )
    )


if __name__ == "__main__":
    main()
