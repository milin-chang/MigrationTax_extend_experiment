"""Reuse only completed native evidence with identical inputs and code semantics."""

import ast
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
from .inputs import ROOT

ENVELOPES = {
    "native_methods/campaign.py",
    "native_methods/limited_job.py",
    "migrationtax/limited_job.py",
}


def completed_conditions(campaign):
    campaign = Path(campaign).resolve()
    data = json.loads((campaign / "campaign.json").read_text())
    return {
        (c["backend"], c["trace"], c["method"], c["seed"]): (campaign, c)
        for c in data["conditions"]
        if c["status"] == "completed"
    }


def copy_verified(previous, current, target):
    base, condition = previous
    source = base / condition["output"]
    record = json.loads((source / "condition.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if condition["input_sha256"] != current["input_sha256"]:
        return None
    if (
        summary.get("status") != "completed"
        or summary.get("completed") != current["expected_requests"]
    ):
        return None
    if record.get("environment_versions") != {
        d.metadata["Name"]: d.version for d in distributions()
    }:
        return None
    comparisons = {}
    for relative, digest in record["imported_source_sha256"].items():
        if relative in ENVELOPES:
            continue
        old = source / "source-snapshot" / relative
        now = ROOT / relative
        if not old.is_file() or not now.is_file():
            return None
        data = old.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("Historical source snapshot checksum mismatch")
        if sha256(now.read_bytes()).hexdigest() == digest:
            comparisons[relative] = "byte_identical"
        elif ast.dump(ast.parse(data), include_attributes=False) == ast.dump(
            ast.parse(now.read_bytes()), include_attributes=False
        ):
            comparisons[relative] = "AST_identical_formatting_only"
        else:
            return None
    config = json.loads((source / "effective-config.json").read_text())
    profile = Path(config["profile"])
    if sha256(profile.read_bytes()).hexdigest() != config["profile_sha256"]:
        return None
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, target / "output")
    old_job = base / condition["job"]
    shutil.copytree(old_job.parent, target / "job")
    proof = dict(
        source_campaign=str(base),
        source_condition=condition,
        rule="Completed full condition; exact input/environment/profile; project code byte-identical or AST-identical formatting.",
        code_comparison=comparisons,
        historical_timeout_seconds=json.loads(old_job.read_text())["timeout_seconds"],
        current_formal_timeout_seconds=None,
        historical_receipts_unchanged=True,
    )
    (target / "reuse.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof
