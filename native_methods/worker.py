"""One native condition with immutable source copies and analytical-time guards."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
from importlib.metadata import distributions
import subprocess
import sys
import traceback
from .inputs import ROOT, load_requests, manifest

SOURCES = {
    "tokensim": "TokenSim/TokenSim/research/native_runtime.py",
    "splitwise": "splitwise-sim/native_method_runtime.py",
    "simai": "SimAI/vidur-alibabacloud/vidur/shared_kv/method_runtime.py",
    "llmservingsim": "LLMServingSim/native_method_runtime.py",
}


def lock_raw_source(trace, root=ROOT):
    manifest_path = root / "Trace/sources.json"
    data = manifest_path.read_bytes()
    source = next(
        item for item in json.loads(data)["files"] if item["name"].lower() == trace
    )
    relative = "Trace/" + source["file"]
    actual = sha256((root / relative).read_bytes()).hexdigest()
    if actual != source["sha256"]:
        raise ValueError("Pinned raw source checksum mismatch: " + relative)
    return dict(
        path=relative,
        sha256=actual,
        manifest_sha256=sha256(data).hexdigest(),
        commit=source["commit"],
        download_url=source["download_url"],
        runtime_role=(
            "Raw prefix-parent reconstruction"
            if trace == "mooncake"
            else "Provenance of normalized input; runtime parents use normalized turns"
        ),
    )


def raw_source_unchanged(record, root=ROOT):
    path = root / record["path"]
    try:
        return (
            sha256(path.read_bytes()).hexdigest() == record["sha256"]
            and sha256((root / "Trace/sources.json").read_bytes()).hexdigest()
            == record["manifest_sha256"]
        )
    except OSError:
        return False


def capture(output, paths):
    hashes = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        copy = output / "source-snapshot" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists():
            if copy.read_bytes() != data:
                raise RuntimeError(
                    "Imported source changed during this condition: " + str(relative)
                )
        else:
            copy.write_bytes(data)
        hashes[str(relative)] = sha256(data).hexdigest()
    return hashes


def forbid_analytical_times():
    from algorithm.model import CostModel

    def forbidden(*args, **kwargs):
        raise RuntimeError("Analytical evaluate/CostModel cannot supply native time")

    for name in (
        "prefill_seconds",
        "move_seconds",
        "decode_step_seconds",
        "reference_seconds",
        "pipeline_cache_ready_seconds",
    ):
        setattr(CostModel, name, forbidden)
    for name in [
        "algorithm.Serial.serial",
        "algorithm.PPD.ppd",
        "algorithm.MigrationTax.dispatch",
    ]:
        __import__(name)
        sys.modules[name].evaluate = forbidden


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True)
    p.add_argument("--method", choices=["Serial", "PPD", "MigrationTax"], required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--trace", choices=["mooncake", "servegen"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--hardware-profile", type=Path)
    p.add_argument("--experiment-config", type=Path)
    p.add_argument(
        "--record-timeline",
        action="store_true",
        help="Stream TokenSim request/worker-step CSV diagnostics into the run output",
    )
    a = p.parse_args()
    if a.record_timeline and a.backend != "tokensim":
        p.error("--record-timeline requires TokenSim")
    a.output = a.output.resolve()
    if a.hardware_profile is not None:
        a.hardware_profile = a.hardware_profile.resolve()
        if a.backend not in ("tokensim", "splitwise"):
            p.error("--hardware-profile is supported only for TokenSim and Splitwise")
    if a.experiment_config is not None:
        a.experiment_config = a.experiment_config.resolve()
        if a.backend != "tokensim":
            p.error("--experiment-config requires TokenSim")
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="starting",
        backend=a.backend,
        method=a.method,
        seed=a.seed,
        input_sha256=sha256(a.input.read_bytes()).hexdigest(),
        limit=a.limit,
        record_timeline=a.record_timeline,
        analytical_evaluate_used_for_timing=False,
        analytical_evaluate_guard=True,
        environment_versions={d.metadata["Name"]: d.version for d in distributions()},
        author_answers_read=False,
        python=platform.python_version(),
        cpu_ids=sorted(os.sched_getaffinity(0)),
        baseline_git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    snapshot = list((ROOT / "native_methods").glob("*.py")) + list(
        (ROOT / "algorithm").rglob("*.py")
    )
    if a.experiment_config is not None:
        snapshot.append(a.experiment_config)
        study = json.loads(a.experiment_config.read_text())
        for key in ("cluster_path", "transfer_path"):
            snapshot.append(ROOT / study[key])
        snapshot.extend((ROOT / study["data_root"]).rglob("*.yaml"))
        snapshot.extend((ROOT / study["data_root"]).rglob("*.parquet"))
    if a.backend in SOURCES:
        if a.backend in ("tokensim", "splitwise"):
            from .hardware_profile import SPEC_PATH, load_profile

            deployment = load_profile(a.hardware_profile)
            snapshot.extend(
                [
                    a.hardware_profile or SPEC_PATH,
                    ROOT / f"TokenSim/data/models/{deployment.model_id}.yaml",
                    ROOT / "TokenSim/data/devices/h100_sxm.yaml",
                ]
            )
            if a.backend == "splitwise":
                timing_path = ROOT / deployment.splitwise_profile
                snapshot.extend(
                    [timing_path, timing_path.with_suffix(".metadata.json")]
                )
        snapshot.append(ROOT / SOURCES[a.backend])
        snapshot.extend((ROOT / SOURCES[a.backend]).parent.glob("native_*.py"))
        snapshot.extend((ROOT / SOURCES[a.backend]).parent.glob("method_*.py"))
        if a.backend == "splitwise":
            snapshot.append(ROOT / "splitwise-sim/native_method_components.py")
    report["initial_source_sha256"] = capture(a.output, snapshot)
    target = a.output / "condition.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if a.backend not in SOURCES:
            raise NotImplementedError("Native adapter not connected")
        for native in [
            "TokenSim",
            "splitwise-sim",
            "SimAI/vidur-alibabacloud",
            "LLMServingSim",
        ]:
            sys.path.insert(0, str(ROOT / native))

        def deny_author_answers(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                name = str(args[0]).lower()
                if any(
                    word in name
                    for word in (
                        "handover-package",
                        "/answers/",
                        "x1_final",
                        "m1_final",
                    )
                ):
                    raise PermissionError(
                        "Native workload conditions cannot read author answers or dumps"
                    )

        sys.addaudithook(deny_author_answers)
        forbid_analytical_times()
        path = ROOT / SOURCES[a.backend]
        spec = importlib.util.spec_from_file_location(
            "native_method_backend_" + a.backend, path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        report["raw_source"] = lock_raw_source(a.trace)
        target.write_text(json.dumps(report, indent=2) + "\n")
        rows = load_requests(a.input, a.trace, limit=a.limit)
        report["input"] = manifest(a.input, a.trace, rows)
        report["adapter_sha256"] = sha256(path.read_bytes()).hexdigest()
        target.write_text(json.dumps(report, indent=2) + "\n")
        options = (
            dict(hardware_profile=a.hardware_profile)
            if a.backend in ("tokensim", "splitwise")
            else {}
        )
        if a.experiment_config is not None:
            options["experiment_config"] = a.experiment_config
        if a.backend == "tokensim":
            options["record_timeline"] = a.record_timeline
        result = module.execute(rows, a.method, a.seed, a.output, **options)
        report.update(status=result["status"])
    except NotImplementedError as error:
        report.update(status="unsupported", reason=str(error))
    except Exception as error:
        report.update(
            status="failed", reason=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        imported = []
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if (
                    path.is_relative_to(ROOT)
                    and path.suffix == ".py"
                    and not any(
                        part.startswith(".venv")
                        for part in path.relative_to(ROOT).parts
                    )
                ):
                    imported.append(path)
        report["imported_source_sha256"] = capture(a.output, imported + snapshot)
        if "raw_source" in report:
            report["raw_source_unchanged"] = raw_source_unchanged(report["raw_source"])
            if not report["raw_source_unchanged"]:
                report.update(
                    status="failed",
                    reason="Pinned raw source or its manifest changed during native execution",
                )
        target.write_text(json.dumps(report, indent=2) + "\n")
        if report.get("raw_source_unchanged") is False:
            raise RuntimeError(report["reason"])
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("sha256")}))


if __name__ == "__main__":
    main()
