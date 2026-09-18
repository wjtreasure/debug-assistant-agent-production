"""Validate Resume Benchmark v2 inputs without starting a provider run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.evaluation.incident import CloudOpsEvaluatorLoader


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "resume_benchmark_v2" / "manifest.json"


def _snapshot_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _assert_no_case_specific_runtime_literals(cases: list[dict[str, Any]]) -> int:
    """Reject benchmark case IDs embedded in runtime diagnosis code."""
    runtime_files = sorted((ROOT / "src" / "debug_assistant").rglob("*.py"))
    hits = []
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        for item in cases:
            case_id = str(item["case_id"])
            if case_id in text:
                hits.append(f"{path.relative_to(ROOT)}:{case_id}")
    if hits:
        raise ValueError("case-specific runtime literals found: " + ", ".join(hits))
    return len(runtime_files)


def validate() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    categories = [str(item["category"]) for item in cases]
    if len(cases) != 12 or {category: categories.count(category) for category in set(categories)} != {
        "codedefect": 3, "performance": 3, "runtime": 3, "service": 3,
    }:
        raise ValueError("benchmark must contain exactly 3 cases in each of 4 categories")

    dataset_root = ROOT / manifest["dataset"]["runtime_root"]
    evaluator_root = ROOT / manifest["dataset"]["evaluator_root"]
    runtime_source_file_count = _assert_no_case_specific_runtime_literals(cases)
    loader = CloudOpsRuntimeCaseLoader()
    evaluator_loader = CloudOpsEvaluatorLoader()
    runtime_paths: list[Path] = []
    checked: list[dict[str, str]] = []
    for item in cases:
        category = str(item["category"])
        number = str(item["case_number"])
        runtime_dir = dataset_root / category / number
        evaluator_dir = evaluator_root / category / number / "evaluator"
        runtime_paths.extend(runtime_dir.rglob("*"))
        loader.load(runtime_dir)
        evaluator_loader.load(evaluator_dir)
        checked.append({
            "case_id": str(item["case_id"]),
            "runtime": str(runtime_dir.relative_to(ROOT)),
            "evaluator": str(evaluator_dir.relative_to(ROOT)),
        })

    actual_hash = _snapshot_hash(runtime_paths + [
        path
        for item in cases
        for path in (evaluator_root / str(item["category"]) / str(item["case_number"])).rglob("*")
    ])
    expected_hash = manifest["dataset"]["selected_runtime_and_evaluator_snapshot_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"snapshot hash mismatch: expected {expected_hash}, got {actual_hash}")

    return {
        "status": "PASS",
        "case_count": len(checked),
        "gold_boundary": manifest["evaluator"]["gold_boundary"],
        "case_specific_runtime_literals": {
            "status": "PASS",
            "scanned_python_files": runtime_source_file_count,
        },
        "snapshot_sha256": actual_hash,
        "checked": checked,
        "provider_run_started": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
