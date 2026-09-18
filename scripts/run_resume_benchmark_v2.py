"""Run one explicitly authorized Resume Benchmark v2 arm.

The command refuses to start provider calls unless
``ALLOW_REAL_LLM_BENCHMARK=1`` is set and the frozen config matches the
manifest. Gold is loaded by the imported holdout runner only after run.json is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from debug_assistant.config import AppConfig

from experiments.resume_benchmark_v1.run_holdout_v1 import (
    _config_snapshot,
    _git_worktree_snapshot,
    _provider_health_check,
    _run_one,
)
from scripts.validate_resume_benchmark_v2 import validate as validate_benchmark

MANIFEST = ROOT / "experiments" / "resume_benchmark_v2" / "manifest.json"
TOPOLOGY = ROOT / "data" / "online_boutique" / "service_topology.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_root = manifest["dataset"]["runtime_root"]
    evaluator_root = manifest["dataset"]["evaluator_root"]
    return [
        {
            **item,
            "snapshot_path": f"{dataset_root}/{item['category']}/{item['case_number']}",
            "gold_location": f"{evaluator_root}/{item['category']}/{item['case_number']}/evaluator",
        }
        for item in manifest["cases"]
    ]


def _assert_frozen_config(app: AppConfig, manifest: dict[str, Any]) -> None:
    contract = manifest["fixed_run_contract"]
    actual = {
        "provider": app.model.provider,
        "model": app.model.planner_model,
        "critic_model": app.model.critic_model or app.model.planner_model,
        "temperature": app.model.temperature,
        "max_steps": app.harness.max_steps,
        "max_tool_calls": app.harness.max_tool_calls,
        "max_llm_calls": app.harness.max_llm_calls,
        "max_total_tokens": app.harness.max_total_tokens,
        "max_wall_time_seconds": app.harness.max_wall_time_seconds,
        "finalization_reserve_seconds": app.harness.finalization_reserve_seconds,
        "planner_timeout_seconds": app.harness.planner_llm_timeout_seconds,
        "review_timeout_seconds": app.harness.reporter_llm_timeout_seconds,
        "native_tool_calling": app.harness.features.native_tool_calling,
        "structured_reflection": app.harness.features.structured_reflection,
    }
    for key, expected in contract.items():
        if key in {"semantic_search", "knowledge_policy", "single_llm_identity", "budget_policy"}:
            continue
        if actual.get(key) != expected:
            raise RuntimeError(f"frozen config mismatch for {key}: {actual.get(key)!r} != {expected!r}")
    if app.harness.semantic_search.enabled:
        raise RuntimeError("Resume Benchmark v2 requires semantic_search=false")


def _v2_quality(eval_path: str | None) -> dict[str, Any]:
    """Project only evaluator-v2 fields into the run manifest.

    The imported legacy runner still returns ``strict_rca`` as its historical
    quality projection.  Keep that field for compatibility, but make the
    denominator-aware v2 fields the ones consumed by the benchmark reports.
    """
    if not eval_path or not Path(eval_path).exists():
        return {"status": "UNAVAILABLE", "evaluator_version": "evaluator-v2-development"}
    score = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    return {
        "status": "AVAILABLE",
        "evaluator_version": "evaluator-v2-development",
        "candidate_rca_evaluable": score.get("candidate_rca_evaluable"),
        "candidate_rca_correct": score.get("candidate_rca_correct"),
        "candidate_rca_status": score.get("candidate_rca_status"),
        "final_rca_evaluable": score.get("final_rca_evaluable"),
        "final_rca_correct": score.get("final_rca_correct"),
        "final_rca_status": score.get("final_rca_status"),
        "component_correct": score.get("final_component_correct"),
        "fault_code_correct": score.get("final_fault_correct"),
        "fault_explanation_semantic_score": score.get("fault_explanation_semantic_score"),
        "fault_explanation_semantic_status": score.get("fault_explanation_semantic_status"),
        "mechanism_evaluated": score.get("mechanism_evaluated"),
        "mechanism_correct": score.get("mechanism_correct"),
        "required_evidence_coverage": score.get("required_evidence_coverage"),
        "unsupported_claim_rate": score.get("unsupported_claim_rate"),
        "run_status": score.get("run_status"),
        "review_status": score.get("review_status"),
    }


def _resolved_config_snapshot(
    app: AppConfig, arm: str, experiment_config: str | None,
) -> dict[str, Any]:
    snapshot = _config_snapshot(app)
    snapshot["arm"] = arm
    snapshot["feature_flags"] = {
        name: bool(getattr(app.harness.features, name))
        for name in app.harness.features.__dataclass_fields__
    }
    snapshot["experiment_config"] = experiment_config
    snapshot["experiment_config_sha256"] = (
        _sha256(Path(experiment_config)) if experiment_config else None
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument(
        "--experiment-config",
        help="Optional feature/context ablation JSON; its hash is recorded in every artifact.",
    )
    args = parser.parse_args()
    if os.environ.get("ALLOW_REAL_LLM_BENCHMARK") != "1":
        print("PENDING: set ALLOW_REAL_LLM_BENCHMARK=1 to authorize provider calls")
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    app = AppConfig.from_env().apply_experiment_file(args.experiment_config)
    _assert_frozen_config(app, manifest)
    validation = validate_benchmark()
    health = _provider_health_check(app)
    if health.get("status") != "PASS":
        print(json.dumps({"status": "NOT_STARTED_PROVIDER_HEALTH_FAILED", "health": health}, ensure_ascii=False, indent=2))
        return 2

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _rows(manifest)
    resolved_config = _resolved_config_snapshot(app, args.arm, args.experiment_config)
    results = []
    for row in rows:
        result = _run_one(row, app, output_root)
        result["arm"] = args.arm
        result["run_type"] = "REAL_LLM"
        result["quality"] = _v2_quality(result.get("eval"))
        result["run_envelope"] = str(output_root / row["case_id"] / "run_envelope.json")
        envelope = {
            "schema_version": "resume-run-envelope-v2",
            "run_type": "REAL_LLM",
            "arm": args.arm,
            "case_id": row["case_id"],
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": _sha256(MANIFEST),
            "config": resolved_config,
            "run": result.get("run"),
            "eval": result.get("eval"),
            "trace": result.get("trace"),
            **_git_worktree_snapshot(),
        }
        (output_root / row["case_id"] / "run_envelope.json").write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)

    aggregate = {
        "schema_version": "resume-benchmark-v2-run",
        "run_type": "REAL_LLM",
        "arm": args.arm,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "provider_health_check": health,
        "input_validation": validation,
        "config": resolved_config,
        **_git_worktree_snapshot(),
        "case_count": len(results),
        "cases": results,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "case_count": len(results), "output": str(output_root)}, ensure_ascii=False, indent=2))
    return 0 if all(item.get("status") in {"PASS", "INCONCLUSIVE"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
