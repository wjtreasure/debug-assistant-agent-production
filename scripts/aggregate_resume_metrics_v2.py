"""Aggregate completed Resume Benchmark v2 run artifacts.

This script is deliberately post-run only.  It never loads runtime cases or
calls a provider, and it never turns an absent evaluator dimension into a
failed observation.  Missing runs are reported as ``UNAVAILABLE``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "experiments" / "resume_benchmark_v2" / "manifest.json"
sys.path.insert(0, str(ROOT))

from debug_assistant.evaluation.incident import canonicalize_fault_code


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_artifact(value: str | None, run_root: Path, fallback: str) -> Path | None:
    if value:
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for base in (ROOT, run_root):
            resolved = base / candidate
            if resolved.exists():
                return resolved
    candidate = run_root / fallback
    return candidate if candidate.exists() else None


def _ratio(rows: list[dict[str, Any]], denominator: str, numerator: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get(denominator) is True]
    correct = sum(row.get(numerator) is True for row in eligible)
    return {
        "value": (correct / len(eligible)) if eligible else None,
        "numerator": correct,
        "denominator": len(eligible),
        "status": "AVAILABLE" if eligible else "UNAVAILABLE",
    }


def _mean(rows: list[dict[str, Any]], field: str, *, status_field: str | None = None) -> dict[str, Any]:
    values = []
    for row in rows:
        if status_field is not None:
            marker = row.get(status_field)
            if marker is not True and marker != "AVAILABLE":
                continue
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return {
        "value": statistics.fmean(values) if values else None,
        "denominator": len(values),
        "status": "AVAILABLE" if values else "UNAVAILABLE",
    }


def _distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _macro_f1(rows: list[dict[str, Any]], gold_field: str, predicted_field: str) -> dict[str, Any]:
    pairs = [
        (
            canonicalize_fault_code(str(row[gold_field])),
            canonicalize_fault_code(str(row.get(predicted_field) or "")) or "__MISSING__",
        )
        for row in rows
        if row.get(gold_field)
    ]
    if not pairs:
        return {"value": None, "denominator": 0, "status": "UNAVAILABLE"}
    labels = sorted({label for pair in pairs for label in pair})
    scores = []
    for label in labels:
        true_positive = sum(gold == label and predicted == label for gold, predicted in pairs)
        false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
        false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return {
        "value": statistics.fmean(scores),
        "denominator": len(pairs),
        "labels": labels,
        "status": "AVAILABLE",
    }


def _rate_from_values(values: list[bool]) -> dict[str, Any]:
    return {
        "value": (sum(values) / len(values)) if values else None,
        "numerator": sum(values),
        "denominator": len(values),
        "status": "AVAILABLE" if values else "UNAVAILABLE",
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["wall_clock_seconds"])
        for row in rows
        if isinstance(row.get("wall_clock_seconds"), (int, float))
    ]
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "denominator": len(values),
        "status": "AVAILABLE" if values else "UNAVAILABLE",
    }


def _metrics(evals: list[dict[str, Any]], run_rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    run_statuses = [row.get("run_status", row.get("status", "UNKNOWN")) for row in run_rows]
    review_statuses = [row.get("review_status", "UNKNOWN") for row in evals]
    efficiency = {
        field: _mean(evals, field)
        for field in ("steps", "tool_calls", "llm_calls", "tokens", "prompt_tokens")
    }
    efficiency["duplicate_tool_call_rate"] = _mean(evals, "_duplicate_tool_call_rate")
    efficiency["wall_clock_seconds"] = _latency(evals)
    return {
        "sample": {
            "expected_cases": expected,
            "run_records": len(run_rows),
            "evaluator_records": len(evals),
            "status": "AVAILABLE" if len(run_rows) == expected else "PARTIAL",
        },
        "primary": {
            "candidate_rca": _ratio(evals, "candidate_rca_evaluable", "candidate_rca_correct"),
            "final_rca": _ratio(evals, "final_rca_evaluable", "final_rca_correct"),
            "mechanism": _ratio(evals, "mechanism_evaluated", "mechanism_correct"),
        },
        "quality": {
            "component": _ratio(evals, "_component_evaluable", "component_correct"),
            "fault_code": _ratio(evals, "_fault_code_evaluable", "fault_code_correct"),
            "fault_macro_f1": _macro_f1(evals, "gold_fault_label", "final_fault_label"),
            "semantic_explanation_score": _mean(
                evals, "fault_explanation_semantic_score", status_field="_explanation_available"
            ),
            "required_evidence_coverage": _mean(evals, "required_evidence_coverage"),
            "unsupported_claim_rate": _mean(evals, "unsupported_claim_rate"),
        },
        "operations": {
            "run_status": _distribution(run_statuses),
            "review_status": _distribution(review_statuses),
            "task_completion_rate": _rate_from_values(
                [value == "PASS" for value in run_statuses]
            ),
            "candidate_completion_rate": _rate_from_values(
                [row.get("candidate_present") is True for row in evals]
            ),
            "review_completion_rate": _rate_from_values(
                [value in {"PASS", "REJECT"} for value in review_statuses]
            ),
            "inconclusive_rate": {
                "value": (sum(value == "INCONCLUSIVE" for value in run_statuses) / len(run_statuses))
                if run_statuses else None,
                "denominator": len(run_statuses),
                "status": "AVAILABLE" if run_statuses else "UNAVAILABLE",
            },
            "review_attempt_rate": {
                "value": (sum(value != "NOT_RUN" for value in review_statuses) / len(review_statuses))
                if review_statuses else None,
                "denominator": len(review_statuses),
                "status": "AVAILABLE" if review_statuses else "UNAVAILABLE",
            },
        },
        "efficiency": efficiency,
    }


def _load_arm(spec: str, benchmark: dict[str, Any]) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"arm must use NAME=PATH: {spec}")
    arm, raw_root = spec.split("=", 1)
    run_root = Path(raw_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root = run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    base = {
        "arm": arm,
        "run_root": str(run_root),
        "expected_cases": len(benchmark["cases"]),
        "status": "UNAVAILABLE",
        "reason": "run_manifest.json not found",
    }
    if not manifest_path.exists():
        return base

    run_manifest = _read_json(manifest_path)
    expected_cases = {str(item["case_id"]) for item in benchmark["cases"]}
    actual_cases = {str(item.get("case_id")) for item in run_manifest.get("cases", [])}
    if run_manifest.get("run_type") != "REAL_LLM":
        return {**base, "status": "INVALID", "reason": "run_type is not REAL_LLM"}
    if run_manifest.get("manifest_sha256") != _sha256(BENCHMARK):
        return {**base, "status": "INVALID", "reason": "benchmark manifest hash mismatch"}
    if actual_cases != expected_cases:
        return {
            **base,
            "status": "INVALID",
            "reason": "case denominator mismatch",
            "missing_cases": sorted(expected_cases - actual_cases),
            "unexpected_cases": sorted(actual_cases - expected_cases),
        }

    evals: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    missing_evaluator: list[str] = []
    for record in run_manifest["cases"]:
        case_id = str(record["case_id"])
        case_dir = run_root / case_id
        eval_path = _resolve_artifact(record.get("eval"), run_root, f"{case_id}/eval.json")
        run_path = _resolve_artifact(record.get("run"), run_root, f"{case_id}/run.json")
        envelope_path = case_dir / "run_envelope.json"
        if run_path is not None:
            run_data = _read_json(run_path)
            run_rows.append({
                "case_id": case_id,
                "status": record.get("status", run_data.get("status", "UNKNOWN")),
                "run_status": run_data.get("status", record.get("status", "UNKNOWN")),
            })
        if eval_path is None:
            missing_evaluator.append(case_id)
            continue
        score = _read_json(eval_path)
        required = (
            "candidate_rca_evaluable", "candidate_rca_correct",
            "final_rca_evaluable", "final_rca_correct",
            "mechanism_evaluated", "mechanism_correct",
        )
        if any(key not in score for key in required):
            return {
                **base,
                "status": "INVALID",
                "reason": f"evaluator-v2 fields missing for {case_id}",
            }
        score["_component_evaluable"] = "final_component_correct" in score
        score["_fault_code_evaluable"] = "final_fault_correct" in score
        score["component_correct"] = score.get("final_component_correct")
        score["fault_code_correct"] = score.get("final_fault_correct")
        score["_explanation_available"] = score.get("fault_explanation_semantic_status") == "AVAILABLE"
        score["_duplicate_tool_call_rate"] = (
            score.get("duplicate_tool_calls_executed", 0) / score["tool_calls"]
            if isinstance(score.get("tool_calls"), (int, float)) and score["tool_calls"] > 0
            else None
        )
        score["wall_clock_seconds"] = (
            score.get("efficiency", {}) or {}
        ).get("wall_clock_seconds")
        score["llm_calls"] = (score.get("efficiency", {}) or {}).get("llm_calls")
        score["prompt_tokens"] = None
        trace_path = _resolve_artifact(record.get("trace"), run_root, f"{case_id}/traces/trace.jsonl")
        if trace_path is not None:
            prompt_tokens = 0
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "LLM_CALL_USAGE":
                    payload = event.get("payload") or {}
                    prompt_tokens += int(
                        payload.get("actual_prompt_tokens", payload.get("prompt_tokens", 0)) or 0
                    )
            score["prompt_tokens"] = prompt_tokens
        score["case_id"] = case_id
        evals.append(score)
        if not envelope_path.exists():
            return {**base, "status": "INVALID", "reason": f"RunEnvelope missing for {case_id}"}

    status = "COMPLETE" if len(run_rows) == len(expected_cases) and not missing_evaluator else "PARTIAL"
    return {
        "arm": arm,
        "run_root": str(run_root),
        "status": status,
        "reason": "all 12 run and evaluator artifacts loaded" if status == "COMPLETE" else "some evaluator artifacts unavailable",
        "missing_evaluator_cases": missing_evaluator,
        "metrics": _metrics(evals, run_rows, len(expected_cases)),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Resume Benchmark v2 Metrics",
        "",
        "This report is generated only from post-run artifacts. `UNAVAILABLE` is not zero.",
        "",
        "| Arm | Status | N | Task | Candidate | Component | Fault | Macro-F1 | Final RCA | Evidence | Unsupported | Tool calls | LLM calls | Prompt tokens | Total tokens | Latency p50/p95 | Inconclusive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        metrics = arm.get("metrics", {})
        sample = metrics.get("sample", {})
        primary = metrics.get("primary", {})
        quality = metrics.get("quality", {})
        operations = metrics.get("operations", {})
        efficiency = metrics.get("efficiency", {})

        def fmt(section: dict[str, Any], name: str) -> str:
            value = section.get(name, {}).get("value")
            return "UNAVAILABLE" if value is None else f"{value:.3f}"

        def fmt_mean(name: str) -> str:
            return fmt(efficiency, name)

        latency = efficiency.get("wall_clock_seconds", {})
        p50 = latency.get("p50")
        p95 = latency.get("p95")
        latency_text = (
            "UNAVAILABLE" if p50 is None or p95 is None
            else f"{p50:.1f}/{p95:.1f}s"
        )

        lines.append(
            f"| {arm['arm']} | {arm['status']} | {sample.get('expected_cases', arm.get('expected_cases', 0))} "
            f"| {fmt(operations, 'task_completion_rate')} "
            f"| {fmt(operations, 'candidate_completion_rate')} "
            f"| {fmt(quality, 'component')} | {fmt(quality, 'fault_code')} "
            f"| {fmt(quality, 'fault_macro_f1')} | {fmt(primary, 'final_rca')} "
            f"| {fmt(quality, 'required_evidence_coverage')} "
            f"| {fmt(quality, 'unsupported_claim_rate')} "
            f"| {fmt_mean('tool_calls')} | {fmt_mean('llm_calls')} "
            f"| {fmt_mean('prompt_tokens')} | {fmt_mean('tokens')} "
            f"| {latency_text} | {fmt(operations, 'inconclusive_rate')} |"
        )
    if report.get("comparisons"):
        lines.extend(["", "## Paired deltas", "", "| Baseline | Final | Metric | Before | After | Delta | Status |", "|---|---|---|---:|---:|---:|---:|"])
        for item in report["comparisons"]:
            def fmt_value(value: Any) -> str:
                return "UNAVAILABLE" if value is None else f"{value:.3f}"

            lines.append(
                f"| {item['baseline_arm']} | {item['final_arm']} | {item['metric']} "
                f"| {fmt_value(item['before'])} | {fmt_value(item['after'])} "
                f"| {fmt_value(item['delta'])} | {item['status']} |"
            )
    lines.extend(["", "## Machine-readable report", "", "See the JSON report generated alongside this file."])
    return "\n".join(lines) + "\n"


def _value_at(value: dict[str, Any], path: str) -> float | None:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, (int, float)) and not isinstance(current, bool) else None


def _comparisons(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(arms) < 2:
        return []
    paths = (
        "operations.task_completion_rate.value",
        "operations.candidate_completion_rate.value",
        "quality.component.value",
        "quality.fault_code.value",
        "quality.fault_macro_f1.value",
        "primary.final_rca.value",
        "quality.required_evidence_coverage.value",
        "quality.unsupported_claim_rate.value",
        "efficiency.tool_calls.value",
        "efficiency.llm_calls.value",
        "efficiency.prompt_tokens.value",
        "efficiency.tokens.value",
        "efficiency.wall_clock_seconds.p50",
        "efficiency.wall_clock_seconds.p95",
    )
    comparisons = []
    baseline = arms[0]
    for final in arms[1:]:
        for path in paths:
            before = _value_at(baseline.get("metrics", {}), path)
            after = _value_at(final.get("metrics", {}), path)
            comparisons.append({
                "baseline_arm": baseline["arm"],
                "final_arm": final["arm"],
                "metric": path,
                "before": before,
                "after": after,
                "delta": (after - before) if before is not None and after is not None else None,
                "status": "AVAILABLE" if before is not None and after is not None else "UNAVAILABLE",
            })
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()
    benchmark = _read_json(BENCHMARK)
    report = {
        "schema_version": "resume-metrics-v2",
        "benchmark": str(BENCHMARK.relative_to(ROOT)),
        "benchmark_sha256": _sha256(BENCHMARK),
        "arms": [_load_arm(spec, benchmark) for spec in args.arm],
    }
    report["comparisons"] = _comparisons(report["arms"])
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(encoded, encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_markdown(report), encoding="utf-8")
    print(encoded, end="")
    return 0 if all(arm["status"] == "COMPLETE" for arm in report["arms"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
