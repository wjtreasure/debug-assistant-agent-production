"""Run a deterministic bounded-convergence comparison.

The trajectory is intentionally semantic-state only: no evaluator Gold, LLM,
or RCA correctness is involved.  It measures whether an unchanged supported
hypothesis reaches an explainable convergence exit before a fixed-step arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from debug_assistant.harness.convergence import ConvergenceController, ProgressKind


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "resume_benchmark_v1" / "convergence_control" / "final_result.json"


def _hypothesis() -> dict[str, object]:
    return {
        "status": "supported",
        "supporting_evidence_ids": ["ev-001"],
        "evidence_sufficient": True,
        "required_missing_evidence": [],
        "contradicting_evidence_ids": [],
        "stable_diagnosis_transitions": 1,
        "diagnosis_fingerprint": "same-diagnosis",
        "required_gap_fingerprint": "no-gap",
        "updated_step": 1,
    }


def run(*, baseline_max_steps: int = 8, no_progress_limit: int = 2) -> dict[str, object]:
    baseline = {
        "arm": "fixed_max_steps",
        "steps": baseline_max_steps,
        "tool_calls": baseline_max_steps,
        "no_progress_calls": baseline_max_steps - 1,
        "tokens": {"status": "UNAVAILABLE", "reason": "no provider trajectory"},
        "termination_reason": "max_steps",
        "premature_stop": False,
        "diagnostic_quality": {"status": "UNAVAILABLE", "reason": "no LLM or Gold"},
    }
    controller = ConvergenceController(no_progress_limit=no_progress_limit)
    hypothesis = _hypothesis()
    steps = 0
    no_progress_calls = 0
    termination_reason = "max_steps"
    for step in range(1, baseline_max_steps + 1):
        steps = step
        hypothesis["updated_step"] = step
        assessment = controller.assess_reflection(hypothesis)
        if assessment.kind is ProgressKind.NO_PROGRESS:
            no_progress_calls += 1
        if controller.state.forced_finalization:
            termination_reason = "convergence_forced_finalization"
            break
    optimized = {
        "arm": "convergence_controller",
        "steps": steps,
        "tool_calls": steps,
        "no_progress_calls": no_progress_calls,
        "tokens": {"status": "UNAVAILABLE", "reason": "no provider trajectory"},
        "termination_reason": termination_reason,
        "premature_stop": termination_reason != "convergence_forced_finalization",
        "diagnostic_quality": {"status": "UNAVAILABLE", "reason": "no LLM or Gold"},
        "controller_state": {
            "mode": controller.state.mode.value,
            "no_progress_streak": controller.state.no_progress_streak,
            "forced_finalization": controller.state.forced_finalization,
        },
    }
    return {
        "benchmark": "convergence_control",
        "run_type": "DETERMINISTIC",
        "trajectory": "same supported hypothesis repeated without semantic progress",
        "control": {"baseline_max_steps": baseline_max_steps, "no_progress_limit": no_progress_limit},
        "baseline": baseline,
        "optimized": optimized,
        "interpretation": (
            "The intervention bounds a repeated supported state with an explicit exit reason; "
            "no diagnostic-quality claim is made without provider inference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "optimized": result["optimized"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
