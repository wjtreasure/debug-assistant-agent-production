from __future__ import annotations

import argparse
import json
from pathlib import Path

from debug_assistant.config import AppConfig
from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.evaluation.incident import CloudOpsEvaluatorLoader, IncidentEvaluator
from debug_assistant.incidents.runtime import DiagnosisHarness, DiagnosisHarnessConfig
from debug_assistant.llm.factory import build_llm


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m debug_assistant.incidents")
    parser.add_argument("--runtime-case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluator", default="", help="Optional evaluator-only directory, loaded after diagnosis.")
    parser.add_argument("--topology", default="")
    parser.add_argument("--planner-timeout", type=float, default=None)
    parser.add_argument("--review-timeout", type=float, default=None)
    args = parser.parse_args(argv)

    app = AppConfig.from_env()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    llm = None
    evaluation_failed = False
    try:
        case = CloudOpsRuntimeCaseLoader().load(args.runtime_case)
        llm = build_llm(app.model)
        diagnosis_config = DiagnosisHarnessConfig.from_app_harness(
            app.harness,
            planner_llm_timeout_seconds=(
                args.planner_timeout if args.planner_timeout is not None
                else app.harness.planner_llm_timeout_seconds
            ),
            review_llm_timeout_seconds=(
                args.review_timeout if args.review_timeout is not None
                else app.harness.reporter_llm_timeout_seconds
            ),
            trace_dir=str(output / "traces"),
        )
        harness = DiagnosisHarness(
            llm,
            model=app.model.planner_model,
            review_model=app.model.critic_model or app.model.planner_model,
            config=diagnosis_config,
            topology_path=args.topology or None,
        )

        # Evaluator data is not loaded until the immutable Agent result exists.
        run = harness.run(case)
        (output / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
        result = {"status": run.status, "run": str(output / "run.json"), "trace": run.trace_path}
    except Exception as exc:
        failure = {
            "status": "FAILED", "error_type": type(exc).__name__, "error_message": str(exc),
        }
        (output / "run.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {**failure, "run": str(output / "run.json")}
    else:
        if args.evaluator:
            try:
                evaluator_data = CloudOpsEvaluatorLoader().load(args.evaluator)
                score = IncidentEvaluator().evaluate(run, evaluator_data)
                (output / "eval.json").write_text(score.model_dump_json(indent=2), encoding="utf-8")
                result["eval"] = str(output / "eval.json")
            except Exception as exc:
                # The diagnosis artifact is immutable once the Diagnosis phase
                # has completed. Keep evaluator failures in a separate artifact.
                evaluation_failed = True
                evaluation_error = {
                    "status": "FAILED", "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                error_path = output / "eval_error.json"
                error_path.write_text(
                    json.dumps(evaluation_error, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result.update({"evaluation_error": evaluation_error,
                               "evaluation_error_file": str(error_path)})
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Preserve the structured run/eval artifacts while making shell automation
    # observe the diagnosis outcome. A written FAILED run must not look like a
    # successful experiment merely because the CLI handled the exception.
    if evaluation_failed:
        return 1
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "INCONCLUSIVE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
