"""唯一 CloudOps 命令入口。

SWE/Repository 诊断已经移到 ``debug-assistant-swe``，本包不再暴露旧的
``AgentHarness``、generic Skill 或 SWE evaluator 命令。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from debug_assistant.config import AppConfig, load_effective_config
from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.incidents.runtime import DiagnosisHarness
from debug_assistant.llm.factory import build_llm


def _run_cloudops(args: argparse.Namespace) -> int:
    base = AppConfig.from_env()
    app = load_effective_config(
        args.protocol or None,
        arm_id=args.arm,
        base=base,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = CloudOpsRuntimeCaseLoader().load(args.runtime_case)
    llm = build_llm(app.model)
    try:
        config = replace(
            app.runtime,
            trace_dir=str(output / "traces"),
            domain_asset_manifest_path=args.manifest,
        )
        harness = DiagnosisHarness(
            llm,
            model=app.model.planner_model,
            transport_profile=app.transport_profile,
            reflection_model=app.model.reflection_model,
            review_model=app.model.critic_model,
            config=config,
            topology_path=args.topology or None,
        )
        run = harness.run(case, mode=config.execution_mode)
        run_path = output / "run.json"
        run_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        result = {"status": run.status, "run": str(run_path), "trace": run.trace_path}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if run.status == "PASS" else (2 if run.status == "INCONCLUSIVE" else 1)
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()


def _validate_protocol() -> int:
    try:
        from scripts.run_final_eval import validate_protocol
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        print(
            "validate-protocol requires the full source checkout and its "
            "versioned protocol assets; use `debug-assistant run` for a "
            "public package checkout.",
        )
        return 2

    result = validate_protocol()
    print(json.dumps({
        "status": "PROTOCOL_VALID",
        "manifest_sha256": result["manifest_sha256"],
        "semantic_runtime_fingerprint": result["semantic_runtime_fingerprint"],
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="debug-assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one CloudOps incident")
    run.add_argument("--runtime-case", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--topology", default="")
    run.add_argument("--protocol", default="", help="Optional versioned ExperimentProtocol")
    run.add_argument("--arm", choices=("A0", "A1", "A2"), default="A2")
    run.set_defaults(handler=_run_cloudops)
    validate = sub.add_parser("validate-protocol", help="validate the frozen final protocol")
    validate.set_defaults(handler=lambda _args: _validate_protocol())
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
