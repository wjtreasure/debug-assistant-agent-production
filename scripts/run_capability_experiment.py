"""Measure static versus capability-aware Planner tool exposure.

This is a deterministic routing experiment.  It loads runtime case snapshots
only; it does not load evaluator Gold, invoke an LLM, or claim RCA quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.knowledge import CapabilitySnapshot
from debug_assistant.knowledge.router import _TOOL_CAPABILITY_REQUIREMENTS
from debug_assistant.tools.cloudops_snapshot import CloudOpsSnapshotToolRegistry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "resume_benchmark_v1" / "holdout_manifest_v2.json"
DEFAULT_TOPOLOGY = ROOT / "data" / "online_boutique" / "service_topology.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "resume_benchmark_v1" / "capability_routing" / "final_result.json"


def _load_runtime_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        {"case_id": row["case_id"], "snapshot_path": row["snapshot_path"]}
        for row in manifest["cases"]
    ]


def run(manifest_path: Path = DEFAULT_MANIFEST, topology_path: Path = DEFAULT_TOPOLOGY) -> dict[str, Any]:
    loader = CloudOpsRuntimeCaseLoader()
    cases: list[dict[str, Any]] = []
    totals = {
        "cases": 0,
        "static_visible_tools": 0,
        "dynamic_visible_tools": 0,
        "static_unavailable_tool_exposure": 0,
        "dynamic_unavailable_tool_exposure": 0,
    }
    for row in _load_runtime_rows(manifest_path):
        case = loader.load(ROOT / row["snapshot_path"])
        registry = CloudOpsSnapshotToolRegistry(case.runtime_data_dir, topology_path)
        source_available = bool(getattr(getattr(registry, "source_binding", None), "available", False))
        snapshot = CapabilitySnapshot.detect(
            case, registry, source_workspace_available=source_available,
        )
        static_names = {
            item["function"]["name"]
            for item in registry.function_schemas()
        }
        dynamic_names = set(snapshot.planner_tool_names(registry))
        unavailable_static = {
            name for name in static_names
            if name not in dynamic_names and name != "finalize_diagnosis"
        }
        # The intervention must never advertise a tool whose declared source
        # is not AVAILABLE.  This is a surface invariant, not a tool-call
        # success or diagnosis metric.
        unavailable_dynamic = {
            name for name in dynamic_names
            if name != "finalize_diagnosis"
            and name in _TOOL_CAPABILITY_REQUIREMENTS
            and not all(
                snapshot.capability_state(capability) == "AVAILABLE"
                for capability in _TOOL_CAPABILITY_REQUIREMENTS[name]
            )
        }
        row_result = {
            "case_id": row["case_id"],
            "evidence_sources": list(case.evidence_sources),
            "capability_states": dict(snapshot.states),
            "static_visible_tools": sorted(static_names),
            "dynamic_visible_tools": sorted(dynamic_names),
            "static_unavailable_tool_exposure": sorted(unavailable_static),
            "dynamic_unavailable_tool_exposure": sorted(unavailable_dynamic),
            "unavailable_tool_calls": {
                "status": "UNAVAILABLE",
                "reason": "No LLM trajectory was run in this deterministic routing experiment",
            },
        }
        cases.append(row_result)
        totals["cases"] += 1
        totals["static_visible_tools"] += len(static_names)
        totals["dynamic_visible_tools"] += len(dynamic_names)
        totals["static_unavailable_tool_exposure"] += len(unavailable_static)
        totals["dynamic_unavailable_tool_exposure"] += len(unavailable_dynamic)
    return {
        "benchmark": "capability_routing",
        "run_type": "DETERMINISTIC",
        "gold_policy": "not_loaded",
        "manifest": str(manifest_path),
        "cases": cases,
        "aggregate": totals,
        "interpretation": (
            "Static exposure is a provider-surface baseline. Dynamic exposure is a "
            "capability-state invariant; no RCA or provider-call quality claim is made."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.manifest, args.topology)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
