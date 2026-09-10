#!/usr/bin/env python3
"""No-LLM smoke test for pinned official Cloud-OpsBench snapshots."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.tools.cloudops_snapshot import CloudOpsSnapshotToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "cloudops_official" / "manifest.json"
TOPOLOGY_PATH = PROJECT_ROOT / "data" / "online_boutique" / "service_topology.json"


def main() -> None:
    started = time.monotonic()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = []
    describe_targets = {
        "cloudops-boutique-service-10": "checkoutservice-8445f8b6cb-pvh7z",
        "cloudops-boutique-runtime-22": "adservice-74c7f4c787-8g8cs",
        "cloudops-boutique-performance-1": "adservice-d68dfc56-lbcrz",
        "cloudops-boutique-codedefect-5": "checkoutservice-6cd8767b6d-jx7k6",
    }
    for item in manifest["cases"]:
        runtime_dir = MANIFEST_PATH.parent / manifest["commit"] / item["official_source_path"]
        cache_path = runtime_dir / "tool_cache.json"
        payload = cache_path.read_bytes()
        cache = json.loads(payload)
        observation_entries = sum(1 for key in cache if ":" in key)
        assert observation_entries == item["official_observation_entries"]
        assert hashlib.sha256(payload).hexdigest() == item["tool_cache_sha256"]
        case = CloudOpsRuntimeCaseLoader().load(runtime_dir)
        tools = CloudOpsSnapshotToolRegistry(runtime_dir, TOPOLOGY_PATH)
        resources = tools.get("get_resources").execute(
            resource_type="pods", namespace="boutique", name="",
        )
        described = tools.get("describe_resource").execute(
            resource_type="pods", namespace="boutique",
            name=describe_targets[item["runtime_case_id"]],
        )
        unavailable = tools.get("describe_resource").execute(
            resource_type="pods", namespace="boutique", name="not-captured",
        )
        assert resources.ok and described.ok
        assert not unavailable.ok
        assert unavailable.metadata["status"] == "UNAVAILABLE"
        assert unavailable.metadata["semantic_negative"] is False
        results.append({
            "case_id": case.case_id,
            "observation_entries": observation_entries,
            "get_resources": "AVAILABLE",
            "describe_resource": "AVAILABLE",
            "snapshot_miss": "UNAVAILABLE",
        })
    print(json.dumps({
        "elapsed_seconds": round(time.monotonic() - started, 4), "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
