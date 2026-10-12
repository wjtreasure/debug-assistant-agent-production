"""Validate the self-contained, non-Gold public smoke assets."""

from __future__ import annotations

import json
from pathlib import Path

from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.skills.investigation import DomainAssetCatalog


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "examples" / "cloudops"


def main() -> int:
    manifest = PUBLIC_ROOT / "DOMAIN_ASSET_MANIFEST.json"
    topology = PUBLIC_ROOT / "service_topology.json"
    case_root = PUBLIC_ROOT / "case_10" / "runtime"
    catalog = DomainAssetCatalog.from_manifest(manifest)
    case = CloudOpsRuntimeCaseLoader().load(case_root)
    services = json.loads(topology.read_text(encoding="utf-8"))["services"]
    if len(catalog.skill_ids) != 12 or case.case_id != "cloudops-boutique-service-10":
        raise SystemExit("public smoke asset validation failed")
    if "checkoutservice" not in services:
        raise SystemExit("public topology validation failed")
    print("public smoke assets: PASS (12 skills, runtime case, topology)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
