"""Self-contained public checkout smoke; no Provider or benchmark labels."""

import json
from pathlib import Path

from debug_assistant.datasets.cloudops_incident import CloudOpsRuntimeCaseLoader
from debug_assistant.skills.investigation import DomainAssetCatalog


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "examples" / "cloudops"


def test_public_runtime_assets_load_without_external_dataset():
    manifest = PUBLIC_ROOT / "DOMAIN_ASSET_MANIFEST.json"
    topology = PUBLIC_ROOT / "service_topology.json"
    case_root = PUBLIC_ROOT / "case_10" / "runtime"

    catalog = DomainAssetCatalog.from_manifest(manifest)
    case = CloudOpsRuntimeCaseLoader().load(case_root)
    services = json.loads(topology.read_text(encoding="utf-8"))["services"]

    assert len(catalog.skill_ids) == 12
    assert case.case_id == "cloudops-boutique-service-10"
    assert "checkoutservice" in services
