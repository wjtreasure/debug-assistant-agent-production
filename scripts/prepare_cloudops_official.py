#!/usr/bin/env python3
"""Download pinned, runtime-safe Cloud-OpsBench tool caches and verify checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "cloudops_official" / "manifest.json"


def prepare(manifest_path: Path) -> list[dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent / manifest["commit"]
    raw_base = f"https://raw.githubusercontent.com/LLM4Ops/Cloud-OpsBench/{manifest['commit']}"
    prepared = []
    for case in manifest["cases"]:
        runtime_dir = root / case["official_source_path"]
        runtime_dir.mkdir(parents=True, exist_ok=True)
        cache_url = f"{raw_base}/{case['official_source_path']}/tool_cache.json"
        payload = urlopen(cache_url, timeout=60).read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != case["tool_cache_sha256"]:
            raise RuntimeError(
                f"checksum mismatch for {case['runtime_case_id']}: {digest}"
            )
        (runtime_dir / "tool_cache.json").write_bytes(payload)
        runtime_case = {
            "case_id": case["runtime_case_id"],
            "summary": case["summary"],
            "system": "online-boutique",
            "namespace": case["namespace"],
            "evidence_sources": case["evidence_sources"],
        }
        (runtime_dir / "case.json").write_text(
            json.dumps(runtime_case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        prepared.append({
            "case_id": case["runtime_case_id"],
            "runtime_dir": str(runtime_dir),
            "tool_cache_sha256": digest,
        })
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps({"prepared": prepare(args.manifest.resolve())}, indent=2))


if __name__ == "__main__":
    main()
