"""Invocation/provenance boundary for the unmodified official snapshot tools."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any
import hashlib

from debug_assistant._vendor.cloudops_snapshot import KubernetesTools


OFFICIAL_SNAPSHOT_SHA256 = "9a1178a2e49c2154a7595e219c549edb6870b0896eb0989d53cd5ef38d70f06b"
OFFICIAL_COMMIT = "d79f1526a4c3315004c09457fbec588eb257dd1c"


class _ObservedCache(dict):
    """Record acquisition provenance without implementing cache lookup policy."""

    def __init__(self, cache):
        super().__init__(cache)
        self.hits: set[str] = set()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.hits.add(key)
        return value

    def get(self, key, default=None):
        if super().__contains__(key):
            return self[key]
        return default


class OfficialCloudOps:
    """One official backend, with operational acquisition facts only."""

    def __init__(self, case_root: str | Path):
        from debug_assistant._vendor import cloudops_snapshot

        implementation_hash = hashlib.sha256(Path(cloudops_snapshot.__file__).read_bytes()).hexdigest()
        if implementation_hash != OFFICIAL_SNAPSHOT_SHA256:
            raise ValueError("official snapshot implementation hash mismatch")
        self.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.case_root = Path(case_root).resolve(strict=True)
        for relative in ("tool_cache.json", "raw_data/logs.json"):
            path = self.case_root / relative
            if path.exists() and not path.resolve().is_relative_to(self.case_root):
                raise ValueError("snapshot input escapes case root")
        self.backend = KubernetesTools(str(self.case_root))
        self.raw_logs_available = (self.case_root / "raw_data/logs.json").is_file()
        self.backend.tool_cache = _ObservedCache(self.backend.tool_cache)
        self._lock = RLock()

    def invoke(self, method: str, arguments: dict[str, Any]):
        with self._lock:
            self.backend.tool_cache.hits.clear()
            identity = {"official_method": method, "official_source_sha256": OFFICIAL_SNAPSHOT_SHA256,
                        "official_commit": OFFICIAL_COMMIT, "local_adapter_sha256": self.adapter_sha256}
            if method == "GetRecentLogs" and not self.raw_logs_available:
                return "Recent logs unavailable: raw_data/logs.json is absent.", False, {
                    **identity,
                    "reason": "raw_logs_unavailable", "official_invoked": False,
                }
            result = getattr(self.backend, method)(**arguments)
            keys = tuple(sorted(self.backend.tool_cache.hits))
            acquired = method == "GetRecentLogs" or bool(keys)
            values = [dict.__getitem__(self.backend.tool_cache, key) for key in keys]
            negative = bool(values) and all(official_negative_result(value) for value in values)
            if method == "GetErrorLogs":
                negative = bool(values) and all(official_negative_result(value) or
                    isinstance(value, dict) and (value.get("total_errors") == 0 or
                        "patterns" in value and not value["patterns"]) for value in values)
            opaque_summary = (method == "GetErrorLogs" and any(isinstance(value, str) for value in values)
                              or method == "GetRecentLogs" and isinstance(result, str))
            return result, acquired, {
                **identity, "official_invoked": True, "acquired_cache_keys": keys,
                "semantic_negative": negative,
                "result_kind": "OPERATIONAL_ONLY" if negative or opaque_summary else "OBSERVATION",
                "query_completeness_proven": False,
            }


def official_result_success(result: Any) -> bool:
    return not (isinstance(result, str) and result.lstrip().startswith(
        ("Error:", " Error:", "Error from server", "An unexpected error")
    ))


def official_negative_result(result: Any) -> bool:
    return result in (None, "", [], {})
