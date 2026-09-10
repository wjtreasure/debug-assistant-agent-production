from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_assistant.incidents.contracts import IncidentCase


_FORBIDDEN_RUNTIME_KEYS = frozenset({
    "ground_truth", "ground_truth_fault_type", "ground_truth_component",
    "fault_type", "fault_taxonomy", "fault_object", "root_cause",
    "golden_trajectory", "milestone", "milestone_answer", "evaluation_label",
})


def _assert_runtime_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"evaluator-only field in runtime case: {path}.{key}")
            _assert_runtime_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_runtime_safe(child, f"{path}[{index}]")


class CloudOpsRuntimeCaseLoader:
    """Loads only Agent-visible case data and rejects evaluator fields recursively."""

    def load(self, runtime_dir: str | Path) -> IncidentCase:
        root = Path(runtime_dir).resolve()
        raw = json.loads((root / "case.json").read_text(encoding="utf-8"))
        cache = json.loads((root / "tool_cache.json").read_text(encoding="utf-8"))
        _assert_runtime_safe(raw)
        _assert_runtime_safe(cache)
        return IncidentCase.model_validate({**raw, "runtime_data_dir": str(root)})

