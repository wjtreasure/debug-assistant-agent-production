"""Read-only execution projection for model-facing Context.

Semantic claim, evidence requirements and lifecycle state are owned by
RCAState. This module exposes only DAG scheduling metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationTaskContext:
    text: str = ""
    linked_evidence_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    task_count: int = 0
    ready_task_count: int = 0
    critical_open_task_count: int = 0
    duplicate_task_count: int = 0


def _dump(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {
        name: getattr(value, name)
        for name in (
            "task_id", "obligation_id", "dependencies", "retry_count",
            "replan_count", "status",
        )
        if hasattr(value, name)
    }


def project_verification_tasks(dag_state: Any, *, max_chars: int = 3_500) -> VerificationTaskContext:
    """Project task IDs/status/readiness without duplicating semantic state."""
    if dag_state is None:
        return VerificationTaskContext()
    raw_tasks = (
        dag_state.get("tasks", ()) if isinstance(dag_state, Mapping)
        else getattr(dag_state, "tasks", ())
    ) or ()
    records = [dict(_dump(item)) for item in raw_tasks]
    ready_count = sum(str(item.get("status", "")).upper() == "READY" for item in records)
    open_count = sum(str(item.get("status", "")).upper() != "COMPLETED" for item in records)
    selected = [
        item for item in records
        if str(item.get("status", "")).upper() in {"READY", "PENDING", "BLOCKED", "FAILED", "RUNNING"}
    ]
    rows = ["VERIFICATION_TASK_STATE (execution metadata only):"]
    for item in selected:
        deps = ",".join(str(dep) for dep in (item.get("dependencies") or ())) or "-"
        rows.append(
            f"- task_id={item.get('task_id', '')} obligation_id={item.get('obligation_id', '')} "
            f"status={str(item.get('status', 'PENDING')).upper()} dependencies={deps}"
        )
    text = "\n".join(rows) if selected else "VERIFICATION_TASK_STATE: (none)"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n(task metadata compacted)"
    return VerificationTaskContext(
        text=text,
        task_ids=tuple(str(item.get("task_id", "")) for item in selected),
        task_count=len(records),
        ready_task_count=ready_count,
        critical_open_task_count=open_count,
    )
