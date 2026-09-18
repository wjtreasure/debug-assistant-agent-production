"""Compact projection of graph task state into model-facing context."""

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
            "task_id", "claim", "evidence_requirement", "dependencies", "critical",
            "status", "supporting_evidence_ids", "contradicting_evidence_ids",
        )
        if hasattr(value, name)
    }


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def project_verification_tasks(
    dag_state: Any,
    *,
    max_chars: int = 3_500,
) -> VerificationTaskContext:
    """Project only READY/critical/invalidated task metadata.

    Full task state stays in the orchestration state. The planner receives the
    minimum routing context plus canonical Evidence pointers; Evidence bodies
    remain under ContextManager packing and provenance rules.
    """

    if dag_state is None:
        return VerificationTaskContext()
    raw_tasks = (
        dag_state.get("tasks", ())
        if isinstance(dag_state, Mapping)
        else getattr(dag_state, "tasks", ())
    ) or ()
    raw_obligations = (
        dag_state.get("obligations", ())
        if isinstance(dag_state, Mapping)
        else getattr(dag_state, "obligations", ())
    ) or ()
    obligations = {}
    for item in raw_obligations:
        record = _dump(item)
        if record.get("id"):
            obligations[str(record["id"])] = record
    records = []
    for item in raw_tasks:
        record = dict(_dump(item))
        obligation = obligations.get(str(record.get("obligation_id") or ""))
        if obligation:
            # The task is execution metadata; semantic prose is projected from
            # the single canonical obligation registry.
            record.update({
                "claim": obligation.get("claim", ""),
                "evidence_requirement": obligation.get("evidence_requirement", ""),
                "critical": obligation.get("critical", True),
            })
        records.append(record)
    selected = []
    linked: list[str] = []
    selected_task_ids: list[str] = []
    ready_count = 0
    critical_open_count = 0
    candidates: list[tuple[Mapping[str, Any], str, bool, tuple[str, ...]]] = []
    for task in records:
        status = str(task.get("status") or "PENDING").upper()
        critical = bool(task.get("critical", True))
        if status == "READY":
            ready_count += 1
        if critical and status != "SATISFIED":
            critical_open_count += 1
        # Keep invalidated branches visible even when optional; a contradiction
        # must not disappear from the next planner prompt.
        if not (status == "READY" or critical or status in {"CONTRADICTED", "BLOCKED"}):
            continue
        support = tuple(str(item) for item in (task.get("supporting_evidence_ids") or ()))
        contradiction = tuple(
            str(item) for item in (task.get("contradicting_evidence_ids") or ())
        )
        pointers = tuple(
            dict.fromkeys(item for item in (*support, *contradiction) if item.startswith("ev-"))
        )
        linked.extend(item for item in pointers if item not in linked)
        # Satisfied tasks still contribute their Evidence pointers, but their
        # question is no longer active planner context.
        if status == "SATISFIED":
            continue
        if not (status == "READY" or critical or status in {"CONTRADICTED", "BLOCKED"}):
            continue
        candidates.append((task, status, critical, pointers))

    groups: dict[tuple[str, str], list[tuple[Mapping[str, Any], str, bool, tuple[str, ...]]]] = {}
    for item in candidates:
        task, _status, _critical, _pointers = item
        key = (_normalize(task.get("claim")), _normalize(task.get("evidence_requirement")))
        groups.setdefault(key, []).append(item)

    def representative(item):
        task, status, critical, _pointers = item
        status_rank = {"READY": 0, "BLOCKED": 1, "CONTRADICTED": 1, "PENDING": 2}.get(status, 3)
        return (status_rank, 0 if critical else 1, str(task.get("task_id") or ""))

    for group in groups.values():
        representative_item = min(group, key=representative)
        task, status, critical, _pointers = representative_item
        merged_pointers = tuple(dict.fromkeys(
            pointer
            for _task, _status, _critical, pointers in group
            for pointer in pointers
        ))
        selected.append((task, status, critical, merged_pointers))
        selected_task_ids.append(str(task.get("task_id") or ""))

    duplicate_task_count = len(candidates) - len(selected)

    if not selected:
        text = "VERIFICATION_TASK_STATE: (none)"
        if linked:
            text += "\nCONFIRMED_EVIDENCE_POINTERS: " + ",".join(linked)
        return VerificationTaskContext(
            text=text,
            linked_evidence_ids=tuple(linked),
            task_ids=tuple(selected_task_ids),
            task_count=len(records),
            ready_task_count=ready_count,
            critical_open_task_count=critical_open_count,
            duplicate_task_count=duplicate_task_count,
        )

    rows = [
        "VERIFICATION_TASK_STATE (runtime-owned routing metadata; Evidence pointers only):"
    ]
    selected_pointers: set[str] = set()
    for task, status, critical, pointers in selected:
        selected_pointers.update(pointers)
        dependencies = ",".join(str(item) for item in (task.get("dependencies") or ())) or "-"
        claim = " ".join(str(task.get("claim") or "").split())[:360]
        requirement = " ".join(str(task.get("evidence_requirement") or "").split())[:360]
        rows.extend(
            [
                f"- task_id={task.get('task_id', '')} status={status} critical={critical} dependencies={dependencies}",
                f"  CLAIM: {claim}",
                f"  EVIDENCE_REQUIREMENT: {requirement}",
                f"  EVIDENCE_POINTERS: {','.join(pointers) or '(none; unresolved)'}",
            ]
        )
    confirmed_pointers = tuple(
        pointer for pointer in linked if pointer not in selected_pointers
    )
    if confirmed_pointers:
        rows.append("CONFIRMED_EVIDENCE_POINTERS: " + ",".join(confirmed_pointers))
    text = "\n".join(rows)
    if len(text) > max_chars:
        # Do not silently omit a task or its pointer. Claim/requirement prose
        # is compacted while every selected task remains represented.
        compact = [rows[0]]
        for task, status, critical, pointers in selected:
            compact.append(
                f"- task_id={task.get('task_id', '')} status={status} critical={critical} "
                f"EVIDENCE_POINTERS={','.join(pointers) or '(none; unresolved)'}"
            )
        if confirmed_pointers:
            compact.append("CONFIRMED_EVIDENCE_POINTERS: " + ",".join(confirmed_pointers))
        text = "\n".join(compact) + "\n(task claim/requirement prose compacted)"
    return VerificationTaskContext(
        text=text,
        linked_evidence_ids=tuple(linked),
        task_ids=tuple(selected_task_ids),
        task_count=len(records),
        ready_task_count=ready_count,
        critical_open_task_count=critical_open_count,
        duplicate_task_count=duplicate_task_count,
    )
