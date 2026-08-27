from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ContextProjection:
    projection_id: str
    observation_id: str
    path: str | None
    source_start_line: int | None
    source_end_line: int | None
    display_start_line: int | None
    display_end_line: int | None
    content: str
    priority: int
    lifecycle: Literal["active", "cold"] = "active"
    pinned: bool = False
    last_used_step: int = 0
    reason: str = ""


@dataclass(slots=True)
class ContextItem:
    context_id: str
    source_kind: str
    title: str
    compact_content: str
    full_content: str = ""
    chars: int = 0
    priority: int = 100
    created_step: int = 0
    raw_observation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    lifecycle: Literal["active", "cold"] = "active"
    pinned: bool = False
    last_used_step: int = 0


@dataclass(slots=True)
class ContextBuildResult:
    text: str
    budget_chars: int
    used_chars: int
    catalog_size: int
    working_set_size: int
    selected: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    invalid_requested_ids: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    known_context_chars: int = 0
    active_item_count: int = 0
    cold_item_count: int = 0
    eviction_count: int = 0
    projection_count: int = 0
    display_coverage: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
