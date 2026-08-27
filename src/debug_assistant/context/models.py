from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

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
