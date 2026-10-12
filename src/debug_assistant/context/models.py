from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Literal


_SNAPSHOT_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|authorization|secret)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
)


def redact_snapshot_value(value: Any) -> Any:
    """Redact credential-shaped material before it reaches trace/artifacts."""
    if isinstance(value, str):
        result = value
        for pattern in _SNAPSHOT_SECRET_PATTERNS:
            if pattern.groups:
                result = pattern.sub(lambda match: f"{match.group(1)}=<REDACTED>", result)
            else:
                result = pattern.sub("<REDACTED>", result)
        return result
    if isinstance(value, dict):
        return {str(key): redact_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_snapshot_value(item) for item in value]
    return value



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
    # Explicit active-context layer used by Dynamic Budget telemetry.
    context_level: Literal["L0", "L1", "L2", "L3"] = "L2"


@dataclass(frozen=True, slots=True)
class ContextView:
    """Detached role view produced by ContextManager after final packing."""

    role: str
    text: str = ""
    visible_observation_ids: tuple[str, ...] = ()
    visible_evidence_refs: tuple[str, ...] = ()
    body_evidence_ids: tuple[str, ...] = ()
    acquired_source_ranges: tuple[tuple[str, int, int], ...] = ()
    visible_source_ranges: tuple[tuple[str, int, int], ...] = ()
    reference_metadata_json: str = "{}"
    display_coverage_json: str = "{}"
    retained_source_ranges: tuple[tuple[str, int, int], ...] = ()
    body_coverage_json: str = "{}"

    @property
    def reference_metadata(self) -> dict[str, Any]:
        return json.loads(self.reference_metadata_json)

    def is_observation_visible(self, observation_id: str) -> bool:
        return observation_id in self.visible_observation_ids


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    """Immutable record of the exact final request visible to a model."""

    text: str
    evidence_ids: tuple[str, ...] = ()
    visible_observation_ids: tuple[str, ...] = ()
    visible_evidence_refs: tuple[str, ...] = ()
    body_evidence_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    acquired_source_ranges: tuple[tuple[str, int, int], ...] = ()
    retained_source_ranges: tuple[tuple[str, int, int], ...] = ()
    visible_source_ranges: tuple[tuple[str, int, int], ...] = ()
    token_count: int = 0
    provider: str = ""
    model: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    tool_schemas_json: str = "[]"
    transport_profile: str = ""
    request_kind: str = ""
    state_snapshot_hash: str = ""
    state_version: str = ""
    temperature: float | None = None
    max_output_tokens: int | None = None
    redacted_request_metadata_json: str = "{}"
    snapshot_hash: str = ""

    @classmethod
    def from_render(
        cls,
        text: str,
        *,
        token_count: int,
        acquired_source_ranges: tuple[tuple[str, int, int], ...] = (),
        visible_source_ranges: tuple[tuple[str, int, int], ...] = (),
        context_view: ContextView | None = None,
        provider: str = "",
        model: str = "",
        system_prompt: str = "",
        user_prompt: str = "",
        tool_schemas: tuple[dict[str, Any], ...] = (),
        transport_profile: str = "",
        request_kind: str = "",
        state_snapshot_hash: str = "",
        state_version: str = "",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        redacted_request_metadata: dict[str, Any] | None = None,
    ) -> "PromptSnapshot":
        rendered = redact_snapshot_value(str(text or ""))
        tool_schemas_json = json.dumps(
            redact_snapshot_value(tool_schemas), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        metadata_json = json.dumps(
            redact_snapshot_value(redacted_request_metadata or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        material = {
            "text": rendered,
            "evidence_ids": tuple(sorted(set(re.findall(r"\bev-[0-9a-f]+\b", rendered)))),
            "visible_observation_ids": context_view.visible_observation_ids if context_view else (),
            "visible_evidence_refs": context_view.visible_evidence_refs if context_view else (),
            "body_evidence_ids": context_view.body_evidence_ids if context_view else (),
            "obligation_ids": tuple(sorted(set(re.findall(r"\bobl-[0-9a-f]+\b", rendered)))),
            "task_ids": tuple(sorted(set(re.findall(r"\bvt-[0-9]+-[0-9]+\b", rendered)))),
            "acquired_source_ranges": tuple(acquired_source_ranges),
            "retained_source_ranges": context_view.retained_source_ranges if context_view else (),
            "visible_source_ranges": context_view.visible_source_ranges if context_view else tuple(visible_source_ranges),
            "token_count": max(0, int(token_count)),
            "provider": str(provider),
            "model": str(model),
            "system_prompt": redact_snapshot_value(str(system_prompt)),
            "user_prompt": redact_snapshot_value(str(user_prompt)),
            "tool_schemas_json": tool_schemas_json,
            "transport_profile": str(transport_profile),
            "request_kind": str(request_kind),
            "state_snapshot_hash": str(state_snapshot_hash),
            "state_version": str(state_version),
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "redacted_request_metadata_json": metadata_json,
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(snapshot_hash=digest, **material)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "visible_observation_ids": list(self.visible_observation_ids),
            "visible_evidence_refs": list(self.visible_evidence_refs),
            "body_evidence_ids": list(self.body_evidence_ids),
            "obligation_ids": list(self.obligation_ids),
            "task_ids": list(self.task_ids),
            "acquired_source_ranges": [list(item) for item in self.acquired_source_ranges],
            "retained_source_ranges": [list(item) for item in self.retained_source_ranges],
            "visible_source_ranges": [list(item) for item in self.visible_source_ranges],
            "token_count": self.token_count,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "tool_schemas": json.loads(self.tool_schemas_json),
            "transport_profile": self.transport_profile,
            "request_kind": self.request_kind,
            "state_snapshot_hash": self.state_snapshot_hash,
            "state_version": self.state_version,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "redacted_request_metadata": json.loads(
                self.redacted_request_metadata_json
            ),
            "snapshot_hash": self.snapshot_hash,
        }


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
    breakdown: dict[str, Any] = field(default_factory=dict)
    known_context_chars: int = 0
    active_item_count: int = 0
    cold_item_count: int = 0
    eviction_count: int = 0
    projection_count: int = 0
    display_coverage: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    budget_tokens: int | None = None
    used_tokens: int = 0
    token_estimator: str = "char4"
    prompt_snapshot: PromptSnapshot | None = None
    context_view: ContextView | None = None
