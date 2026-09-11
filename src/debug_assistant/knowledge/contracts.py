from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


KnowledgeSource = Literal["domain_rag", "incident_memory", "static_kg"]
VerificationStatus = Literal["TEMPORARY", "VERIFIED"]


class KnowledgeProvenance(BaseModel):
    """Structured origin metadata for a prior candidate.

    A free-form provenance string is deliberately not accepted.  ``source_id``
    is suitable for logical assets such as an incident or document, while
    ``path`` identifies a file-backed source.  Version/hash fields are optional
    because not every deterministic fixture has repository metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    source_id: str | None = None
    path: str | None = None
    version: str | None = None
    repo_commit: str | None = None
    timestamp: str | None = None
    content_hash: str | None = None
    loader_version: str | None = None
    chunker_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_source_locator(self) -> "KnowledgeProvenance":
        if not self.source_id and not self.path:
            raise ValueError("provenance requires source_id or path")
        return self

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Validate ISO values when supplied, while keeping the serialized
        # contract string-based and compatible with existing JSON artifacts.
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("provenance timestamp must be ISO-8601") from exc
        return value


class KnowledgeCandidate(BaseModel):
    """A retrievable prior item, never a current incident Evidence item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    source_type: KnowledgeSource
    content: str = Field(min_length=1)
    parent_id: str | None = None
    service: str | None = None
    module: str | None = None
    fault_type: str | None = None
    software_version: str | None = None
    repo_commit: str | None = None
    timestamp: str | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: KnowledgeProvenance

    @field_validator("candidate_id")
    @classmethod
    def _not_an_evidence_id(cls, value: str) -> str:
        if value.startswith("ev-"):
            raise ValueError("KnowledgeCandidate cannot use the ev-* namespace")
        return value

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("candidate timestamp must be ISO-8601") from exc
        return value


class KnowledgeQuery(BaseModel):
    """Bounded query understood by Knowledge adapters, not by a database."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    service: str | None = None
    module: str | None = None
    fault_type: str | None = None
    software_version: str | None = None
    repo_commit: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    requested_sources: tuple[KnowledgeSource, ...] = ()
    top_k: int = Field(default=5, ge=1, le=100)
    token_budget: int = Field(default=1200, ge=1, le=100_000)


class KnowledgeRetrievalDiagnostics(BaseModel):
    """Operational metadata for a retrieval attempt.

    ``degraded`` is explicit: unavailable sources are not represented as a
    zero-quality score or silently omitted from the accounting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_sources: tuple[KnowledgeSource, ...] = ()
    effective_sources: tuple[KnowledgeSource, ...] = ()
    degraded: bool = False
    reason: str = ""
    per_source_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    token_estimate: int = Field(default=0, ge=0)
    version_filters: dict[str, str] = Field(default_factory=dict)
    before_dedup: int | None = Field(default=None, ge=0)
    after_dedup: int | None = Field(default=None, ge=0)
    retrieval_mode: str = "lexical"
    reranker_status: str = "unavailable"
    rerank_delta: float | None = None
    per_source_count: dict[str, int] = Field(default_factory=dict)
    diversity_applied: bool = False
    token_budget: int | None = Field(default=None, ge=1)
    packed_candidates: int | None = Field(default=None, ge=0)
    dropped_candidates: int | None = Field(default=None, ge=0)


class KnowledgeRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[KnowledgeCandidate, ...] = ()
    diagnostics: KnowledgeRetrievalDiagnostics

    @field_validator("candidates")
    @classmethod
    def _reject_evidence_namespace(cls, values: tuple[KnowledgeCandidate, ...]) -> tuple[KnowledgeCandidate, ...]:
        if any(item.candidate_id.startswith("ev-") for item in values):
            raise ValueError("Knowledge retrieval cannot return ev-* identifiers")
        return values


class PriorContext(BaseModel):
    """A bounded, explicitly prior-only context projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: str = Field(min_length=1)
    candidates: tuple[KnowledgeCandidate, ...] = ()
    source_types: tuple[KnowledgeSource, ...] = ()
    retrieval_query: KnowledgeQuery
    provenance: tuple[KnowledgeProvenance, ...] = ()
    prior_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    packed_chars: int = Field(default=0, ge=0)
    packed_tokens: int = Field(default=0, ge=0)

    @field_validator("context_id")
    @classmethod
    def _not_an_evidence_context(cls, value: str) -> str:
        if value.startswith("ev-"):
            raise ValueError("PriorContext cannot use the ev-* namespace")
        return value

    @model_validator(mode="after")
    def _validate_prior_projection(self) -> "PriorContext":
        if any(candidate.candidate_id.startswith("ev-") for candidate in self.candidates):
            raise ValueError("PriorContext cannot contain ev-* candidates")
        candidate_sources = tuple(dict.fromkeys(item.source_type for item in self.candidates))
        if self.source_types and not set(candidate_sources).issubset(set(self.source_types)):
            raise ValueError("source_types must cover every candidate source")
        if not self.source_types and candidate_sources:
            object.__setattr__(self, "source_types", candidate_sources)
        if not self.provenance and self.candidates:
            unique = {(item.provenance.source, item.provenance.source_id, item.provenance.path): item.provenance for item in self.candidates}
            object.__setattr__(self, "provenance", tuple(unique.values()))
        if self.packed_chars == 0:
            object.__setattr__(self, "packed_chars", sum(len(item.content) for item in self.candidates))
        if self.packed_tokens == 0:
            object.__setattr__(self, "packed_tokens", max(0, (self.packed_chars + 3) // 4))
        return self


class IncidentMemoryRecord(BaseModel):
    """Historical diagnosis record with an explicit TEMPORARY/VERIFIED state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1)
    symptom: str = ""
    summary: str = ""
    service: str | None = None
    module: str | None = None
    fault_type: str | None = None
    root_cause: str = ""
    key_evidence_ids: tuple[str, ...] = ()
    key_evidence_summary: tuple[str, ...] = ()
    skills_used: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    solution: str = ""
    repo: str | None = None
    repo_commit: str | None = None
    software_version: str | None = None
    timestamp: str | None = None
    verification_status: VerificationStatus = "TEMPORARY"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: KnowledgeProvenance

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("memory timestamp must be ISO-8601") from exc
        return value
