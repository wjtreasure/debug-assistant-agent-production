from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    IncidentMemoryRecord,
    KnowledgeCandidate,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
)


class MemoryVerificationError(ValueError):
    """Raised when a historical memory state transition is not explicit/legal."""


class KnowledgeStore(Protocol):
    """Storage and retrieval port consumed by future Runtime integration."""

    def put_memory(self, record: IncidentMemoryRecord) -> IncidentMemoryRecord:
        ...

    def get_memory(self, incident_id: str) -> IncidentMemoryRecord | None:
        ...

    def list_memory(
        self,
        *,
        verification_status: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[IncidentMemoryRecord, ...]:
        ...

    def verify_memory(self, incident_id: str) -> IncidentMemoryRecord:
        ...

    def put_candidate(self, candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        ...

    def retrieve(self, query: KnowledgeQuery) -> KnowledgeRetrievalResult:
        ...
