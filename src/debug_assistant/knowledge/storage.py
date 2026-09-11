from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    IncidentMemoryRecord,
    KnowledgeCandidate,
    KnowledgeQuery,
    KnowledgeRetrievalDiagnostics,
    KnowledgeRetrievalResult,
    KnowledgeSource,
    PriorContext,
)
from .ports import MemoryVerificationError


_SOURCES: tuple[KnowledgeSource, ...] = ("domain_rag", "incident_memory", "static_kg")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+", re.UNICODE)


def _matches_filters(record: IncidentMemoryRecord, filters: dict[str, Any] | None) -> bool:
    for key, expected in (filters or {}).items():
        if not hasattr(record, key):
            return False
        actual = getattr(record, key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif expected is not None and actual != expected:
            return False
    return True


def _record_candidate(record: IncidentMemoryRecord, query: KnowledgeQuery) -> KnowledgeCandidate:
    parts = [record.summary, record.symptom, record.root_cause, record.solution]
    parts.extend(record.key_evidence_summary)
    content = "\n".join(part for part in parts if part).strip() or record.incident_id
    query_terms = {token.casefold() for token in _TOKEN_RE.findall(query.query_text) if len(token) > 1}
    content_terms = {token.casefold() for token in _TOKEN_RE.findall(content) if len(token) > 1}
    overlap = len(query_terms & content_terms)
    score = overlap / max(1, len(query_terms))
    if query.service and query.service == record.service:
        score += 0.25
    if query.module and query.module == record.module:
        score += 0.20
    if query.software_version and query.software_version == record.software_version:
        score += 0.15
    return KnowledgeCandidate(
        candidate_id=f"memory-{record.incident_id}",
        source_type="incident_memory",
        content=content,
        service=record.service,
        module=record.module,
        fault_type=record.fault_type,
        software_version=record.software_version,
        repo_commit=record.repo_commit,
        timestamp=record.timestamp,
        retrieval_score=min(1.0, score),
        confidence=record.confidence,
        provenance=record.provenance,
    )


def _candidate_matches_query(candidate: KnowledgeCandidate, query: KnowledgeQuery) -> bool:
    if query.service and candidate.service != query.service:
        return False
    if query.module and candidate.module != query.module:
        return False
    if query.fault_type and candidate.fault_type != query.fault_type:
        return False
    if query.software_version and candidate.software_version != query.software_version:
        return False
    if query.repo_commit and candidate.repo_commit != query.repo_commit:
        return False
    return _matches_filters(candidate, query.filters)


def _pack_candidates(query: KnowledgeQuery, candidates: Iterable[KnowledgeCandidate]) -> tuple[KnowledgeCandidate, ...]:
    budget = query.token_budget * 4
    selected: list[KnowledgeCandidate] = []
    used = 0
    for candidate in candidates:
        size = len(candidate.content)
        if selected and used + size > budget:
            continue
        selected.append(candidate)
        used += min(size, budget)
        if len(selected) >= query.top_k:
            break
    return tuple(selected)


class _RetrievalMixin:
    """Shared deterministic retrieval semantics for both storage adapters."""

    def _memory_records(self) -> tuple[IncidentMemoryRecord, ...]:
        raise NotImplementedError

    def _stored_candidates(self) -> tuple[KnowledgeCandidate, ...]:
        raise NotImplementedError

    def retrieve(self, query: KnowledgeQuery) -> KnowledgeRetrievalResult:
        started = time.monotonic()
        requested = query.requested_sources or _SOURCES
        effective: list[KnowledgeSource] = []
        raw: list[KnowledgeCandidate] = []
        counts: dict[str, int] = {}
        unavailable = []

        if "incident_memory" in requested:
            records = [
                record for record in self._memory_records()
                if _matches_filters(record, query.filters)
                and (not query.service or record.service == query.service)
                and (not query.module or record.module == query.module)
                and (not query.fault_type or record.fault_type == query.fault_type)
                and (not query.software_version or record.software_version == query.software_version)
                and (not query.repo_commit or record.repo_commit == query.repo_commit)
            ]
            raw.extend(_record_candidate(record, query) for record in records)
            counts["incident_memory"] = len(records)
            effective.append("incident_memory")
        for source in requested:
            if source == "incident_memory":
                continue
            stored = [
                candidate for candidate in self._stored_candidates()
                if candidate.source_type == source and _candidate_matches_query(candidate, query)
            ]
            raw.extend(stored)
            counts[source] = len(stored)
            if stored:
                effective.append(source)
            else:
                unavailable.append(source)

        raw.sort(key=lambda item: (-(item.retrieval_score or 0.0), item.candidate_id))
        packed = _pack_candidates(query, raw)
        reason = ""
        degraded = bool(unavailable)
        if unavailable:
            reason = "requested knowledge source unavailable: " + ", ".join(unavailable)
        diagnostics = KnowledgeRetrievalDiagnostics(
            requested_sources=tuple(requested),
            effective_sources=tuple(dict.fromkeys(effective)),
            degraded=degraded,
            reason=reason,
            per_source_counts=counts,
            latency_ms=(time.monotonic() - started) * 1000.0,
            token_estimate=sum((len(item.content) + 3) // 4 for item in packed),
            version_filters={
                key: value for key, value in {
                    "software_version": query.software_version,
                    "repo_commit": query.repo_commit,
                }.items() if value
            },
            before_dedup=len(raw),
            after_dedup=len(packed),
        )
        return KnowledgeRetrievalResult(candidates=packed, diagnostics=diagnostics)

    @staticmethod
    def prior_context(query: KnowledgeQuery, result: KnowledgeRetrievalResult, *, context_id: str | None = None) -> PriorContext:
        candidates = tuple(result.candidates)
        return PriorContext(
            context_id=context_id or f"prior-{query.incident_id}",
            candidates=candidates,
            source_types=tuple(dict.fromkeys(item.source_type for item in candidates)),
            retrieval_query=query,
            provenance=tuple({
                (item.provenance.source, item.provenance.source_id, item.provenance.path): item.provenance
                for item in candidates
            }.values()),
            packed_chars=sum(len(item.content) for item in candidates),
            packed_tokens=sum((len(item.content) + 3) // 4 for item in candidates),
        )


class InMemoryKnowledgeStore(_RetrievalMixin):
    """Deterministic task/test adapter with explicit memory state transitions."""

    def __init__(self) -> None:
        self._memory: dict[str, IncidentMemoryRecord] = {}
        self._candidates: dict[str, KnowledgeCandidate] = {}

    def put_memory(self, record: IncidentMemoryRecord) -> IncidentMemoryRecord:
        existing = self._memory.get(record.incident_id)
        if existing is not None and existing.verification_status != record.verification_status:
            raise MemoryVerificationError(
                "verification_status changes require verify_memory(), not put_memory()"
            )
        self._memory[record.incident_id] = record
        return record

    def get_memory(self, incident_id: str) -> IncidentMemoryRecord | None:
        return self._memory.get(incident_id)

    def list_memory(self, *, verification_status: str | None = None, filters: dict[str, Any] | None = None) -> tuple[IncidentMemoryRecord, ...]:
        rows = [record for record in self._memory.values() if _matches_filters(record, filters)]
        if verification_status is not None:
            rows = [record for record in rows if record.verification_status == verification_status]
        return tuple(sorted(rows, key=lambda item: item.incident_id))

    def verify_memory(self, incident_id: str) -> IncidentMemoryRecord:
        record = self._memory.get(incident_id)
        if record is None:
            raise MemoryVerificationError(f"unknown incident memory: {incident_id}")
        if record.verification_status != "TEMPORARY":
            raise MemoryVerificationError(
                f"illegal memory transition {record.verification_status}->VERIFIED"
            )
        verified = record.model_copy(update={"verification_status": "VERIFIED"})
        self._memory[incident_id] = verified
        return verified

    def put_candidate(self, candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        if candidate.candidate_id.startswith("ev-"):
            raise ValueError("Knowledge candidate cannot be stored in ev-* namespace")
        self._candidates[candidate.candidate_id] = candidate
        return candidate

    def _memory_records(self) -> tuple[IncidentMemoryRecord, ...]:
        return tuple(self._memory.values())

    def _stored_candidates(self) -> tuple[KnowledgeCandidate, ...]:
        return tuple(self._candidates.values())


class SQLiteKnowledgeStore(_RetrievalMixin):
    """Small stdlib SQLite adapter; no backend-specific contract leaks out."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS incident_memory (
                incident_id TEXT PRIMARY KEY,
                verification_status TEXT NOT NULL,
                service TEXT,
                module TEXT,
                fault_type TEXT,
                software_version TEXT,
                repo_commit TEXT,
                timestamp TEXT,
                payload TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_candidates (
                candidate_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def put_memory(self, record: IncidentMemoryRecord) -> IncidentMemoryRecord:
        existing = self.get_memory(record.incident_id)
        if existing is not None and existing.verification_status != record.verification_status:
            raise MemoryVerificationError(
                "verification_status changes require verify_memory(), not put_memory()"
            )
        self._conn.execute(
            """INSERT INTO incident_memory
               (incident_id,verification_status,service,module,fault_type,software_version,repo_commit,timestamp,payload)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(incident_id) DO UPDATE SET
                 verification_status=excluded.verification_status, service=excluded.service,
                 module=excluded.module, fault_type=excluded.fault_type,
                 software_version=excluded.software_version, repo_commit=excluded.repo_commit,
                 timestamp=excluded.timestamp, payload=excluded.payload""",
            (
                record.incident_id, record.verification_status, record.service,
                record.module, record.fault_type, record.software_version,
                record.repo_commit, record.timestamp,
                json.dumps(record.model_dump(mode="json"), sort_keys=True),
            ),
        )
        self._conn.commit()
        return record

    def get_memory(self, incident_id: str) -> IncidentMemoryRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM incident_memory WHERE incident_id=?", (incident_id,)
        ).fetchone()
        return IncidentMemoryRecord.model_validate(json.loads(row[0])) if row else None

    def list_memory(self, *, verification_status: str | None = None, filters: dict[str, Any] | None = None) -> tuple[IncidentMemoryRecord, ...]:
        clauses: list[str] = []
        args: list[Any] = []
        if verification_status is not None:
            clauses.append("verification_status=?")
            args.append(verification_status)
        for key in ("service", "module", "fault_type", "software_version", "repo_commit"):
            if filters and key in filters:
                clauses.append(f"{key}=?")
                args.append(filters[key])
        sql = "SELECT payload FROM incident_memory"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY incident_id"
        rows = tuple(IncidentMemoryRecord.model_validate(json.loads(row[0])) for row in self._conn.execute(sql, args))
        return tuple(record for record in rows if _matches_filters(record, filters))

    def verify_memory(self, incident_id: str) -> IncidentMemoryRecord:
        record = self.get_memory(incident_id)
        if record is None:
            raise MemoryVerificationError(f"unknown incident memory: {incident_id}")
        if record.verification_status != "TEMPORARY":
            raise MemoryVerificationError(
                f"illegal memory transition {record.verification_status}->VERIFIED"
            )
        verified = record.model_copy(update={"verification_status": "VERIFIED"})
        self._conn.execute(
            "UPDATE incident_memory SET verification_status=?, payload=? WHERE incident_id=?",
            (
                verified.verification_status,
                json.dumps(verified.model_dump(mode="json"), sort_keys=True),
                incident_id,
            ),
        )
        self._conn.commit()
        return verified

    def put_candidate(self, candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        if candidate.candidate_id.startswith("ev-"):
            raise ValueError("Knowledge candidate cannot be stored in ev-* namespace")
        self._conn.execute(
            "INSERT INTO knowledge_candidates(candidate_id,source_type,payload) VALUES (?,?,?) "
            "ON CONFLICT(candidate_id) DO UPDATE SET source_type=excluded.source_type,payload=excluded.payload",
            (candidate.candidate_id, candidate.source_type, json.dumps(candidate.model_dump(mode="json"), sort_keys=True)),
        )
        self._conn.commit()
        return candidate

    def _memory_records(self) -> tuple[IncidentMemoryRecord, ...]:
        return self.list_memory()

    def _stored_candidates(self) -> tuple[KnowledgeCandidate, ...]:
        return tuple(
            KnowledgeCandidate.model_validate(json.loads(row[0]))
            for row in self._conn.execute("SELECT payload FROM knowledge_candidates ORDER BY candidate_id")
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteKnowledgeStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
