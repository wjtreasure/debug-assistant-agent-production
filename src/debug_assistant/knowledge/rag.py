from __future__ import annotations

"""Deterministic Domain Knowledge RAG adapters.

The document layer is deliberately independent from the Incident lifecycle.
It produces ``KnowledgeCandidate``/``PriorContext`` only.  Lexical recall is
always offline; dense retrieval and cross-encoder reranking are optional ports
whose unavailable state is surfaced in diagnostics instead of becoming a fake
zero score.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
import time
from typing import Any, Iterable, Protocol

from .contracts import (
    KnowledgeCandidate,
    KnowledgeProvenance,
    KnowledgeQuery,
    KnowledgeRetrievalDiagnostics,
    KnowledgeRetrievalResult,
    PriorContext,
)
from .storage import _pack_candidates
from debug_assistant.repository.search_engine import reciprocal_rank_fusion


DOMAIN_CHUNKER_VERSION = "domain-structure-v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+", re.UNICODE)
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_ENDPOINT_RE = re.compile(
    r"^(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(?P<path>/\S+)",
    re.IGNORECASE,
)
_FAILURE_ROLES = {"symptom", "cause", "diagnosis", "resolution", "impact", "mitigation"}


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(value) if len(item) > 1}


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    document_id: str
    source: str
    version: str
    content_hash: str
    ingestion_version: str = DOMAIN_CHUNKER_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "version": self.version,
            "content_hash": self.content_hash,
            "ingestion_version": self.ingestion_version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ParentChunk:
    parent_id: str
    document_id: str
    title: str
    content: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]
    provenance: KnowledgeProvenance


@dataclass(frozen=True, slots=True)
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    content: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]
    provenance: KnowledgeProvenance


@dataclass(frozen=True, slots=True)
class IngestedDomainDocument:
    manifest: DocumentManifest
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]


@dataclass(frozen=True, slots=True)
class _Section:
    title: str
    content: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]


class StructureAwareChunker:
    """Split documents by structure first, then bound oversized sections."""

    def __init__(self, *, max_child_chars: int = 1800, overlap_chars: int = 180) -> None:
        self.max_child_chars = max(32, int(max_child_chars))
        self.overlap_chars = max(0, min(int(overlap_chars), self.max_child_chars // 2))

    @staticmethod
    def _doc_type(source: str, metadata: dict[str, Any]) -> str:
        value = str(metadata.get("document_type") or metadata.get("type") or "").casefold()
        if value:
            return value
        suffix = Path(source).suffix.casefold()
        return "markdown" if suffix in {".md", ".markdown"} else "plain_text"

    def _sections(self, content: str, *, source: str, metadata: dict[str, Any]) -> list[_Section]:
        lines = content.splitlines()
        if not lines:
            return []
        doc_type = self._doc_type(source, metadata)
        headings: list[tuple[int, int, str, dict[str, Any]]] = []
        for index, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            endpoint = _ENDPOINT_RE.match(line.strip()) if doc_type in {"api", "api_docs", "api_doc"} else None
            if match:
                title = match.group("title").strip()
                level = len(match.group("marks"))
                section_meta = {"heading_level": level, "structure": "heading"}
                endpoint_match = _ENDPOINT_RE.match(title)
                if endpoint_match:
                    section_meta.update({"endpoint": endpoint_match.group("path"), "http_method": endpoint_match.group("method").upper()})
                headings.append((index, level, title, section_meta))
            elif endpoint:
                headings.append((index, 2, line.strip(), {"structure": "endpoint", "endpoint": endpoint.group("path"), "http_method": endpoint.group("method").upper()}))

        sections: list[_Section] = []
        if headings:
            for position, (start, level, title, section_meta) in enumerate(headings):
                end = headings[position + 1][0] - 1 if position + 1 < len(headings) else len(lines) - 1
                # A heading section owns its nested content.  The level is kept
                # as metadata; avoiding nested duplicate parents makes parent
                # expansion deterministic and prevents context overcounting.
                text = "\n".join(lines[start:end + 1]).strip()
                if text:
                    sections.append(_Section(title, text, start + 1, end + 1, dict(section_meta)))
            return sections

        # Plain text and headingless runbooks retain paragraph structure rather
        # than being split into an arbitrary fixed-token stream.
        starts = [0]
        for index, line in enumerate(lines):
            if not line.strip() and index + 1 < len(lines) and lines[index + 1].strip():
                starts.append(index + 1)
        for position, start in enumerate(starts):
            end = starts[position + 1] - 1 if position + 1 < len(starts) else len(lines) - 1
            text = "\n".join(lines[start:end + 1]).strip()
            if text:
                title = next((line.strip() for line in lines[start:end + 1] if line.strip()), "section")[:120]
                role = title.casefold().rstrip(":")
                meta = {"structure": "paragraph"}
                if role in _FAILURE_ROLES:
                    meta["failure_role"] = role
                sections.append(_Section(title, text, start + 1, end + 1, meta))
        return sections or [_Section("document", content, 1, len(lines), {"structure": "document"})]

    def _windows(self, section: _Section) -> list[tuple[str, int, int]]:
        if len(section.content) <= self.max_child_chars:
            return [(section.content, section.start_line, section.end_line)]
        lines = section.content.splitlines()
        windows: list[tuple[str, int, int]] = []
        begin = 0
        while begin < len(lines):
            used = 0
            end = begin
            while end < len(lines):
                addition = len(lines[end]) + (1 if end > begin else 0)
                if end > begin and used + addition > self.max_child_chars:
                    break
                used += addition
                end += 1
                if used >= self.max_child_chars:
                    break
            end = max(begin + 1, end)
            text = "\n".join(lines[begin:end])
            windows.append((text, section.start_line + begin, section.start_line + end - 1))
            if end >= len(lines):
                break
            # Overlap is line-safe and bounded.  It exists only for oversized
            # sections, preserving enough local context around a child hit.
            overlap = 0
            chars = 0
            for index in range(end - 1, begin - 1, -1):
                chars += len(lines[index]) + 1
                overlap += 1
                if chars >= self.overlap_chars:
                    break
            begin = max(begin + 1, end - overlap)
        return windows

    def chunk(
        self,
        *,
        document_id: str,
        source: str,
        content: str,
        version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IngestedDomainDocument:
        metadata = dict(metadata or {})
        manifest = DocumentManifest(
            document_id=document_id,
            source=source,
            version=version,
            content_hash=_hash(content),
            metadata={**metadata, "document_type": self._doc_type(source, metadata)},
        )
        provenance = KnowledgeProvenance(
            source=source,
            source_id=document_id,
            path=source,
            version=version or None,
            content_hash=manifest.content_hash,
            loader_version=manifest.ingestion_version,
            chunker_version=manifest.ingestion_version,
            metadata={"document_id": document_id, **metadata},
        )
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for index, section in enumerate(self._sections(content, source=source, metadata=metadata)):
            section_meta = {**metadata, **section.metadata, "document_id": document_id, "document_version": version}
            parent_id = f"parent-{document_id}-{_hash(f'{index}:{section.title}:{section.start_line}')[:12]}"
            parent = ParentChunk(
                parent_id, document_id, section.title, section.content,
                section.start_line, section.end_line, section_meta, provenance,
            )
            parents.append(parent)
            windows = self._windows(section)
            for child_index, (child_content, start, end) in enumerate(windows):
                child_id = f"child-{parent_id}-{child_index}"
                children.append(ChildChunk(
                    child_id, parent_id, document_id, child_content, start, end,
                    {**section_meta, "child_index": child_index, "child_count": len(windows)},
                    provenance,
                ))
        return IngestedDomainDocument(manifest, tuple(parents), tuple(children))


class DomainDocumentIngestor:
    """Ingest Markdown/plain/API/runbook documents into an in-memory corpus."""

    def __init__(self, chunker: StructureAwareChunker | None = None) -> None:
        self.chunker = chunker or StructureAwareChunker()

    def ingest_text(self, *, document_id: str, source: str, content: str, version: str = "", metadata: dict[str, Any] | None = None) -> IngestedDomainDocument:
        return self.chunker.chunk(document_id=document_id, source=source, content=content, version=version, metadata=metadata)

    def ingest_path(self, path: str | Path, *, document_id: str | None = None, source: str | None = None, version: str = "", metadata: dict[str, Any] | None = None) -> IngestedDomainDocument:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        return self.ingest_text(
            document_id=document_id or _hash(str(file_path))[:16],
            source=source or str(file_path), content=content, version=version,
            metadata=metadata,
        )


class Reranker(Protocol):
    name: str
    runtime_type: str

    def rerank(self, query: str, candidates: list[KnowledgeCandidate]) -> list[KnowledgeCandidate]:
        ...


class DeterministicReranker:
    """Test-only reranker; never reported as a production model."""

    name = "deterministic-fake"
    runtime_type = "DETERMINISTIC_FAKE"

    def rerank(self, query: str, candidates: list[KnowledgeCandidate]) -> list[KnowledgeCandidate]:
        query_terms = _tokens(query)
        ranked: list[KnowledgeCandidate] = []
        for candidate in candidates:
            overlap = len(query_terms & _tokens(candidate.content)) / max(1, len(query_terms))
            old = candidate.retrieval_score or 0.0
            ranked.append(candidate.model_copy(update={"rerank_score": overlap, "retrieval_score": old}))
        return sorted(ranked, key=lambda item: (-(item.rerank_score or 0.0), -(item.retrieval_score or 0.0), item.candidate_id))


class UnavailableReranker:
    name = "unavailable"
    runtime_type = "UNAVAILABLE"

    def rerank(self, query: str, candidates: list[KnowledgeCandidate]) -> list[KnowledgeCandidate]:
        raise RuntimeError("local cross-encoder reranker weights are unavailable")


class DomainKnowledgeRetriever:
    """Parent-child lexical/dense/hybrid retriever over ingested documents."""

    def __init__(
        self,
        documents: Iterable[IngestedDomainDocument] = (),
        *,
        semantic_index: Any | None = None,
        reranker: Reranker | None = None,
        max_recall: int = 30,
        max_per_source: int = 2,
    ) -> None:
        self.documents: dict[str, IngestedDomainDocument] = {item.manifest.document_id: item for item in documents}
        self.semantic_index = semantic_index
        self.reranker = reranker
        self.max_recall = max(1, int(max_recall))
        self.max_per_source = max(1, int(max_per_source))

    def add_document(self, document: IngestedDomainDocument) -> None:
        self.documents[document.manifest.document_id] = document

    def build_semantic_index(self, provider: Any, cache: Any | None = None, *, deadline: Any | None = None) -> Any:
        """Build the optional dense index through the existing SemanticIndex.

        Domain child chunks are adapted to the repository ``CodeChunk`` shape
        only at this port.  This keeps model/cache/version validation in the
        existing implementation rather than duplicating a second dense index.
        """
        from debug_assistant.repository.chunks import CHUNKER_VERSION, ChunkManifest, CodeChunk
        from debug_assistant.repository.semantic_index import SemanticIndex

        chunks = []
        for child in self.children:
            parent = self.parents[child.parent_id]
            path = child.provenance.path or child.provenance.source
            content_hash = _hash(child.content)
            embedding_text_hash = _hash(
                f"File: {path}\nKind: domain\nSymbol: {parent.title}\nCode:\n{child.content}"
            )
            chunks.append(CodeChunk(
                chunk_id=child.child_id, path=path, language="text", symbol=None,
                qualified_name=parent.title, kind="domain", start_line=child.start_line,
                end_line=child.end_line, signature=parent.title, docstring="",
                content=child.content, content_hash=content_hash,
                embedding_text_hash=embedding_text_hash, parent_symbol=child.parent_id,
            ))
        digest = _hash("\n".join(f"{item.chunk_id}:{item.embedding_text_hash}" for item in chunks))
        manifest = ChunkManifest(
            chunker_version=f"{DOMAIN_CHUNKER_VERSION}+{CHUNKER_VERSION}",
            chunks=chunks, digest=digest,
        )
        self.semantic_index = SemanticIndex(
            manifest, provider, cache, chunker_version=manifest.chunker_version,
        )
        return self.semantic_index.build(deadline=deadline)

    @property
    def children(self) -> tuple[ChildChunk, ...]:
        return tuple(child for document in self.documents.values() for child in document.children)

    @property
    def parents(self) -> dict[str, ParentChunk]:
        return {parent.parent_id: parent for document in self.documents.values() for parent in document.parents}

    def _metadata_match(self, child: ChildChunk, query: KnowledgeQuery) -> bool:
        metadata = child.metadata
        checks = {
            "service": query.service,
            "module": query.module,
            "fault_type": query.fault_type,
            "software_version": query.software_version,
            "repo_commit": query.repo_commit,
        }
        for key, value in checks.items():
            if value and metadata.get(key) not in {value, None}:
                return False
        for key, value in query.filters.items():
            if key in metadata and value is not None and metadata.get(key) != value:
                return False
        return True

    def _candidate_from_child(self, child: ChildChunk, score: float, *, content: str | None = None, rerank_score: float | None = None) -> KnowledgeCandidate:
        parent = self.parents[child.parent_id]
        metadata = child.metadata
        version = metadata.get("software_version") or metadata.get("document_version") or None
        return KnowledgeCandidate(
            candidate_id=f"rag-{child.child_id}",
            source_type="domain_rag",
            content=content or parent.content,
            parent_id=parent.parent_id,
            service=metadata.get("service"),
            module=metadata.get("module"),
            fault_type=metadata.get("fault_type"),
            software_version=version,
            repo_commit=metadata.get("repo_commit"),
            retrieval_score=max(0.0, min(1.0, float(score))),
            rerank_score=rerank_score,
            confidence=0.75,
            provenance=child.provenance,
        )

    def _lexical(self, query: KnowledgeQuery) -> list[KnowledgeCandidate]:
        terms = _tokens(query.query_text)
        ranked: list[KnowledgeCandidate] = []
        for child in self.children:
            if not self._metadata_match(child, query):
                continue
            child_terms = _tokens(child.content)
            overlap = len(terms & child_terms) / max(1, len(terms))
            phrase_bonus = 0.15 if query.query_text.casefold() in child.content.casefold() else 0.0
            score = min(1.0, overlap + phrase_bonus)
            if score <= 0 and terms:
                continue
            ranked.append(self._candidate_from_child(child, score))
        # Parent expansion makes multiple child hits refer to one final context.
        ranked.sort(key=lambda item: (-(item.retrieval_score or 0.0), item.candidate_id))
        return ranked[:self.max_recall]

    def _dense(self, query: KnowledgeQuery) -> tuple[list[KnowledgeCandidate], str]:
        if self.semantic_index is None or not bool(getattr(self.semantic_index, "available", True)):
            return [], "dense_index_unavailable"
        try:
            rows = self.semantic_index.search(query.query_text, limit=self.max_recall)
            by_child = {child.child_id: child for child in self.children}
            out: list[KnowledgeCandidate] = []
            for rank, row in enumerate(rows or [], start=1):
                child = by_child.get(str(row.get("chunk_id") or row.get("child_id") or ""))
                if child is None or not self._metadata_match(child, query):
                    continue
                out.append(self._candidate_from_child(child, float(row.get("score") or 1.0 / rank)))
            return out, ""
        except Exception as exc:
            return [], f"dense_query_failed:{type(exc).__name__}"

    @staticmethod
    def _dedup(candidates: Iterable[KnowledgeCandidate]) -> list[KnowledgeCandidate]:
        selected: list[KnowledgeCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.parent_id or _hash(candidate.content)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
        return selected

    @staticmethod
    def _source_key(candidate: KnowledgeCandidate) -> str:
        provenance = candidate.provenance
        return provenance.source_id or provenance.path or provenance.source

    def _diverse_pack(self, query: KnowledgeQuery, candidates: list[KnowledgeCandidate]) -> tuple[list[KnowledgeCandidate], dict[str, int], int, int]:
        grouped: dict[str, list[KnowledgeCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(self._source_key(candidate), []).append(candidate)
        ordered: list[KnowledgeCandidate] = []
        # Round-robin ensures one source cannot consume the entire context when
        # multiple documents are available.
        for index in range(max((len(value) for value in grouped.values()), default=0)):
            for source in sorted(grouped):
                rows = grouped[source]
                if index < len(rows) and len([item for item in ordered if self._source_key(item) == source]) < self.max_per_source:
                    ordered.append(rows[index])
        budget = query.token_budget * 4
        packed: list[KnowledgeCandidate] = []
        used = 0
        dropped = 0
        for candidate in ordered:
            remaining = budget - used
            if remaining <= 0:
                dropped += 1
                continue
            content = candidate.content
            if len(content) > remaining:
                # The caller owns a token ceiling.  Do not fill it with a
                # chopped parent section; retain complete high-value parents
                # and let the next query/turn retrieve another slice.
                dropped += 1
                continue
            packed.append(candidate.model_copy(update={"content": content}))
            used += len(content)
            if len(packed) >= query.top_k:
                dropped += len(ordered) - ordered.index(candidate) - 1
                break
        counts: dict[str, int] = {}
        for candidate in packed:
            key = self._source_key(candidate)
            counts[key] = counts.get(key, 0) + 1
        return packed, counts, used, dropped

    def retrieve(self, query: KnowledgeQuery, *, mode: str = "lexical") -> KnowledgeRetrievalResult:
        started = time.monotonic()
        mode = str(mode).casefold()
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError("mode must be one of: lexical, dense, hybrid")
        lexical = self._lexical(query)
        dense, dense_reason = (self._dense(query) if mode in {"dense", "hybrid"} else ([], ""))
        degraded = False
        reasons: list[str] = []
        if mode in {"dense", "hybrid"} and not dense:
            degraded = True
            reasons.append(dense_reason or "dense_index_unavailable")
        if mode == "lexical":
            recalled = lexical
            effective = "domain_rag_lexical"
        elif mode == "dense":
            recalled = dense or lexical
            effective = "domain_rag_dense" if dense else "domain_rag_lexical"
        else:
            if dense:
                lex_rows = [{"path": item.candidate_id, "chunk_id": item.candidate_id, "score": item.retrieval_score, "candidate": item} for item in lexical]
                dense_rows = [{"path": item.candidate_id, "chunk_id": item.candidate_id, "score": item.retrieval_score, "candidate": item} for item in dense]
                fused = reciprocal_rank_fusion([lex_rows, dense_rows], limit=self.max_recall)
                by_id = {item.candidate_id: item for item in (*lexical, *dense)}
                recalled = [by_id[item.get("path")] .model_copy(update={"retrieval_score": float(item.get("score") or 0.0)}) for item in fused if item.get("path") in by_id]
                effective = "domain_rag_hybrid"
            else:
                recalled = lexical
                effective = "domain_rag_lexical"

        deduped = self._dedup(recalled)
        reranker_status = "unavailable"
        rerank_delta: float | None = None
        if self.reranker is not None and deduped:
            try:
                before = deduped[0].retrieval_score or 0.0
                deduped = list(self.reranker.rerank(query.query_text, deduped))
                reranker_status = getattr(self.reranker, "runtime_type", getattr(self.reranker, "name", "available"))
                after = deduped[0].rerank_score if deduped else None
                rerank_delta = (after - before) if after is not None else None
            except Exception as exc:
                reranker_status = "unavailable"
                reasons.append(f"reranker_unavailable:{type(exc).__name__}")
                degraded = True
        else:
            reasons.append("reranker_unavailable")
            degraded = True
        packed, per_source, used_chars, dropped = self._diverse_pack(query, deduped)
        diagnostics = KnowledgeRetrievalDiagnostics(
            requested_sources=("domain_rag",), effective_sources=("domain_rag",),
            degraded=degraded, reason=";".join(dict.fromkeys(reasons)),
            per_source_counts={"domain_rag": len(packed)},
            latency_ms=(time.monotonic() - started) * 1000.0,
            token_estimate=max(0, (used_chars + 3) // 4),
            version_filters={key: value for key, value in {"software_version": query.software_version, "repo_commit": query.repo_commit}.items() if value},
            before_dedup=len(recalled), after_dedup=len(deduped),
            retrieval_mode=mode, reranker_status=reranker_status,
            rerank_delta=rerank_delta, per_source_count=per_source,
            diversity_applied=len({self._source_key(item) for item in deduped}) > 1,
            token_budget=query.token_budget, packed_candidates=len(packed),
            dropped_candidates=dropped,
            packing_stop_reason=("top_k_reached" if len(packed) >= query.top_k else "candidates_exhausted_or_ceiling"),
        )
        return KnowledgeRetrievalResult(candidates=tuple(packed), diagnostics=diagnostics)

    def prior_context(self, query: KnowledgeQuery, *, mode: str = "lexical", context_id: str | None = None) -> PriorContext:
        result = self.retrieve(query, mode=mode)
        return PriorContext(
            context_id=context_id or f"prior-rag-{query.incident_id}",
            candidates=result.candidates,
            source_types=("domain_rag",) if result.candidates else (),
            retrieval_query=query,
            provenance=tuple({(item.provenance.source, item.provenance.source_id, item.provenance.path): item.provenance for item in result.candidates}.values()),
            packed_chars=sum(len(item.content) for item in result.candidates),
            packed_tokens=result.diagnostics.token_estimate,
        )
