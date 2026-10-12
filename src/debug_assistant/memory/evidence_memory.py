from __future__ import annotations
from hashlib import sha1
import json
import re
from debug_assistant.models import Evidence, ToolObservation
from debug_assistant.memory.evidence_boundary import is_canonical_source_evidence

_LINE_RE = re.compile(r"^\s*(\d+)\s*\|")


def _complete_line_excerpt(content: str, max_chars: int) -> tuple[str, bool]:
    """Keep only complete lines. Never fabricate coverage from a mid-line char cut."""
    if len(content) <= max_chars:
        return content, False
    kept=[]; used=0
    for line in content.splitlines():
        addition=len(line) + (1 if kept else 0)
        if used + addition > max_chars:
            break
        kept.append(line); used += addition
    if not kept:
        # Extremely long single line: preserve bounded text, but coverage cannot be trusted.
        return content[:max_chars], True
    return "\n".join(kept), True


def _read_file_excerpt_coverage(excerpt: str) -> tuple[int | None, int | None]:
    nums=[]
    for line in excerpt.splitlines():
        m=_LINE_RE.match(line)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return None, None
    return nums[0], nums[-1]


class EvidenceMemory:
    """Task-scoped compact evidence.

    Raw bounded ToolObservations remain authoritative in trace/state. Evidence is a compact,
    provenance-backed historical representation and must never overstate excerpt coverage.
    """
    def __init__(self):
        self.pinned: list[Evidence]=[]
        self._seen=set()
        self._evidence_by_fingerprint: dict[str, Evidence] = {}
        self._evidence_by_observation_id: dict[str, Evidence] = {}

    def publish_staged(self, staged: "EvidenceMemory") -> None:
        """Publish an already validated, short-lived transaction image.

        Preserve the canonical list identity used by detached Runtime readers.
        Semantic validation must finish before calling this write boundary.
        """
        self.pinned[:] = staged.pinned
        self._seen = staged._seen
        self._evidence_by_fingerprint = staged._evidence_by_fingerprint
        self._evidence_by_observation_id = staged._evidence_by_observation_id

    def evidence_for_observation(self, observation_id: str) -> Evidence | None:
        """Return the canonical Evidence representing an immutable Observation.

        Observation IDs are internal provenance.  Model-facing context uses the
        returned Evidence ID, including when two observations contain the same fact.
        """
        return self._evidence_by_observation_id.get(observation_id)

    def get(self, evidence_id: str | None) -> Evidence | None:
        return next((item for item in self.pinned if item.evidence_id == evidence_id), None)

    def projection(self, *, facts=(), evidence_ids=(), max_chars: int = 6000) -> list[dict]:
        """Bounded admitted-Evidence references, not historical raw output.

        Relationship meaning belongs to typed projected facts. Body/provenance
        stays here; this detached projection is rebuilt for each immutable turn.
        """
        facts = tuple(facts)
        allowed = set(evidence_ids)
        rows = []
        for evidence in reversed(self.pinned):
            if evidence.evidence_id not in allowed:
                continue
            relations = [fact for fact in facts if fact.evidence_id == evidence.evidence_id]
            row = {
                "evidence_ref": evidence.evidence_id,
                "claim": (relations[-1].fact if relations else evidence.summary)[:500],
                "source": evidence.source,
                "observation_refs": [evidence.raw_observation_id],
                "provenance": {key: value for key, value in evidence.provenance.items()
                               if key in {"context_kind", "information_source", "path"}},
                "confidence": evidence.confidence,
                "qualification": "ADMITTED",
                "canonical_source": is_canonical_source_evidence(None, evidence, evidence_memory=self),
                "relations": [{"hypothesis_id": fact.hypothesis_id,
                               "hypothesis_scope": fact.hypothesis_fingerprint,
                               "obligation_id": fact.obligation_id,
                               "relation": fact.relation,
                               "hypothesis_relation": fact.hypothesis_relation,
                               "covered_fact_slots": list(fact.covered_fact_slots),
                               "reason": fact.reason,
                               "rationale": fact.rationale,
                               "source_spans": list(fact.source_spans)}
                              for fact in relations[-8:]],
            }
            if len(json.dumps(rows + [row], ensure_ascii=False, separators=(",", ":"))) <= max_chars:
                rows.append(row)
        return rows

    def add_observation(self, obs: ToolObservation, *, evidence_id: str | None = None,
                        kind: str | None = None, source: str | None = None,
                        summary: str | None = None, excerpt: str | None = None,
                        target: str | None = None, tags: list[str] | None = None,
                        enforce_admission: bool = True,
                        deduplicate: bool = True):
        if not obs.ok or not obs.content.strip():
            return None
        # A captured query can be available while returning an empty collection
        # (for example, no error logs). Preserve that raw Observation, but do not
        # promote the absence of records into a citable causal fact.
        if enforce_admission and ((obs.metadata or {}).get('semantic_negative') is True
                or (obs.metadata or {}).get('result_kind') == 'OPERATIONAL_ONLY'):
            return None
        # Retrieval results are candidate locations, not causal evidence. They must be
        # verified by a source-reading observation before entering the evidence ledger.
        information_source = (obs.metadata or {}).get('information_source')
        if enforce_admission and information_source in {
            'candidate_retrieval', 'prior', 'knowledge_candidate',
            'incident_memory', 'domain_rag', 'static_kg',
        }:
            return None
        # Even if a custom symbol tool labels its bounded preview as source_read,
        # symbol lookup is still discovery. Only read_file is the
        # canonical source-verification observation.
        if enforce_admission and obs.tool == 'symbol_search':
            return None
        meta = obs.metadata or {}
        key=sha1((obs.tool+'|'+obs.content).encode('utf-8','ignore')).hexdigest()[:12]
        if deduplicate and key in self._seen:
            existing = self._evidence_by_fingerprint.get(key)
            if existing is not None:
                self._evidence_by_observation_id[obs.observation_id] = existing
            return None
        self._seen.add(key)
        if excerpt is None:
            evidence_excerpt, excerpt_truncated=_complete_line_excerpt(obs.content, 1800)
        else:
            evidence_excerpt, excerpt_truncated=excerpt, False
        source_read = obs.tool == 'read_file'
        source_start=meta.get('start_line') if source_read else None
        source_end=meta.get('end_line') if source_read else None
        excerpt_start=excerpt_end=None
        if obs.tool == 'read_file':
            excerpt_start, excerpt_end=_read_file_excerpt_coverage(evidence_excerpt)
        provenance = dict(meta.get('provenance') or {})
        if meta.get('information_source'):
            provenance.setdefault('information_source', meta.get('information_source'))
        if meta.get('context_kind'):
            provenance.setdefault('context_kind', meta.get('context_kind'))
        if meta.get('path'):
            provenance.setdefault('path', meta.get('path'))
        ev=Evidence(
            evidence_id=evidence_id or f"ev-{key}", kind=kind or obs.tool, source=source or obs.tool,
            summary=summary or obs.content[:700].replace('\n',' '), target=target,
            excerpt=evidence_excerpt,
            file=meta.get('path'),
            raw_observation_id=obs.observation_id,
            source_start_line=source_start, source_end_line=source_end,
            excerpt_start_line=excerpt_start, excerpt_end_line=excerpt_end,
            excerpt_truncated=excerpt_truncated,
            confidence=0.65, tags=list(tags or []),
            provenance=provenance,
        )
        self.pinned.append(ev)
        self._evidence_by_fingerprint[key] = ev
        self._evidence_by_observation_id[obs.observation_id] = ev
        return ev

    def source_evidence_ids(self) -> tuple[str, ...]:
        """Return only canonical bounded ``read_file`` CODE Evidence IDs."""
        return tuple(
            item.evidence_id for item in self.pinned
            if is_canonical_source_evidence(None, item, evidence_memory=self)
        )
