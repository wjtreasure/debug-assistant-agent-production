from __future__ import annotations
from hashlib import sha1
import re
from debug_assistant.models import Evidence, ToolObservation

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

    def evidence_for_observation(self, observation_id: str) -> Evidence | None:
        """Return the canonical Evidence representing an immutable Observation.

        Observation IDs are internal provenance.  Model-facing context uses the
        returned Evidence ID, including when two observations contain the same fact.
        """
        return self._evidence_by_observation_id.get(observation_id)

    def add_observation(self, obs: ToolObservation, *, evidence_id: str | None = None,
                        kind: str | None = None, source: str | None = None,
                        summary: str | None = None, excerpt: str | None = None,
                        target: str | None = None, tags: list[str] | None = None):
        if not obs.ok or not obs.content.strip():
            return None
        # A captured query can be available while returning an empty collection
        # (for example, no error logs). Preserve that raw Observation, but do not
        # promote the absence of records into a citable causal fact.
        if (obs.metadata or {}).get('semantic_negative') is True:
            return None
        # Retrieval results are candidate locations, not causal evidence. They must be
        # verified by a source-reading observation before entering the evidence ledger.
        if (obs.metadata or {}).get('information_source') == 'candidate_retrieval':
            return None
        # Even if a legacy/custom symbol tool labels its bounded preview as
        # source_read, symbol lookup is still discovery.  Only read_file is the
        # canonical source-verification observation.
        if obs.tool == 'symbol_search':
            return None
        meta = obs.metadata or {}
        key=sha1((obs.tool+'|'+obs.content).encode('utf-8','ignore')).hexdigest()[:12]
        if key in self._seen:
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
        ev=Evidence(
            evidence_id=evidence_id or f"ev-{key}", kind=kind or obs.tool, source=source or obs.tool,
            summary=summary or obs.content[:700].replace('\n',' '), target=target,
            excerpt=evidence_excerpt,
            file=meta.get('path'), line_start=source_start, line_end=source_end,
            raw_observation_id=obs.observation_id,
            source_start_line=source_start, source_end_line=source_end,
            excerpt_start_line=excerpt_start, excerpt_end_line=excerpt_end,
            excerpt_truncated=excerpt_truncated,
            confidence=0.65, tags=list(tags or []),
        )
        self.pinned.append(ev)
        self._evidence_by_fingerprint[key] = ev
        self._evidence_by_observation_id[obs.observation_id] = ev
        return ev
