from __future__ import annotations
from collections import deque
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
    def __init__(self, max_recent=8):
        self.recent=deque(maxlen=max_recent)
        self.pinned: list[Evidence]=[]
        self._seen=set()

    def add_observation(self, obs: ToolObservation):
        self.recent.append(obs)
        if not obs.ok or not obs.content.strip():
            return None
        # Retrieval results are candidate locations, not causal evidence. They must be
        # verified by a source-reading observation before entering the evidence ledger.
        if (obs.metadata or {}).get('information_source') == 'candidate_retrieval':
            return None
        key=sha1((obs.tool+'|'+obs.content).encode('utf-8','ignore')).hexdigest()[:12]
        if key in self._seen:
            return None
        self._seen.add(key)
        meta=obs.metadata
        excerpt, excerpt_truncated=_complete_line_excerpt(obs.content, 1800)
        source_start=meta.get('start_line') if obs.tool == 'read_file' else None
        source_end=meta.get('end_line') if obs.tool == 'read_file' else None
        excerpt_start=excerpt_end=None
        if obs.tool == 'read_file':
            excerpt_start, excerpt_end=_read_file_excerpt_coverage(excerpt)
        ev=Evidence(
            evidence_id=f"ev-{key}", kind=obs.tool, source=obs.tool,
            summary=obs.content[:700].replace('\n',' '), excerpt=excerpt,
            file=meta.get('path'), line_start=source_start, line_end=source_end,
            raw_observation_id=obs.observation_id,
            source_start_line=source_start, source_end_line=source_end,
            excerpt_start_line=excerpt_start, excerpt_end_line=excerpt_end,
            excerpt_truncated=excerpt_truncated,
            confidence=0.65,
        )
        self.pinned.append(ev)
        return ev

    def context(self, max_chars=24000, *, recent_observation_ids: set[str] | None=None) -> str:
        recent_observation_ids=recent_observation_ids or set()
        chunks=[]; used=0
        for e in reversed(self.pinned):
            location=''
            if e.file:
                if e.source_start_line is not None and e.source_end_line is not None:
                    location=f" {e.file}:{e.source_start_line}-{e.source_end_line}"
                else:
                    location=f" {e.file}"
            # Recent raw observations own the full content. Historical ledger keeps only concise metadata/summary.
            if e.raw_observation_id in recent_observation_ids:
                text=(f"[{e.evidence_id}] {e.kind}{location} "
                      f"(raw={e.raw_observation_id}; full bounded content is in RECENT_RAW_OBSERVATIONS)\n")
            else:
                coverage=''
                if e.excerpt_truncated:
                    if e.excerpt_start_line is not None and e.excerpt_end_line is not None:
                        coverage=f" [compressed excerpt {e.excerpt_start_line}-{e.excerpt_end_line}; source {e.source_start_line}-{e.source_end_line}]"
                    else:
                        coverage=" [compressed excerpt; exact excerpt line coverage unavailable]"
                text=f"[{e.evidence_id}] {e.kind}{location}{coverage}: {e.summary}\n"
            if used+len(text)>max_chars:
                break
            chunks.append(text); used+=len(text)
        return ''.join(reversed(chunks)) or '(no evidence yet)'
