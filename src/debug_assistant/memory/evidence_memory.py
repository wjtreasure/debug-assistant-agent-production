from __future__ import annotations
from collections import deque
from dataclasses import asdict
from hashlib import sha1
from debug_assistant.models import Evidence, ToolObservation, AgentState

class EvidenceMemory:
    """Task-scoped memory: raw observations stay in trace; durable context stores compact, provenance-backed evidence."""
    def __init__(self, max_recent=8):
        self.recent=deque(maxlen=max_recent)
        self.pinned: list[Evidence]=[]
        self._seen=set()

    def add_observation(self, obs: ToolObservation):
        self.recent.append(obs)
        if not obs.ok or not obs.content.strip(): return None
        key=sha1((obs.tool+'|'+obs.content[:2000]).encode('utf-8','ignore')).hexdigest()[:12]
        if key in self._seen: return None
        self._seen.add(key)
        meta=obs.metadata
        ev=Evidence(
            evidence_id=f"ev-{key}", kind=obs.tool, source=obs.tool,
            summary=obs.content[:700].replace('\n',' '), excerpt=obs.content[:1800],
            file=meta.get('path'), line_start=meta.get('start_line'), line_end=meta.get('end_line'), confidence=0.65,
        )
        self.pinned.append(ev)
        return ev

    def context(self, max_chars=24000) -> str:
        chunks=[]; used=0
        for e in reversed(self.pinned):
            location=f" {e.file}:{e.line_start}-{e.line_end}" if e.file else ''
            text=f"[{e.evidence_id}] {e.kind}{location}: {e.summary}\n"
            if used+len(text)>max_chars: break
            chunks.append(text); used+=len(text)
        return ''.join(reversed(chunks)) or '(no evidence yet)'
