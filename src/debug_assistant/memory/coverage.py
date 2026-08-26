from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ReadCoverageEntry:
    path: str
    start_line: int
    end_line: int
    observation_id: str

class ReadCoverageIndex:
    """Task-scoped exact/full-range reuse for immutable read_file observations."""
    def __init__(self): self._by_file: dict[str, list[ReadCoverageEntry]] = {}

    def add(self, *, path: str, start_line: int, end_line: int, observation_id: str) -> None:
        if not path or start_line is None or end_line is None: return
        ent=ReadCoverageEntry(path,start_line,end_line,observation_id)
        entries=self._by_file.setdefault(path,[])
        if ent not in entries: entries.append(ent)

    def find_covering(self, *, path: str, start_line: int, end_line: int) -> ReadCoverageEntry | None:
        candidates=[e for e in self._by_file.get(path,[]) if e.start_line <= start_line and e.end_line >= end_line]
        if not candidates: return None
        return min(candidates,key=lambda e:(e.end_line-e.start_line,e.start_line))
