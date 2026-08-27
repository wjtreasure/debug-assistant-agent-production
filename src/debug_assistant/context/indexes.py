from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import re

_LINE_RE = re.compile(r"^\s*(\d+)\s*\|")
_GREP_RE = re.compile(r"(?P<path>[^\s:]+):(?P<line>\d+):")


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    clean=sorted((min(a,b),max(a,b)) for a,b in ranges if a is not None and b is not None)
    if not clean: return []
    out=[clean[0]]
    for start,end in clean[1:]:
        ps,pe=out[-1]
        if start <= pe + 1:
            out[-1]=(ps,max(pe,end))
        else:
            out.append((start,end))
    return out


class DisplayCoverageIndex:
    """Coverage of source lines actually rendered into the last LLM prompt.

    This index is rebuilt only after final context rendering/packing. It must never be
    populated from raw read coverage or from planned-but-dropped projections.
    """
    def __init__(self): self._by_file: dict[str,list[tuple[int,int,str]]] = {}

    def clear(self) -> None: self._by_file.clear()

    def add(self,path:str,start_line:int,end_line:int,projection_id:str) -> None:
        if not path or start_line is None or end_line is None: return
        self._by_file.setdefault(path,[]).append((start_line,end_line,projection_id))

    def covers(self,path:str,start_line:int,end_line:int) -> bool:
        ranges=merge_ranges([(a,b) for a,b,_ in self._by_file.get(path,[])])
        return any(a <= start_line and b >= end_line for a,b in ranges)

    def export(self) -> dict[str,list[tuple[int,int]]]:
        return {p:merge_ranges([(a,b) for a,b,_ in ents]) for p,ents in self._by_file.items()}


@dataclass(slots=True,frozen=True)
class KnownRange:
    path: str
    start_line: int
    end_line: int
    source: str = "read_file"


class KnownContextIndex:
    """Tiny model-visible pointer index. It exposes existence, never raw bodies."""
    def __init__(self):
        self._ranges: dict[str,list[tuple[int,int]]] = defaultdict(list)
        self._grep_points: dict[str,set[int]] = defaultdict(set)

    def rebuild(self, observations, read_coverage=None) -> None:
        self._ranges.clear(); self._grep_points.clear()
        for obs in observations.all():
            m=obs.metadata or {}
            if obs.tool=='read_file':
                p=m.get('path'); a=m.get('start_line'); b=m.get('end_line')
                if p and isinstance(a,int) and isinstance(b,int): self._ranges[p].append((a,b))
            elif obs.tool in {'grep','code_search','symbol_search'}:
                for match in _GREP_RE.finditer(obs.content or ''):
                    try: self._grep_points[match.group('path')].add(int(match.group('line')))
                    except ValueError: pass
        for p in list(self._ranges): self._ranges[p]=merge_ranges(self._ranges[p])

    def _point_superseded(self,path:str,line:int) -> bool:
        return any(a <= line <= b for a,b in self._ranges.get(path,[]))

    def filter_search_content(self, content: str) -> tuple[str,int]:
        """Drop only search-hit lines whose exact file/line is covered by a read range.

        This is range-level supersession: unrelated hits from the same observation remain.
        """
        kept=[]; removed=0
        for line in (content or '').splitlines():
            m=_GREP_RE.search(line)
            if m:
                try:
                    n=int(m.group('line')); path=m.group('path')
                except Exception:
                    kept.append(line); continue
                if self._point_superseded(path,n):
                    removed+=1; continue
            kept.append(line)
        return '\n'.join(kept),removed

    def render(self,max_chars:int=3500) -> str:
        rows=[]; used=0
        paths=sorted(set(self._ranges)|set(self._grep_points))
        for path in paths:
            parts=[]
            ranges=self._ranges.get(path,[])
            if ranges:
                parts.append('read ' + ', '.join(f'{a}-{b}' for a,b in ranges))
            remaining=sorted(x for x in self._grep_points.get(path,set()) if not self._point_superseded(path,x))
            if remaining:
                shown=remaining[:8]
                tail='…' if len(remaining)>len(shown) else ''
                parts.append('search hits ' + ','.join(map(str,shown)) + tail)
            if not parts: continue
            row=f"- {path}: {'; '.join(parts)}\n"
            if used+len(row)>max_chars: break
            rows.append(row); used+=len(row)
        return ''.join(rows) or '(none)'

    def stats(self) -> dict:
        return {
            'known_files':len(set(self._ranges)|set(self._grep_points)),
            'known_read_ranges':sum(len(x) for x in self._ranges.values()),
            'unsuperseded_search_points':sum(sum(not self._point_superseded(p,n) for n in pts) for p,pts in self._grep_points.items()),
            'superseded_search_points':sum(sum(self._point_superseded(p,n) for n in pts) for p,pts in self._grep_points.items()),
        }


def extract_numbered_range(content:str,start_line:int,end_line:int) -> tuple[str,int|None,int|None]:
    """Return exact numbered lines from an immutable read_file observation."""
    kept=[]; nums=[]
    for line in (content or '').splitlines():
        m=_LINE_RE.match(line)
        if not m: continue
        n=int(m.group(1))
        if start_line <= n <= end_line:
            kept.append(line); nums.append(n)
    if not kept: return '',None,None
    return '\n'.join(kept),nums[0],nums[-1]
