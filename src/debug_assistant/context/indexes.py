from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import re

_LINE_RE = re.compile(r"^\s*(\d+)\s*(?:\||:)\s?")
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
            if obs.tool in {'read_file', 'get_source_code'}:
                p=m.get('path'); lines=retained_source_lines(obs)
                if p and lines: self._ranges[p].append((lines[0][0], lines[-1][0]))
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


def retained_source_lines(obs) -> list[tuple[int, str]]:
    """Exact stored coverage, independent of remotely acquired coverage.

    Explicit adapter format/base metadata wins. Historical pipe/colon inputs
    are accepted only as complete, consecutive numbered records, never fuzzy hits.
    Plain source requires a matching canonical stored span and newline count.
    """
    if obs.tool not in {'read_file', 'get_source_code'} or not obs.content:
        return []
    meta = obs.metadata or {}
    rows = obs.content.splitlines()
    start = meta.get('retained_start_line', meta.get('start_line'))
    end = meta.get('retained_end_line', meta.get('end_line'))
    fmt = meta.get('source_body_format')
    if fmt not in {None, 'plain', 'numbered_pipe', 'numbered_colon'}:
        return []
    matches = [_LINE_RE.match(line) for line in rows] if fmt != 'plain' else []
    if matches and all(matches):
        numbers = [int(match.group(1)) for match in matches]
        if numbers != list(range(numbers[0], numbers[0] + len(rows))):
            return []
        if fmt == 'numbered_pipe' and any('|' not in line[:match.end()] for line, match in zip(rows, matches)):
            return []
        if fmt == 'numbered_colon' and any(':' not in line[:match.end()] for line, match in zip(rows, matches)):
            return []
        if start is not None and start != numbers[0] or end is not None and end != numbers[-1]:
            return []
        return list(zip(numbers, rows))
    if fmt in {'numbered_pipe', 'numbered_colon'}:
        return []
    if (type(start) is int and type(end) is int and start >= 1
            and end - start + 1 == len(rows)):
        return [(start + i, f'{start + i} | {line}') for i, line in enumerate(rows)]
    return []


def rendered_line_ranges(content: str) -> list[tuple[int, int]]:
    """Only actual deterministic numbered records become display coverage."""
    return merge_ranges([(int(match.group(1)), int(match.group(1)))
                         for line in content.splitlines() if (match := _LINE_RE.match(line))])
