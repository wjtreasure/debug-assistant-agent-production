from __future__ import annotations
import ast
from pathlib import Path

def _range_lines(line_ranges: list[dict], key_start: str = 'start_line', key_end: str = 'end_line') -> set[int]:
    touched=set()
    for r in line_ranges:
        start=r.get(key_start)
        end=r.get(key_end)
        if start is None or end is None:
            continue
        touched.update(range(int(start),int(end)+1))
    return touched


def _locate(tree: ast.AST, touched: set[int], anchors: list[int]) -> list[dict]:
    parents={}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child]=parent

    def qualified(node):
        names=[node.name]; parent=parents.get(node)
        while parent is not None:
            if isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                names.append(parent.name)
            parent=parents.get(parent)
        return '.'.join(reversed(names))

    out=[]
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            end=getattr(n,'end_lineno',n.lineno)
            if any(n.lineno<=x<=end for x in touched) or any(
                n.lineno <= anchor <= end or n.lineno <= anchor + 1 <= end
                for anchor in anchors
            ):
                out.append({"symbol":n.name,"qualified_symbol":qualified(n),"kind":type(n).__name__,"start_line":n.lineno,"end_line":end})
    # smallest spans first often identify the actual function rather than enclosing class
    return sorted(out,key=lambda x:(x['end_line']-x['start_line'],x['start_line'],x['qualified_symbol']))


def locate_symbols(path:Path, line_ranges:list[dict], insertion_anchors:list[int]|None=None, *, source_text:str|None=None):
    """Locate enclosing symbols for old or new-side edit ranges.

    ``line_ranges`` uses ``start_line``/``end_line``. The old parser's
    ``old_start``/``old_end`` shape is accepted for compatibility.
    """
    try:
        text=source_text if source_text is not None else path.read_text(encoding='utf-8',errors='ignore')
        tree=ast.parse(text)
    except Exception:
        return []
    if line_ranges and 'old_start' in line_ranges[0]:
        touched=_range_lines(line_ranges,'old_start','old_end')
    else:
        touched=_range_lines(line_ranges)
    return _locate(tree,touched,list(insertion_anchors or []))
