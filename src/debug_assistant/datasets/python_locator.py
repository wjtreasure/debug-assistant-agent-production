from __future__ import annotations
import ast
from pathlib import Path

def locate_symbols(path:Path, line_ranges:list[dict]):
    try: tree=ast.parse(path.read_text(encoding='utf-8',errors='ignore'))
    except Exception: return []
    touched=set()
    for r in line_ranges: touched.update(range(r['old_start'],r['old_end']+1))
    out=[]
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            end=getattr(n,'end_lineno',n.lineno)
            if any(n.lineno<=x<=end for x in touched):
                out.append({"symbol":n.name,"kind":type(n).__name__,"start_line":n.lineno,"end_line":end})
    # smallest spans first often identify the actual function rather than enclosing class
    return sorted(out,key=lambda x:(x['end_line']-x['start_line'],x['start_line']))
