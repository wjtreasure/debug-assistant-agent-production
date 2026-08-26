from __future__ import annotations
from pathlib import Path
from typing import Any
import ast, fnmatch, os, re, subprocess, time
from .base import Tool, ToolSpec
from debug_assistant.models import ToolObservation

IGNORED={'.git','.venv','venv','node_modules','dist','build','__pycache__','.tox','.mypy_cache','.pytest_cache'}

def _safe(root: Path, rel: str) -> Path:
    p=(root/rel).resolve(); rr=root.resolve()
    if p != rr and rr not in p.parents: raise ValueError("path escapes repository workspace")
    return p

def _obs(name, started, ok, content, **meta):
    return ToolObservation(name,ok,content,meta, None if ok else meta.get('error_type'), (time.time()-started)*1000)

class RepoTreeTool(Tool):
    spec=ToolSpec('repo_tree','List repository files/directories with bounded depth.')
    def __init__(self, root): self.root=Path(root)
    def execute(self, path='.', depth=3, max_entries=300, **kwargs):
        t=time.time(); base=_safe(self.root,path); out=[]; base_depth=len(base.parts)
        try:
            for cur, dirs, files in os.walk(base):
                dirs[:]=[d for d in dirs if d not in IGNORED]
                d=len(Path(cur).parts)-base_depth
                if d>=int(depth): dirs[:]=[]
                for f in sorted(files):
                    p=Path(cur)/f; out.append(str(p.relative_to(self.root)))
                    if len(out)>=int(max_entries): break
                if len(out)>=int(max_entries): break
            return _obs(self.spec.name,t,True,'\n'.join(out),entries=len(out))
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)

class GrepTool(Tool):
    spec=ToolSpec('grep','Regex/literal search over repository text files.',('query',))
    def __init__(self, root): self.root=Path(root)
    def execute(self, query, glob='*', max_results=80, **kwargs):
        t=time.time(); results=[]
        try: rx=re.compile(query,re.I)
        except re.error: rx=re.compile(re.escape(query),re.I)
        try:
            for p in self.root.rglob('*'):
                if not p.is_file() or any(x in IGNORED for x in p.parts) or not fnmatch.fnmatch(p.name,glob): continue
                try:
                    for i,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1):
                        if rx.search(line):
                            results.append(f"{p.relative_to(self.root)}:{i}: {line[:500]}")
                            if len(results)>=int(max_results): return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results))
                except OSError: pass
            return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results))
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)

class ReadFileTool(Tool):
    spec=ToolSpec('read_file','Read a bounded source range with line numbers.',('path',))
    def __init__(self, root): self.root=Path(root)
    def execute(self,path,start_line=1,end_line=220,**kwargs):
        t=time.time()
        try:
            p=_safe(self.root,path); lines=p.read_text(encoding='utf-8',errors='ignore').splitlines()
            s=max(1,int(start_line)); e=min(len(lines),int(end_line),s+399)
            text='\n'.join(f"{i:5d} | {lines[i-1]}" for i in range(s,e+1))
            return _obs(self.spec.name,t,True,text,path=str(Path(path)),start_line=s,end_line=e)
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)

class SymbolSearchTool(Tool):
    spec=ToolSpec('symbol_search','Find Python classes/functions and their line ranges.',('query',))
    def __init__(self, root): self.root=Path(root)
    def execute(self,query,max_results=60,**kwargs):
        t=time.time(); q=query.lower(); out=[]
        for p in self.root.rglob('*.py'):
            if any(x in IGNORED for x in p.parts): continue
            try: tree=ast.parse(p.read_text(encoding='utf-8',errors='ignore'))
            except Exception: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and q in n.name.lower():
                    out.append(f"{p.relative_to(self.root)}:{n.lineno}-{getattr(n,'end_lineno',n.lineno)} {type(n).__name__} {n.name}")
                    if len(out)>=int(max_results): return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out))
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out))

class GitLogTool(Tool):
    spec=ToolSpec('git_log','Read-only git log, optionally filtered by path.')
    def __init__(self,root): self.root=Path(root)
    def execute(self,path='',max_count=20,**kwargs):
        t=time.time(); cmd=['git','-C',str(self.root),'log',f'-{min(int(max_count),50)}','--oneline','--decorate=no']
        if path: cmd += ['--',str(_safe(self.root,path).relative_to(self.root))]
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            return _obs(self.spec.name,t,r.returncode==0,(r.stdout or r.stderr)[:12000],returncode=r.returncode)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)

class GitShowTool(Tool):
    spec=ToolSpec('git_show','Read one historical commit diff; no checkout or mutation.',('commit',))
    def __init__(self,root): self.root=Path(root)
    def execute(self,commit,path='',**kwargs):
        t=time.time()
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}',str(commit)): return _obs(self.spec.name,t,False,'invalid commit id',error_type='ValidationError')
        cmd=['git','-C',str(self.root),'show','--stat','--patch','--no-ext-diff',str(commit)]
        if path: cmd += ['--',str(_safe(self.root,path).relative_to(self.root))]
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            return _obs(self.spec.name,t,r.returncode==0,(r.stdout or r.stderr)[:14000],returncode=r.returncode)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)

class DiscoverTestsTool(Tool):
    spec=ToolSpec('discover_tests','Find likely tests related to a symbol/path. Does not execute or install anything.')
    def __init__(self,root): self.root=Path(root)
    def execute(self,query='',max_results=80,**kwargs):
        t=time.time(); q=query.lower(); out=[]
        for p in self.root.rglob('*'):
            if not p.is_file() or any(x in IGNORED for x in p.parts): continue
            rel=str(p.relative_to(self.root)); lname=p.name.lower()
            if ('test' in lname or '/tests/' in '/'+rel.replace('\\','/')+'/') and (not q or q in lname or q in rel.lower()):
                out.append(rel)
                if len(out)>=int(max_results): break
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out))
