from __future__ import annotations
from pathlib import Path
import ast, fnmatch, os, re, subprocess, time
from pydantic import Field, model_validator
from .base import Tool, ToolSpec, ToolArgs
from debug_assistant.models import ToolObservation

IGNORED={'.git','.venv','venv','node_modules','dist','build','__pycache__','.tox','.mypy_cache','.pytest_cache'}


def _safe(root: Path, rel: str) -> Path:
    p=(root/rel).resolve(); rr=root.resolve()
    if p != rr and rr not in p.parents: raise ValueError("path escapes repository workspace")
    return p


def _obs(name, started, ok, content, *, error_type=None, retryable=False, **meta):
    meta.setdefault('retryable', retryable)
    return ToolObservation(name,ok,content,meta,error_type,(time.time()-started)*1000)


class RepoTreeArgs(ToolArgs):
    path: str = '.'
    depth: int = Field(default=3, ge=1, le=8)
    max_entries: int = Field(default=300, ge=1, le=300)

class GrepArgs(ToolArgs):
    query: str = Field(min_length=1)
    glob: str = '*'
    max_results: int = Field(default=50, ge=1, le=50)

class ReadFileArgs(ToolArgs):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)
    @model_validator(mode='after')
    def check_range(self):
        if self.start_line > self.end_line:
            raise ValueError('start_line must be <= end_line')
        if self.end_line - self.start_line + 1 > 200:
            raise ValueError('read_file may request at most 200 lines per call')
        return self

class SymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=60, ge=1, le=60)

class GitLogArgs(ToolArgs):
    path: str = ''
    max_count: int = Field(default=20, ge=1, le=50)

class GitShowArgs(ToolArgs):
    commit: str = Field(min_length=7, max_length=40)
    path: str = ''

class DiscoverTestsArgs(ToolArgs):
    query: str = ''
    max_results: int = Field(default=50, ge=1, le=50)


class RepoTreeTool(Tool):
    spec=ToolSpec('repo_tree','List repository files/directories with bounded depth.',RepoTreeArgs,'repository_read','light','none',12000)
    def __init__(self, root): self.root=Path(root)
    def execute(self, path='.', depth=3, max_entries=300):
        t=time.time()
        try:
            base=_safe(self.root,path); out=[]; base_depth=len(base.parts)
            for cur, dirs, files in os.walk(base):
                dirs[:]=[d for d in dirs if d not in IGNORED]
                d=len(Path(cur).parts)-base_depth
                if d>=int(depth): dirs[:]=[]
                for f in sorted(files):
                    p=Path(cur)/f; out.append(str(p.relative_to(self.root)))
                    if len(out)>=int(max_entries): break
                if len(out)>=int(max_entries): break
            return _obs(self.spec.name,t,True,'\n'.join(out),entries=len(out),truncated=len(out)>=int(max_entries))
        except ValueError as e: return _obs(self.spec.name,t,False,str(e),error_type='path_violation')
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class GrepTool(Tool):
    spec=ToolSpec('grep','Regex/literal search over repository text files.',GrepArgs,'repository_search','light','none',14000)
    def __init__(self, root): self.root=Path(root)
    def execute(self, query, glob='*', max_results=50):
        t=time.time(); results=[]; limit=min(int(max_results),50); truncated=False
        try: rx=re.compile(query,re.I)
        except re.error: rx=re.compile(re.escape(query),re.I)
        try:
            for p in self.root.rglob('*'):
                if not p.is_file() or any(x in IGNORED for x in p.parts) or not fnmatch.fnmatch(p.name,glob): continue
                try:
                    for i,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1):
                        if rx.search(line):
                            results.append(f"{p.relative_to(self.root)}:{i}: {line[:500]}")
                            if len(results)>=limit:
                                truncated=True
                                return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results),truncated=truncated,max_results=limit)
                except OSError: pass
            return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results),truncated=False,max_results=limit)
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class ReadFileTool(Tool):
    spec=ToolSpec('read_file','Read source lines with stable line numbers. start_line/end_line are inclusive and end_line - start_line + 1 must be <= 200 (e.g. 200-399 is valid; 200-400 is 201 lines).',ReadFileArgs,'repository_read','light','none',16000)
    def __init__(self, root): self.root=Path(root)
    def execute(self,path,start_line=1,end_line=200):
        t=time.time()
        try:
            p=_safe(self.root,path); lines=p.read_text(encoding='utf-8',errors='ignore').splitlines()
            s=max(1,int(start_line)); requested_end=int(end_line); e=min(len(lines),requested_end,s+199)
            text='\n'.join(f"{i:5d} | {lines[i-1]}" for i in range(s,e+1))
            return _obs(self.spec.name,t,True,text,path=str(Path(path)),start_line=s,end_line=e,requested_end_line=requested_end,truncated=e<min(len(lines),requested_end))
        except ValueError as e: return _obs(self.spec.name,t,False,str(e),error_type='path_violation')
        except FileNotFoundError as e: return _obs(self.spec.name,t,False,str(e),error_type='not_found')
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class SymbolSearchTool(Tool):
    spec=ToolSpec('symbol_search','Find Python classes/functions and their line ranges.',SymbolSearchArgs,'repository_search','medium','none',12000)
    def __init__(self, root): self.root=Path(root)
    def execute(self,query,max_results=60):
        t=time.time(); q=query.lower(); out=[]; limit=min(int(max_results),60)
        for p in self.root.rglob('*.py'):
            if any(x in IGNORED for x in p.parts): continue
            try: tree=ast.parse(p.read_text(encoding='utf-8',errors='ignore'))
            except Exception: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and q in n.name.lower():
                    out.append(f"{p.relative_to(self.root)}:{n.lineno}-{getattr(n,'end_lineno',n.lineno)} {type(n).__name__} {n.name}")
                    if len(out)>=limit: return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=True)
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=False)


class GitLogTool(Tool):
    spec=ToolSpec('git_log','Read-only git log, optionally filtered by path.',GitLogArgs,'git_read','light','none',12000)
    def __init__(self,root): self.root=Path(root)
    def execute(self,path='',max_count=20):
        t=time.time(); cmd=['git','-C',str(self.root),'log',f'-{min(int(max_count),50)}','--oneline','--decorate=no']
        try:
            if path: cmd += ['--',str(_safe(self.root,path).relative_to(self.root))]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            ok=r.returncode==0; err=None if ok else 'git_error'
            return _obs(self.spec.name,t,ok,(r.stdout or r.stderr)[:12000],error_type=err,returncode=r.returncode)
        except ValueError as e:return _obs(self.spec.name,t,False,str(e),error_type='path_violation')
        except subprocess.TimeoutExpired as e:return _obs(self.spec.name,t,False,str(e),error_type='timeout',retryable=True)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class GitShowTool(Tool):
    spec=ToolSpec('git_show','Read one historical commit diff; no checkout or mutation.',GitShowArgs,'git_read','medium','none',14000)
    def __init__(self,root): self.root=Path(root)
    def execute(self,commit,path=''):
        t=time.time()
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}',str(commit)): return _obs(self.spec.name,t,False,'invalid commit id',error_type='schema_validation')
        cmd=['git','-C',str(self.root),'show','--stat','--patch','--no-ext-diff',str(commit)]
        try:
            if path: cmd += ['--',str(_safe(self.root,path).relative_to(self.root))]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            ok=r.returncode==0
            return _obs(self.spec.name,t,ok,(r.stdout or r.stderr)[:14000],error_type=None if ok else 'git_error',returncode=r.returncode,truncated=len(r.stdout or r.stderr)>14000)
        except ValueError as e:return _obs(self.spec.name,t,False,str(e),error_type='path_violation')
        except subprocess.TimeoutExpired as e:return _obs(self.spec.name,t,False,str(e),error_type='timeout',retryable=True)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class DiscoverTestsTool(Tool):
    spec=ToolSpec('discover_tests','Find likely tests related to a symbol/path. Does not execute or install anything.',DiscoverTestsArgs,'test_discovery','medium','none',12000)
    def __init__(self,root): self.root=Path(root)
    def execute(self,query='',max_results=50):
        t=time.time(); q=query.lower(); out=[]; limit=min(int(max_results),50)
        for p in self.root.rglob('*'):
            if not p.is_file() or any(x in IGNORED for x in p.parts): continue
            rel=str(p.relative_to(self.root)); lname=p.name.lower()
            if ('test' in lname or '/tests/' in '/'+rel.replace('\\','/')+'/') and (not q or q in lname or q in rel.lower()):
                out.append(rel)
                if len(out)>=limit: break
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=len(out)>=limit)
