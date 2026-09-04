from __future__ import annotations

from pathlib import Path
import ast, os, re, subprocess, time
from pydantic import Field, model_validator

from .base import Tool, ToolSpec, ToolArgs
from debug_assistant.models import ToolObservation
from debug_assistant.repository.safe_fs import SafeRepositoryFS, IGNORED
from debug_assistant.repository.paths import (
    RepositoryPathResolver, RepositoryPathMatcher, ResolutionMode,
    RepositoryPathError, PathRejectedError,
)


def _obs(name, started, ok, content, *, error_type=None, retryable=False, **meta):
    meta.setdefault('retryable', retryable)
    return ToolObservation(name,ok,content,meta,error_type,(time.time()-started)*1000)


def _path_error_obs(name, started, exc: RepositoryPathError, **meta):
    payload=exc.metadata(); payload.update(meta)
    retryable=bool(payload.pop('retryable',False))
    return _obs(name,started,False,str(exc),error_type=exc.error_type,retryable=retryable,**payload)


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


class _RepositoryTool(Tool):
    def _init_paths(self, root, *, fs=None, resolver=None, matcher=None):
        self.root=Path(root).resolve()
        self.fs=fs or SafeRepositoryFS(self.root)
        self.resolver=resolver or RepositoryPathResolver(self.fs)
        self.matcher=matcher or RepositoryPathMatcher()


class RepoTreeTool(_RepositoryTool):
    spec=ToolSpec(
        'repo_tree',
        'List repository files/directories with bounded depth. Path inputs are canonicalized; narrow repository-relative paths are preferred.',
        RepoTreeArgs,'repository_read','light','none',12000)
    def __init__(self, root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self, path='.', depth=3, max_entries=300):
        t=time.time()
        try:
            resolved=self.resolver.resolve_directory(path,mode=ResolutionMode.READ_TOLERANT)
            base=resolved.absolute_path; out=[]; base_depth=len(base.parts)
            for cur, dirs, files in os.walk(base,followlinks=False):
                dirs[:]=[d for d in dirs if d not in IGNORED]
                d=len(Path(cur).parts)-base_depth
                if d>=int(depth): dirs[:]=[]
                for f in sorted(files):
                    raw=Path(cur)/f
                    try:
                        rp=raw.resolve()
                        if rp != self.fs.root and self.fs.root not in rp.parents: continue
                    except OSError: continue
                    out.append(str(raw.relative_to(self.fs.root)).replace('\\','/'))
                    if len(out)>=int(max_entries): break
                if len(out)>=int(max_entries): break
            return _obs(self.spec.name,t,True,'\n'.join(out),entries=len(out),truncated=len(out)>=int(max_entries),
                        path=resolved.relative_path,path_resolution=resolved.metadata(str(path)))
        except RepositoryPathError as e: return _path_error_obs(self.spec.name,t,e)
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class GrepTool(_RepositoryTool):
    spec=ToolSpec(
        'grep',
        "Regex/literal search over repository text files. glob without '/' matches basenames recursively (e.g. *.py); glob with '/' is anchored to the repository path (e.g. astroid/*.py, astroid/modutils.py, astroid/**/*.py).",
        GrepArgs,'repository_search','light','none',14000)
    def __init__(self, root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self, query, glob='*', max_results=50):
        t=time.time(); results=[]; limit=min(int(max_results),50); truncated=False
        try: rx=re.compile(query,re.I)
        except re.error: rx=re.compile(re.escape(query),re.I)
        try:
            pattern=self.matcher.normalize_pattern(glob)
            for sf in self.fs.iter_files():
                if not self.matcher.matches(sf.rel,pattern): continue
                try:
                    for i,line in enumerate(self.fs.read_text(sf.rel).splitlines(),1):
                        if rx.search(line):
                            results.append(f"{sf.rel}:{i}: {line[:500]}")
                            if len(results)>=limit:
                                truncated=True
                                return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results),truncated=truncated,max_results=limit,
                                            path_pattern=pattern.metadata(str(glob)))
                except OSError: pass
            return _obs(self.spec.name,t,True,'\n'.join(results),matches=len(results),truncated=False,max_results=limit,
                        path_pattern=pattern.metadata(str(glob)))
        except RepositoryPathError as e: return _path_error_obs(self.spec.name,t,e,pattern=str(glob))
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class ReadFileTool(_RepositoryTool):
    spec=ToolSpec(
        'read_file',
        'Read source lines with stable line numbers. Repository paths are canonicalized; unique read-only suffix/basename recovery is allowed, ambiguity is returned as a structured tool error. start_line/end_line are inclusive and at most 200 lines.',
        ReadFileArgs,'repository_read','light','none',16000)
    def __init__(self, root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,path,start_line=1,end_line=200):
        t=time.time()
        try:
            resolved=self.resolver.resolve_file(path,mode=ResolutionMode.READ_TOLERANT)
            lines=self.fs.read_text(resolved.relative_path).splitlines()
            s=max(1,int(start_line)); requested_end=int(end_line)
            if s > len(lines):
                return _obs(self.spec.name,t,False,
                            f'start_line {s} is beyond end of file',
                            error_type='range_out_of_bounds',actual_line_count=len(lines),
                            requested_start_line=s,requested_end_line=requested_end,
                            path=resolved.relative_path)
            e=min(len(lines),requested_end,s+199)
            text='\n'.join(f"{i:5d} | {lines[i-1]}" for i in range(s,e+1))
            return _obs(self.spec.name,t,True,text,path=resolved.relative_path,start_line=s,end_line=e,requested_end_line=requested_end,
                        requested_start_line=s,actual_start_line=s,actual_end_line=e,
                        clamped=e<requested_end,truncated=e<min(len(lines),requested_end),path_resolution=resolved.metadata(str(path)))
        except RepositoryPathError as e: return _path_error_obs(self.spec.name,t,e)
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class SymbolSearchTool(_RepositoryTool):
    spec=ToolSpec('symbol_search','Find Python classes/functions and their line ranges. Results always use canonical repository-relative paths.',SymbolSearchArgs,'repository_search','medium','none',12000)
    def __init__(self, root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,query,max_results=60):
        t=time.time(); q=query.lower(); out=[]; limit=min(int(max_results),60)
        for sf in self.fs.iter_files(suffixes={'.py'}):
            try: tree=ast.parse(self.fs.read_text(sf.rel))
            except Exception: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and q in n.name.lower():
                    out.append(f"{sf.rel}:{n.lineno}-{getattr(n,'end_lineno',n.lineno)} {type(n).__name__} {n.name}")
                    if len(out)>=limit: return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=True)
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=False)


class GitLogTool(_RepositoryTool):
    spec=ToolSpec('git_log','Read-only git log, optionally filtered by an exact repository path. Fuzzy path recovery is disabled.',GitLogArgs,'git_read','light','none',12000)
    def __init__(self,root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,path='',max_count=20):
        t=time.time(); cmd=['git','-C',str(self.root),'log',f'-{min(int(max_count),50)}','--oneline','--decorate=no']; resolution=None
        try:
            if path:
                resolution=self.resolver.resolve_path(path,mode=ResolutionMode.EXACT)
                cmd += ['--',resolution.relative_path]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            ok=r.returncode==0; err=None if ok else 'git_error'
            meta={'returncode':r.returncode}
            if resolution: meta['path_resolution']=resolution.metadata(str(path)); meta['path']=resolution.relative_path
            return _obs(self.spec.name,t,ok,(r.stdout or r.stderr)[:12000],error_type=err,**meta)
        except RepositoryPathError as e:return _path_error_obs(self.spec.name,t,e)
        except subprocess.TimeoutExpired as e:return _obs(self.spec.name,t,False,str(e),error_type='timeout',retryable=True)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class GitShowTool(_RepositoryTool):
    spec=ToolSpec('git_show','Read one historical commit diff; optional path filter is exact-only. No checkout or mutation.',GitShowArgs,'git_read','medium','none',14000)
    def __init__(self,root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,commit,path=''):
        t=time.time()
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}',str(commit)): return _obs(self.spec.name,t,False,'invalid commit id',error_type='schema_validation')
        cmd=['git','-C',str(self.root),'show','--stat','--patch','--no-ext-diff',str(commit)]; resolution=None
        try:
            if path:
                resolution=self.resolver.resolve_path(path,mode=ResolutionMode.EXACT)
                cmd += ['--',resolution.relative_path]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
            ok=r.returncode==0; meta={'returncode':r.returncode,'truncated':len(r.stdout or r.stderr)>14000}
            if resolution: meta['path_resolution']=resolution.metadata(str(path)); meta['path']=resolution.relative_path
            return _obs(self.spec.name,t,ok,(r.stdout or r.stderr)[:14000],error_type=None if ok else 'git_error',**meta)
        except RepositoryPathError as e:return _path_error_obs(self.spec.name,t,e)
        except subprocess.TimeoutExpired as e:return _obs(self.spec.name,t,False,str(e),error_type='timeout',retryable=True)
        except Exception as e:return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class DiscoverTestsTool(_RepositoryTool):
    spec=ToolSpec('discover_tests','Find likely tests related to a symbol/path. Does not execute or install anything. Results use canonical repository-relative paths.',DiscoverTestsArgs,'test_discovery','medium','none',12000)
    def __init__(self,root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,query='',max_results=50):
        t=time.time(); q=query.lower(); out=[]; limit=min(int(max_results),50)
        for sf in self.fs.iter_files():
            rel=sf.rel; lname=Path(rel).name.lower()
            if ('test' in lname or '/tests/' in '/'+rel+'/') and (not q or q in lname or q in rel.lower()):
                out.append(rel)
                if len(out)>=limit: break
        return _obs(self.spec.name,t,True,'\n'.join(out),matches=len(out),truncated=len(out)>=limit)
