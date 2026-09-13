from __future__ import annotations

from pathlib import Path
import ast, json, os, re, subprocess, time
from pydantic import Field

from .base import Tool, ToolSpec, ToolArgs
from debug_assistant.models import ToolObservation
from debug_assistant.repository.safe_fs import SafeRepositoryFS, IGNORED, TEXT_SUFFIXES
from debug_assistant.repository.paths import (
    RepositoryPathResolver, RepositoryPathMatcher, ResolutionMode,
    RepositoryPathError, PathRejectedError, PathNotFoundError, normalize_path_syntax,
)


# Source-returning repository tools share these bounds.  ``read_file`` and
# ``symbol_search`` therefore cannot silently grow separate unbounded read
# paths as their result formats evolve.
REPOSITORY_SOURCE_MAX_LINES = 200
REPOSITORY_SOURCE_MAX_CHARS = 12000
REPOSITORY_SYMBOL_MAX_RESULTS = 12


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
    line_count: int = Field(default=REPOSITORY_SOURCE_MAX_LINES, ge=1, le=REPOSITORY_SOURCE_MAX_LINES)

class SymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=REPOSITORY_SYMBOL_MAX_RESULTS, ge=1, le=REPOSITORY_SYMBOL_MAX_RESULTS)

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


def _numbered_source_context(
    lines: list[str], start_line: int, end_line: int, *, max_chars: int,
) -> tuple[str, int | None, int | None, bool]:
    """Return a complete-line, bounded source slice with trustworthy coverage."""
    start = max(1, int(start_line))
    requested_end = max(start, int(end_line))
    bounded_end = min(len(lines), requested_end, start + REPOSITORY_SOURCE_MAX_LINES - 1)
    rendered: list[str] = []
    used = 0
    truncated = bounded_end < requested_end
    for line_no in range(start, bounded_end + 1):
        value = f"{line_no:5d} | {lines[line_no - 1]}"
        addition = len(value) + (1 if rendered else 0)
        if used + addition > max(0, int(max_chars)):
            truncated = True
            break
        rendered.append(value)
        used += addition
    if not rendered:
        return "", None, None, True
    return "\n".join(rendered), start, start + len(rendered) - 1, truncated


def _balanced_brace_end(lines: list[str], start_line: int) -> int:
    """Find a conservative brace-delimited declaration end without parsing code."""
    depth = 0
    opened = False
    block_comment = False
    quote: str | None = None
    escaped = False
    for line_no in range(max(1, int(start_line)), len(lines) + 1):
        line = lines[line_no - 1]
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""
            if block_comment:
                if char == "*" and next_char == "/":
                    block_comment = False
                    index += 2
                    continue
                index += 1
                continue
            if quote is not None:
                if quote != "`" and escaped:
                    escaped = False
                elif quote != "`" and char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue
            if char == "/" and next_char == "/":
                break
            if char == "/" and next_char == "*":
                block_comment = True
                index += 2
                continue
            if char in {'"', "'", '`'}:
                quote = char
            elif char == "{":
                opened = True
                depth += 1
            elif char == "}" and opened:
                depth -= 1
                if depth == 0:
                    return line_no
            index += 1
        if opened and depth == 0:
            return line_no
    # A malformed or brace-less declaration is represented only by its
    # declaration line; callers must not infer a complete body from a fallback.
    return min(len(lines), max(1, int(start_line)))


def _symbol_match(
    *, path: str, name: str, kind: str, start_line: int, end_line: int,
    lines: list[str], max_source_chars: int,
    callers: list[dict] | None = None, callees: list[dict] | None = None,
) -> dict:
    context, context_start, context_end, source_truncated = _numbered_source_context(
        lines, start_line, end_line, max_chars=max_source_chars,
    )
    signature = lines[start_line - 1].strip() if 0 < start_line <= len(lines) else ""
    return {
        "symbol": name,
        "name": name,
        "kind": kind,
        "file": path,
        "start_line": int(start_line),
        "end_line": int(end_line),
        "signature": signature,
        "source_context": context,
        "source_context_start_line": context_start,
        "source_context_end_line": context_end,
        "source_context_truncated": bool(source_truncated),
        "truncated": bool(source_truncated),
        "callers": list(callers or []),
        "callees": list(callees or []),
        "relations_available": bool(callers or callees),
    }


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
        f'Read source lines with stable line numbers. Repository paths are canonicalized; unique read-only suffix/basename recovery is allowed, ambiguity is returned as a structured tool error. Use start_line and line_count; line_count is at most {REPOSITORY_SOURCE_MAX_LINES}.',
        ReadFileArgs,'repository_read','light','none',16000)
    def __init__(self, root, *, fs=None, resolver=None, matcher=None):
        self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
        self._failed_paths=set()
    def execute(self,path,start_line=1,line_count=REPOSITORY_SOURCE_MAX_LINES):
        t=time.time()
        try:
            normalized=normalize_path_syntax(path)
            if normalized in self._failed_paths:
                return _obs(self.spec.name,t,False,
                            f'path not found in repository (cached): {path}',
                            error_type='path_not_found',input_path=path,candidates=[],
                            planner_retryable=False,cached=True)
            resolved=self.resolver.resolve_file(path,mode=ResolutionMode.READ_TOLERANT)
            lines=self.fs.read_text(resolved.relative_path).splitlines()
            s=max(1,int(start_line)); requested_count=int(line_count)
            if requested_count < 1 or requested_count > REPOSITORY_SOURCE_MAX_LINES:
                return _obs(
                    self.spec.name, t, False,
                    f'line_count must be between 1 and {REPOSITORY_SOURCE_MAX_LINES}',
                    error_type='invalid_line_count',
                    requested_start_line=s, requested_line_count=requested_count,
                    path=resolved.relative_path,
                )
            requested_end=s+requested_count-1
            if s > len(lines):
                return _obs(self.spec.name,t,False,
                            f'start_line {s} is beyond end of file',
                            error_type='range_out_of_bounds',actual_line_count=len(lines),
                            requested_start_line=s,requested_line_count=requested_count,
                            path=resolved.relative_path)
            e=min(len(lines),requested_end,s+REPOSITORY_SOURCE_MAX_LINES-1)
            text='\n'.join(f"{i:5d} | {lines[i-1]}" for i in range(s,e+1))
            return _obs(self.spec.name,t,True,text,path=resolved.relative_path,start_line=s,end_line=e,
                        requested_start_line=s,requested_line_count=requested_count,
                        actual_start_line=s,actual_end_line=e,
                        clamped=e<requested_end,truncated=e<min(len(lines),requested_end),path_resolution=resolved.metadata(str(path)))
        except PathNotFoundError as e:
            self._failed_paths.add(normalized)
            return _path_error_obs(self.spec.name,t,e)
        except RepositoryPathError as e: return _path_error_obs(self.spec.name,t,e)
        except Exception as e: return _obs(self.spec.name,t,False,str(e),error_type=type(e).__name__)


class SymbolSearchTool(_RepositoryTool):
    spec=ToolSpec(
        'symbol_search',
        'Locate declared symbols and return bounded candidate context in the same result. '
        'Each match includes a canonical file, signature, exact symbol range, '
        'numbered preview, and caller/callee fields when available. The result is '
        'discovery only; use read_file to verify source and create CODE Evidence.',
        SymbolSearchArgs, 'repository_search', 'medium', 'none', 12000,
    )
    def __init__(self, root, *, fs=None, resolver=None, matcher=None): self._init_paths(root,fs=fs,resolver=resolver,matcher=matcher)
    def execute(self,query,max_results=60):
        t=time.time(); q=str(query).lower(); out=[]; limit=min(int(max_results),REPOSITORY_SYMBOL_MAX_RESULTS); stopped=False
        for sf in self.fs.iter_files(suffixes=TEXT_SUFFIXES):
            try:
                text=self.fs.read_text(sf.rel)
            except (OSError, ValueError):
                continue
            if sf.rel.lower().endswith('.py'):
                try: tree=ast.parse(text)
                except Exception: continue
                lines=text.splitlines()
                for n in ast.walk(tree):
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and q in n.name.lower():
                        out.append((sf.rel, n.name, type(n).__name__, n.lineno,
                                    getattr(n, 'end_lineno', n.lineno), lines))
                        if len(out)>=limit:
                            stopped=True
                            break
                if stopped: break
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                declaration = _generic_symbol_declaration(line)
                if declaration is None or q not in declaration[1].lower():
                    continue
                kind, name = declaration
                out.append((sf.rel, name, kind, line_no,
                            _balanced_brace_end(text.splitlines(), line_no),
                            text.splitlines()))
                if len(out)>=limit:
                    stopped=True
                    break
            if stopped: break

        # The tool output has a shared character budget.  Give the first few
        # matches enough context to be useful while keeping a broad query from
        # expanding into an entire repository dump.  Later matches remain valid
        # retrieval locations, but explicitly report omitted context.
        output_limit = int(self.spec.output_limit or REPOSITORY_SOURCE_MAX_CHARS)
        source_budget = min(REPOSITORY_SOURCE_MAX_CHARS, max(1000, output_limit - output_limit // 3))
        context_matches = max(1, min(len(out), 6))
        per_match_budget = max(1000, source_budget // context_matches)
        matches=[]
        for index, (path, name, kind, start, end, lines) in enumerate(out):
            if index < context_matches:
                budget = min(REPOSITORY_SOURCE_MAX_CHARS, per_match_budget)
                match = _symbol_match(
                    path=path, name=name, kind=kind, start_line=start, end_line=end,
                    lines=lines, max_source_chars=budget,
                )
            else:
                match = {
                    "symbol": name, "name": name, "kind": kind, "file": path,
                    "start_line": int(start), "end_line": int(end),
                    "signature": lines[start - 1].strip() if start <= len(lines) else "",
                    "source_context": "", "source_context_start_line": None,
                    "source_context_end_line": None, "source_context_truncated": True,
                    "truncated": True, "callers": [], "callees": [],
                    "relations_available": False, "source_context_omitted": True,
                }
            matches.append(match)
        payload={
            "query": str(query), "matches": matches,
            "truncated": bool(stopped or any(item.get("truncated") for item in matches)),
            "source_context_budget_chars": source_budget,
        }
        return _obs(
            self.spec.name, t, True, json.dumps(payload, ensure_ascii=False),
            matches=len(matches), truncated=payload["truncated"],
            information_source="candidate_retrieval",
        )


_GO_FUNCTION = re.compile(r'^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(')
_TYPE_DECLARATION = re.compile(r'^\s*(?:type|class|interface|struct|enum)\s+(?P<name>[A-Za-z_]\w*)\b')
_FUNCTION_DECLARATION = re.compile(
    r'^\s*(?:(?:public|private|protected|internal|static|async|virtual|override|final|synchronized|abstract|export)\s+)*'
    r'(?:[A-Za-z_$][\w$<>\[\],.?]*\s+)+(?P<name>[A-Za-z_$][\w$]*)\s*\('
)
_JS_FUNCTION = re.compile(r'^\s*(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(')


def _generic_symbol_declaration(line: str) -> tuple[str, str] | None:
    """Recognize conservative declaration forms outside Python AST parsing."""
    for pattern, kind in (
        (_GO_FUNCTION, "Function"),
        (_TYPE_DECLARATION, "Type"),
        (_JS_FUNCTION, "Function"),
        (_FUNCTION_DECLARATION, "Function"),
    ):
        match = pattern.match(line)
        if match:
            return kind, match.group("name")
    return None


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
