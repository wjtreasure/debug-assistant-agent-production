from __future__ import annotations

"""Bounded, read-only CloudOps application-source tools.

This is the CloudOps source adapter used by the optional code-observation
capability. It intentionally contains no SWE planner, benchmark, or generic
repository ToolRegistry dependency.
"""

import ast
import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from debug_assistant.models import ToolObservation
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.tools.base import Tool, ToolArgs, ToolSpec
from debug_assistant.tools.official_cloudops import OfficialCloudOps, official_result_success


MAX_SOURCE_LINES = 800
_TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".c", ".h", ".cc", ".cpp", ".hpp", ".java", ".js", ".ts", ".tsx", ".go", ".rs"}
_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/]|//|\\\\)")


class SourceBindingError(ValueError):
    """The case source directory cannot be safely bound."""


class CloudOpsSourceBinding:
    relative_root = "code"

    def __init__(self, runtime_data_dir: str | Path):
        self.case_root = Path(runtime_data_dir).resolve()
        candidate = self.case_root / self.relative_root
        if not candidate.exists():
            self.source_root = None
            self.available = False
            return
        try:
            root = candidate.resolve(strict=True)
        except OSError as exc:
            raise SourceBindingError("application source root cannot be resolved") from exc
        if not root.is_dir() or (root != self.case_root and self.case_root not in root.parents):
            raise SourceBindingError("application source root is invalid")
        self.source_root = root
        self.available = True

    def filesystem(self) -> SafeRepositoryFS:
        if not self.available or self.source_root is None:
            raise SourceBindingError("application source is unavailable")
        return SafeRepositoryFS(self.source_root)


class RepoTreeArgs(ToolArgs):
    path: str = "."


class GrepArgs(ToolArgs):
    pattern: str = Field(min_length=1)
    glob: str = "*"


class ReadFileArgs(ToolArgs):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=MAX_SOURCE_LINES, ge=1, le=MAX_SOURCE_LINES)


class SourceCodeArgs(ToolArgs):
    app_name: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    line_count: int | None = Field(default=None, ge=1, le=MAX_SOURCE_LINES)


class SymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)


class CodeSearchArgs(ToolArgs):
    query: str = Field(min_length=1)


class InspectSymbolContextArgs(ToolArgs):
    symbol: str = Field(min_length=1)
    file: str = Field(min_length=1)
    include_source: bool = False


def _normalize_path(raw: Any, root: Path) -> tuple[str | None, dict[str, str], str | None]:
    original = str(raw).strip().strip("'\"`").replace("\\", "/") or "."
    if ".." in [part for part in original.split("/") if part not in {"", "."}]:
        return None, {"original_path": original, "normalized_path": original, "rule_applied": "rejected_out_of_root", "result": "rejected"}, "source path escapes bound code root"
    value = original
    if _ABSOLUTE_PATH.match(value) or Path(value).is_absolute():
        try:
            value = Path(value).resolve(strict=False).relative_to(root).as_posix() or "."
        except (OSError, ValueError):
            return None, {"original_path": original, "normalized_path": "", "rule_applied": "rejected_out_of_root", "result": "rejected"}, "absolute source path is outside bound code root"
        rule = "abs_to_rel"
    else:
        value = value.lstrip("/") or "."
        if value == "code":
            value, rule = ".", "stripped_bound_root_name"
        elif value.startswith("code/"):
            value, rule = value[5:] or ".", "stripped_bound_root_name"
        elif value.startswith("src/"):
            # The case binding exposes ``code`` as the source root, while
            # planners may address the same checkout using its common
            # repository-level ``src/`` prefix.  Accept this alias only when
            # the unprefixed path exists; never perform basename recovery or
            # silently choose among duplicate files.
            candidate = root / value[4:]
            if not (root / value).exists() and candidate.exists():
                value, rule = value[4:] or ".", "stripped_source_alias"
            else:
                rule = "already_relative"
        else:
            rule = "already_relative"
    return value, {"original_path": original, "normalized_path": value, "rule_applied": rule, "result": "accepted"}, None


def source_provenance_profile(operation):
    if operation in {"read_file", "get_source_code"}:
        return {"context_kind": "CODE", "information_source": "source_read"}
    return {"context_kind": "CODE_INDEX" if operation == "repo_tree" else "CODE_SEARCH",
            "information_source": "candidate_retrieval"}


def source_provenance_categories(operation):
    from debug_assistant.tools.provenance import profile_categories
    return profile_categories((source_provenance_profile(operation),))


class _SourceTool(Tool):
    def __init__(self, spec: ToolSpec, binding: CloudOpsSourceBinding, fn: Callable[[dict[str, Any]], Any]):
        self.spec, self.binding, self._fn = spec, binding, fn

    def provenance_profiles(self):
        return (source_provenance_profile(self.spec.name),)

    def provenance_categories(self):
        return source_provenance_categories(self.spec.name)

    def execute(self, **kwargs: Any) -> ToolObservation:
        try:
            result = self._fn(kwargs)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            metadata = {"arguments": dict(kwargs), "source_scope": "code", "source_relative_root": "code", "read_only": True}
            if self.spec.name in {"read_file", "get_source_code"}:
                payload = result if isinstance(result, dict) else {}
                metadata.update(payload.get("metadata") or {})
                metadata.update(source_provenance_profile(self.spec.name))
                metadata.setdefault("provenance", {"source_root": "code", "path": metadata.get("path"), "start_line": metadata.get("start_line"), "end_line": metadata.get("end_line")})
                content = str(payload.get("content", ""))
            elif self.spec.name in {"repo_tree", "grep", "symbol_search", "code_search", "inspect_symbol_context"}:
                metadata.update(source_provenance_profile(self.spec.name))
            return ToolObservation(tool=self.spec.name, ok=True, content=content, metadata=metadata)
        except FileNotFoundError:
            # Keep exact-path lookup strict, but expose bounded candidates so a
            # basename such as main.go can recover without repeating the error.
            candidates: list[str] = []
            if self.spec.name == "read_file" and self.binding.source_root is not None:
                requested = str(kwargs.get("path") or "").replace("\\", "/").strip("/")
                basename = Path(requested).name
                if basename:
                    try:
                        candidates = sorted(
                            item.rel for item in self.binding.filesystem().iter_files()
                            if Path(item.rel).name == basename
                        )[:8]
                    except OSError:
                        candidates = []
            payload = {
                "status": "UNAVAILABLE",
                "reason": "source_path_not_found",
                "requested_path": str(kwargs.get("path") or ""),
                "candidate_paths": candidates,
                "hint": "Use an exact relative path from repo_tree or code_search before retrying read_file.",
            }
            return ToolObservation(
                tool=self.spec.name,
                ok=False,
                content=json.dumps(payload, ensure_ascii=False),
                metadata={
                    "arguments": dict(kwargs), "source_scope": "code", "read_only": True,
                    "status": "UNAVAILABLE", "error_type": "source_path_not_found",
                    "candidate_paths": candidates,
                },
                error_type="source_path_not_found",
            )
        except Exception as exc:
            return ToolObservation(tool=self.spec.name, ok=False, content=str(exc), metadata={"arguments": dict(kwargs), "source_scope": "code", "read_only": True}, error_type="source_tool_error")


class CloudOpsSourceToolRegistry:
    _EXPOSED = ("repo_tree", "grep", "read_file", "symbol_search", "code_search", "inspect_symbol_context")

    def __init__(self, binding: CloudOpsSourceBinding, *, search_engine=None, official=None):
        if not binding.available or binding.source_root is None:
            raise SourceBindingError("application source is unavailable")
        self.binding = binding
        self.official = official or OfficialCloudOps(binding.case_root)
        self.search_engine = None
        self._tools = {
            "repo_tree": _SourceTool(ToolSpec("repo_tree", "List source files under a bounded path.", RepoTreeArgs, capability="source_read", parallel_safe=True), binding, self._repo_tree),
            "grep": _SourceTool(ToolSpec("grep", "Search source text for an exact pattern.", GrepArgs, capability="source_read", parallel_safe=True), binding, self._grep),
            "read_file": _SourceTool(ToolSpec("read_file", "Read a bounded source range using an exact relative path previously returned by repo_tree/code_search; only this operation can produce source evidence.", ReadFileArgs, capability="source_read", parallel_safe=True, companion_tools=("repo_tree", "grep", "symbol_search", "code_search")), binding, self._read_file),
            "get_source_code": _SourceTool(ToolSpec("get_source_code", "Acquire listed app source through official GetSourceCode with optional visible line range.", SourceCodeArgs, capability="source_read"), binding, self._get_source_code),
            "symbol_search": _SourceTool(ToolSpec("symbol_search", "Find source symbols by name.", SymbolSearchArgs, capability="source_read", parallel_safe=True), binding, self._symbol_search),
            "code_search": _SourceTool(ToolSpec("code_search", "Find candidate source files and symbols; verify with read_file.", CodeSearchArgs, capability="source_read", parallel_safe=True), binding, self._code_search),
            "inspect_symbol_context": _SourceTool(ToolSpec("inspect_symbol_context", "Inspect bounded Python symbol callers and callees.", InspectSymbolContextArgs, capability="source_read", parallel_safe=True), binding, self._inspect_symbol),
        }

    def tools(self) -> dict[str, Tool]:
        return dict(self._tools)

    def _fs_path(self, raw: Any) -> tuple[Path, dict[str, str]]:
        assert self.binding.source_root is not None
        normalized, action, reason = _normalize_path(raw, self.binding.source_root)
        if reason or normalized is None:
            raise ValueError(reason or "invalid source path")
        return self.binding.filesystem().resolve(normalized), action

    def _repo_tree(self, args: dict[str, Any]) -> dict[str, Any]:
        path, _ = self._fs_path(args.get("path", "."))
        root = self.binding.source_root
        files = [item.rel for item in self.binding.filesystem().iter_files(path=path, suffixes=_TEXT_SUFFIXES)]
        return {"root": str(path.relative_to(root) or ".").replace("\\", "/"), "files": sorted(files)[:500]}

    def _grep(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        fs = self.binding.filesystem()
        pattern, glob = str(args["pattern"]), str(args.get("glob", "*"))
        try:
            matcher = re.compile(pattern, re.IGNORECASE)
        except re.error:
            matcher = None
        out = []
        glob_variants = [glob]
        for prefix in ("code/", "src/"):
            if glob.startswith(prefix):
                glob_variants.append(glob[len(prefix):] or "*")
        for item in fs.iter_files(suffixes=_TEXT_SUFFIXES):
            relative = item.rel
            # Tool callers use repository-relative globs, while the old
            # adapter matched only the basename.  Support both forms and
            # treat ``**/`` as recursive without exposing paths outside the
            # bounded source root.
            path_globs = [item.replace("**/", "") for item in glob_variants]
            if not (
                any(fnmatch.fnmatch(Path(relative).name, item) for item in glob_variants)
                or any(fnmatch.fnmatch(relative, item) for item in glob_variants)
                or any(fnmatch.fnmatch(relative, item) for item in path_globs)
            ):
                continue
            try: lines = fs.read_text(relative).splitlines()
            except (OSError, ValueError, RuntimeError): continue
            for number, line in enumerate(lines, 1):
                literal_match = pattern.casefold() in line.casefold()
                matched = (
                    bool(matcher.search(line)) or literal_match
                    if matcher is not None else literal_match
                )
                if matched: out.append({"path": relative, "line": number, "text": line[:500]})
        return out[:200]

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path, action = self._fs_path(args["path"])
        if not path.is_file():
            raise FileNotFoundError(args["path"])
        start, count = int(args.get("start_line", 1)), int(args.get("line_count", MAX_SOURCE_LINES))
        relative = action["normalized_path"]
        parts = Path(relative).parts
        if len(parts) < 2:
            raise SourceBindingError("official source acquisition requires an app-scoped path")
        return self._acquire_source(parts[0], "/".join(parts[1:]), start, count, action)

    def _get_source_code(self, args: dict[str, Any]) -> dict[str, Any]:
        app = str(args["app_name"])
        file = str(args["file_path"])
        if Path(app).name != app or Path(file).is_absolute() or ".." in Path(file).parts:
            raise SourceBindingError("source identity must be app-scoped and relative")
        return self._acquire_source(app, file, int(args.get("start_line", 1)),
                                    args.get("line_count"), {})

    def _acquire_source(self, app, file, start, count, action):
        path = self.binding.filesystem().resolve(f"{app}/{file}")
        app_root = self.binding.source_root / app
        if not app_root.resolve().is_relative_to(self.binding.source_root) or not path.is_relative_to(app_root):
            raise SourceBindingError("source path escapes selected app")
        if not path.is_file():
            raise FileNotFoundError(file)
        result, acquired, provenance = self.official.invoke(
            "GetSourceCode", {"app_name": app, "file_path": file},
        )
        if not acquired or not official_result_success(result):
            raise SourceBindingError(str(result))
        lines = result.splitlines()
        selected = lines[start - 1:] if count is None else lines[start - 1:start - 1 + count]
        relative = f"{app}/{file}"
        end = start + len(selected) - 1
        content = result if count is None and start == 1 else "\n".join(f"{start + i}: {line}" for i, line in enumerate(selected))
        return {"content": content, "metadata": {**provenance, "path": relative,
                "start_line": start, "end_line": end, "acquired_start_line": 1,
                "acquired_end_line": len(lines), "requested_line_count": count,
                "retained_start_line": start, "retained_end_line": end,
                "source_body_format": "plain" if count is None and start == 1 else "numbered_colon",
                "normalization_action": action}}

    def _symbols(self, file: Path) -> list[dict[str, Any]]:
        try: tree = ast.parse(self.binding.filesystem().read_text(file))
        except (OSError, ValueError, RuntimeError, SyntaxError): return []
        rows = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rows.append({"symbol": node.name, "kind": type(node).__name__, "source_range": [node.lineno, getattr(node, "end_lineno", node.lineno)]})
        return rows

    def _symbol_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(args["query"]).casefold(); out = []
        for item in self.binding.filesystem().iter_files(suffixes={".py"}):
            for row in self._symbols(Path(item.rel)):
                if query in row["symbol"].casefold(): out.append({"file": item.rel, **row})
        return out[:100]

    def _code_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"]).casefold(); fs = self.binding.filesystem(); results = []
        for item in fs.iter_files(suffixes=_TEXT_SUFFIXES):
            try: text = fs.read_text(item.rel)
            except (OSError, ValueError, RuntimeError): continue
            if query not in text.casefold(): continue
            matches = [row for row in self._symbols(Path(item.rel)) if query in row["symbol"].casefold()]
            results.append({"path": item.rel, "matched_symbols": matches})
        return {"results": results[:100]}

    def _inspect_symbol(self, args: dict[str, Any]) -> dict[str, Any]:
        path, _ = self._fs_path(args["file"]); symbol = str(args["symbol"])
        rows = [row for row in self._symbols(path) if row["symbol"] == symbol]
        return {"ok": bool(rows), "file": str(path.relative_to(self.binding.source_root)).replace("\\", "/"), "symbol": symbol, "matches": rows[:1]}
