from __future__ import annotations

"""Read-only binding of a CloudOps case-local application source tree.

The official incident cache is still owned by ``cloudops_snapshot``.  This module
only binds the optional ``<case>/code`` directory to the already existing
repository tools.  It deliberately does not inspect evaluator data, choose a
diagnosis, or add another runtime lifecycle.
"""

import re
import json
import posixpath
from pathlib import Path
from typing import Any

from debug_assistant.models import ToolObservation
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.repository.search_engine import RepositorySearchEngine
from debug_assistant.tools.base import Tool
from debug_assistant.tools.registry import ToolRegistry


class SourceBindingError(ValueError):
    """The case source directory exists but cannot be safely bound."""


class CloudOpsSourceBinding:
    """Resolve the optional case-local ``code`` directory once, read-only."""

    relative_root = "code"

    def __init__(self, runtime_data_dir: str | Path):
        self.case_root = Path(runtime_data_dir).resolve()
        source_candidate = self.case_root / self.relative_root
        if not source_candidate.exists():
            self.source_root: Path | None = None
            self.available = False
            return
        try:
            source_root = source_candidate.resolve(strict=True)
        except OSError as exc:
            raise SourceBindingError("application source root cannot be resolved") from exc
        if not source_root.is_dir():
            raise SourceBindingError("application source root is not a directory")
        if source_root != self.case_root and self.case_root not in source_root.parents:
            raise SourceBindingError("application source root escapes the incident case")
        self.source_root = source_root
        self.available = True

    def filesystem(self) -> SafeRepositoryFS:
        if not self.available or self.source_root is None:
            raise SourceBindingError("application source is not available for this case")
        return SafeRepositoryFS(self.source_root)


_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/]|//|\\\\)")
_PATH_ARGUMENTS = {"repo_tree": "path", "grep": "glob", "read_file": "path"}
_DISCOVERY_TOOLS = frozenset({"repo_tree", "grep", "symbol_search", "code_search", "inspect_symbol_context"})
_PATH_NORMALIZATION_RULES = frozenset({
    "stripped_bound_root_name",
    "abs_to_rel",
    "already_relative",
    "rejected_out_of_root",
})


def _path_contract_result(
    *, original_path: Any, normalized_path: str, rule_applied: str, result: str,
) -> dict[str, str]:
    if rule_applied not in _PATH_NORMALIZATION_RULES:
        raise ValueError(f"unknown path normalization rule: {rule_applied}")
    return {
        "original_path": str(original_path),
        "normalized_path": normalized_path,
        "rule_applied": rule_applied,
        "result": result,
    }


def _normalize_bound_path(
    raw: Any, *, source_root: Path, bound_root_name: str,
) -> tuple[str | None, dict[str, str], str | None]:
    """Normalize one source-tool path against the already-bound source root."""
    original = str(raw)
    value = original.strip().strip("'\"`").replace("\\", "/")
    if not value:
        value = "."
    raw_parts = [part for part in value.split("/") if part not in {"", "."}]
    if ".." in raw_parts:
        action = _path_contract_result(
            original_path=original,
            normalized_path=value,
            rule_applied="rejected_out_of_root",
            result="rejected",
        )
        return None, action, "source path escapes the bound code root"
    value = posixpath.normpath(re.sub(r"/{2,}", "/", value))
    while value.startswith("./"):
        value = value[2:]
    value = value or "."

    is_absolute = bool(_ABSOLUTE_PATH.match(value)) or Path(value).is_absolute()
    if is_absolute:
        try:
            candidate = Path(value).resolve(strict=False)
            relative = candidate.relative_to(source_root)
        except (OSError, ValueError):
            action = _path_contract_result(
                original_path=original,
                normalized_path="",
                rule_applied="rejected_out_of_root",
                result="rejected",
            )
            return None, action, "absolute source path is outside the bound code root"
        normalized = relative.as_posix() or "."
        action = _path_contract_result(
            original_path=original,
            normalized_path=normalized,
            rule_applied="abs_to_rel",
            result="accepted",
        )
        return normalized, action, None

    # Strip the bound root name exactly once, only at a segment boundary.
    if value == bound_root_name:
        normalized = "."
        rule = "stripped_bound_root_name"
    elif value.startswith(bound_root_name + "/"):
        normalized = value[len(bound_root_name) + 1:] or "."
        rule = "stripped_bound_root_name"
    else:
        normalized = value
        rule = "already_relative"

    action = _path_contract_result(
        original_path=original,
        normalized_path=normalized,
        rule_applied=rule,
        result="accepted",
    )
    return normalized, action, None


class _BoundRepositoryTool(Tool):
    """Apply the Runtime-owned bound-source path contract before execution.

    Repository primitives receive paths relative to ``binding.source_root``;
    this adapter alone interprets the display name ``code`` and emits the
    normalization metadata used by the Incident trace.
    """

    def __init__(self, inner: Tool, *, binding: CloudOpsSourceBinding, symbol_index=None):
        self.inner = inner
        self.spec = inner.spec
        self.binding = binding
        self.symbol_index = symbol_index

    def execute(self, **kwargs: Any) -> ToolObservation:
        argument_name = _PATH_ARGUMENTS.get(self.spec.name)
        normalized_kwargs = dict(kwargs)
        normalization_action = None
        default_path = {
            "repo_tree": ".",
            "grep": "*",
        }.get(self.spec.name)
        if argument_name and (argument_name in kwargs or default_path is not None):
            normalized, normalization_action, reason = _normalize_bound_path(
                kwargs.get(argument_name, default_path),
                source_root=self.binding.source_root or self.binding.case_root,
                bound_root_name=Path(self.binding.relative_root).name,
            )
            if reason:
                return ToolObservation(
                    tool=self.spec.name,
                    ok=False,
                    content=reason,
                    metadata={
                        "arguments": dict(kwargs),
                        "source_scope": "code",
                        "source_relative_root": "code",
                        "normalization_action": normalization_action,
                        "retryable": False,
                        "planner_retryable": False,
                    },
                    error_type="path_contract_error",
                )
            normalized_kwargs[argument_name] = normalized
        observation = self.inner.execute(**normalized_kwargs)
        metadata = dict(observation.metadata or {})
        if normalization_action is not None:
            metadata["normalization_action"] = normalization_action
        metadata.update({
            "source_scope": "code",
            "source_relative_root": "code",
            "read_only": True,
        })
        if self.spec.name == "read_file":
            metadata.update({
                "context_kind": "CODE",
                "information_source": "source_read",
            })
            if metadata.get("path") and metadata.get("start_line") is not None and metadata.get("end_line") is not None:
                metadata.setdefault("provenance", {
                    "source_root": "code",
                    "path": metadata["path"],
                    "start_line": metadata["start_line"],
                    "end_line": metadata["end_line"],
                    "content_origin": "SafeRepositoryFS",
                })
        elif self.spec.name in _DISCOVERY_TOOLS:
            metadata.update({
                "context_kind": "CODE_SEARCH" if self.spec.name != "repo_tree" else "CODE_INDEX",
                "information_source": "candidate_retrieval",
            })
            if self.spec.name == "symbol_search" and observation.ok:
                # Enrich Python candidates with existing call relations, but do
                # not promote a symbol preview to source evidence.  The source
                # boundary is an explicit read_file observation.
                try:
                    payload = json.loads(observation.content)
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                matches = payload.get("matches") if isinstance(payload, dict) else None
                if isinstance(matches, list) and self.symbol_index is not None:
                    # The Go/JS/etc. fallback search has no call graph, but keep
                    # the existing AST relation capability for Python matches.
                    for match in matches:
                        if not isinstance(match, dict) or not str(match.get("file") or "").endswith(".py"):
                            continue
                        try:
                            relation = self.symbol_index.inspect_symbol_context(
                                str(match.get("symbol") or match.get("name") or ""),
                                str(match.get("file")), include_source=False,
                            )
                        except Exception:
                            relation = {}
                        if relation.get("ok"):
                            match["callers"] = relation.get("callers") or []
                            match["callees"] = relation.get("callees") or []
                            match["relations_available"] = bool(match["callers"] or match["callees"])
                    observation.content = json.dumps(payload, ensure_ascii=False)
        if metadata.get("information_source") == "source_read":
            provenance = dict(metadata.get("provenance") or {})
            provenance.setdefault("observation_id", observation.observation_id)
            metadata["provenance"] = provenance
        observation.metadata = metadata
        return observation


class CloudOpsSourceToolRegistry:
    """Expose the safe Incident code-investigation subset of repository tools."""

    # Keep the underlying repository primitives available for shared/SWE users,
    # while the incident Planner receives the smaller surface below through the
    # snapshot registry's planner-visible schema projection.
    _EXPOSED = ("repo_tree", "grep", "read_file", "symbol_search", "code_search", "inspect_symbol_context")

    def __init__(self, binding: CloudOpsSourceBinding, *, search_engine=None):
        if not binding.available or binding.source_root is None:
            raise SourceBindingError("application source is not available for this case")
        fs = binding.filesystem()
        self.index = None
        if search_engine is None:
            # SQLite :memory: keeps the adapter task-scoped and avoids creating
            # repository artifacts in a case snapshot. Semantic search is an
            # optional injected capability; lexical BM25 is always available.
            self.index = RepositoryIndex(binding.source_root, Path(":memory:"), fs=fs)
            self.index.build()
            search_engine = RepositorySearchEngine(self.index, None)
        elif hasattr(search_engine, "lexical"):
            self.index = search_engine.lexical
        # Keep the existing text/symbol tool behavior (including lightweight
        # non-Python declaration lookup) while adding indexed code_search. The
        # shared indexed registry is used only for the new adapter tools.
        repository = ToolRegistry(binding.source_root, fs=fs)
        indexed_repository = ToolRegistry(
            binding.source_root,
            index=search_engine,
            fs=fs,
            # Incident Planner sees one high-level code-search operation.  The
            # bound tool selects hybrid_ast internally; the provider does not
            # control BM25/embedding/RRF/AST policy knobs.
            code_search_default_mode="hybrid_ast",
        )
        self._tools = {
            name: _BoundRepositoryTool(
                indexed_repository.get(name) if name in {"code_search", "inspect_symbol_context"}
                else repository.get(name),
                binding=binding,
                symbol_index=self.index,
            )
            for name in self._EXPOSED
            if (indexed_repository.get(name) if name in {"code_search", "inspect_symbol_context"}
                else repository.get(name)) is not None
        }
        self.search_engine = search_engine

    def tools(self) -> dict[str, Tool]:
        return dict(self._tools)
