from __future__ import annotations

"""Read-only binding of a CloudOps case-local application source tree.

The official incident cache is still owned by ``cloudops_snapshot``.  This module
only binds the optional ``<case>/code`` directory to the already existing
repository tools.  It deliberately does not inspect evaluator data, choose a
diagnosis, or add another runtime lifecycle.
"""

import re
import json
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


def _reject_model_path(raw: Any) -> str | None:
    """Reject absolute and traversal spellings before repository resolution."""
    value = str(raw or "").strip().strip("'\"`").replace("\\", "/")
    if not value:
        return None
    if _ABSOLUTE_PATH.match(value):
        return "absolute source paths are not accepted; use a path relative to code/"
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if ".." in parts:
        return "source path traversal is not accepted"
    return None


class _BoundRepositoryTool(Tool):
    """Add CloudOps source scope and provenance to an existing repository tool."""

    def __init__(self, inner: Tool, *, symbol_index=None):
        self.inner = inner
        self.spec = inner.spec
        self.symbol_index = symbol_index

    def execute(self, **kwargs: Any) -> ToolObservation:
        argument_name = _PATH_ARGUMENTS.get(self.spec.name)
        if argument_name:
            reason = _reject_model_path(kwargs.get(argument_name))
            if reason:
                return ToolObservation(
                    tool=self.spec.name,
                    ok=False,
                    content=reason,
                    metadata={
                        "arguments": dict(kwargs),
                        "source_scope": "code",
                        "source_relative_root": "code",
                        "retryable": False,
                        "planner_retryable": False,
                    },
                    error_type="path_rejected",
                )
        observation = self.inner.execute(**kwargs)
        metadata = dict(observation.metadata or {})
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
                symbol_index=self.index,
            )
            for name in self._EXPOSED
            if (indexed_repository.get(name) if name in {"code_search", "inspect_symbol_context"}
                else repository.get(name)) is not None
        }
        self.search_engine = search_engine

    def tools(self) -> dict[str, Tool]:
        return dict(self._tools)
