from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from debug_assistant.datasets.ground_truth import normalize_repo_path
from debug_assistant.evaluation.localization import _gold_locations
from debug_assistant.repository.chunks import build_chunk_manifest
from debug_assistant.repository.embeddings import (
    EmbeddingCache,
    SiliconFlowEmbeddingProvider,
)
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.search_engine import (
    RepositorySearchEngine,
    RetrievalDiagnostics,
    reciprocal_rank_fusion,
)


RETRIEVAL_MODES = ("lexical", "semantic", "hybrid", "hybrid_ast")


def _norm(path: object) -> str:
    try:
        return normalize_repo_path(path) or ""
    except (TypeError, ValueError):
        return ""


def _fixed_manifest(root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the checked-in fixed subset, rejecting accidental resampling."""
    manifest = root.parent / "swe_lite_retrieval_subset_20.json"
    if root.name != "swe_lite_dev" or not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ids = data.get("instance_ids")
    if not isinstance(ids, list) or len(ids) != 20:
        raise ValueError("fixed SWE-lite retrieval manifest must contain exactly 20 instance_ids")
    if len({str(value) for value in ids}) != 20:
        raise ValueError("fixed SWE-lite retrieval manifest contains duplicate instance_ids")
    return manifest, data


def _resolve_workspace(root: Path, value: object) -> Path:
    workspace = Path(str(value))
    if workspace.is_absolute():
        return workspace
    return (root.parent.parent / workspace).resolve()


def _git_head(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _gold_files(gold: dict[str, Any]) -> set[str]:
    return {
        path
        for location in _gold_locations(gold)
        for path in (_norm(location.get("old_path")), _norm(location.get("new_path")))
        if path
    }


def _unique_paths(paths: list[str]) -> list[str]:
    """Convert chunk-ranked results to file-ranked results for file metrics."""
    return list(dict.fromkeys(path for path in paths if path))


def _metric_values(paths: list[str], gold_files: set[str], top_k: int) -> dict[str, Any]:
    rank = next((index + 1 for index, path in enumerate(paths) if path in gold_files), None)
    return {
        "first_relevant_rank": rank,
        # A miss is a valid zero for an available retrieval path. ``None`` is
        # reserved for UNAVAILABLE/FAILED rows so aggregation keeps the full
        # available-case denominator.
        "file_hit_at_1": int(rank is not None and rank <= 1),
        "file_hit_at_3": int(rank is not None and rank <= 3),
        "recall_at_10": int(rank is not None and rank <= top_k),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def _unavailable_row(
    *,
    task_id: str,
    mode: str,
    gold_files: set[str],
    repo: object,
    base_commit: object,
    workspace: Path,
    availability: str,
    reason: str,
    semantic_available: bool,
    ast_available: bool,
    ast_status: str,
) -> dict[str, Any]:
    return {
        "instance_id": task_id,
        "task_id": task_id,
        "repo": str(repo or ""),
        "base_commit": str(base_commit or ""),
        "workspace": str(workspace),
        "mode": mode,
        "availability": availability,
        "failure_reason": reason,
        "gold_files": sorted(gold_files),
        "top10_predicted_files": [],
        "paths": [],
        "first_relevant_rank": None,
        "file_hit_at_1": None,
        "file_hit_at_3": None,
        "recall_at_10": None,
        "mrr": None,
        "latency_ms": None,
        "semantic_available": semantic_available,
        "ast_available": ast_available,
        "ast_status": ast_status,
        "diagnostics": {},
    }


def _refine_with_ast(
    index: RepositoryIndex,
    rows: list[dict[str, Any]],
    query: str,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapter for the repository-owned, candidate-contained AST refinement."""
    return index.refine_hybrid_candidates(rows, query, limit=limit)


def _provider_preflight(semcfg: Any) -> tuple[dict[str, Any], SiliconFlowEmbeddingProvider | None]:
    result: dict[str, Any] = {
        "provider": str(getattr(semcfg, "provider", "")),
        "base_url": str(getattr(semcfg, "base_url", "")),
        "model": str(getattr(semcfg, "model", "")),
        "configured_dimension": int(getattr(semcfg, "dimension", 0) or 0),
        "configured_timeout_seconds": float(getattr(semcfg, "timeout", 0) or 0),
        "configured_max_retries": int(getattr(semcfg, "max_retries", 0) or 0),
        "configured_batch_size": int(getattr(semcfg, "batch_size", 0) or 0),
        "status": "UNAVAILABLE",
        "error_type": "",
        "error": "",
    }
    if not getattr(semcfg, "enabled", False):
        result["error_type"] = "semantic_search_disabled"
        result["error"] = "semantic search is disabled"
        return result, None

    if not getattr(semcfg, "api_key", ""):
        result["error_type"] = "embedding_api_key_missing"
        result["error"] = "embedding API key is missing"
        return result, None
    if str(getattr(semcfg, "provider", "")).lower() != "siliconflow":
        result["error_type"] = "unsupported_embedding_provider"
        result["error"] = "retrieval evaluator currently reuses SiliconFlowEmbeddingProvider"
        return result, None

    try:
        provider = SiliconFlowEmbeddingProvider(
            api_key=semcfg.api_key,
            model=semcfg.model,
            base_url=semcfg.base_url,
            dimension=semcfg.dimension,
            timeout=semcfg.timeout,
            batch_size=semcfg.batch_size,
            max_retries=semcfg.max_retries,
            max_isolation_depth=semcfg.max_isolation_depth,
        )
        vector = provider.smoke_test()
        result.update(
            {
                "status": "AVAILABLE",
                "returned_dimension": len(vector),
                "requests": provider.stats.requests,
                "failures": provider.stats.failures,
            }
        )
        if len(vector) != provider.dimension:
            result["status"] = "UNAVAILABLE"
            result["error_type"] = "embedding_dimension_mismatch"
            result["error"] = "provider returned an unexpected vector dimension"
            return result, None
        return result, provider
    except Exception as exc:
        result.update(
            {
                "status": "UNAVAILABLE",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return result, None


def _semantic_search_with_vector(semantic_index: Any, vector: list[float], limit: int) -> list[dict[str, Any]]:
    """Search an already-built SemanticIndex without a second provider round trip."""
    if not semantic_index.available:
        return []
    query = np.asarray(vector, dtype=np.float32)
    if query.shape != (semantic_index.provider.dimension,):
        raise ValueError("query embedding dimension mismatch")
    norm = float(np.linalg.norm(query))
    query = query / (norm or 1.0)
    nlimit = max(1, int(limit))
    if semantic_index.faiss_index is not None:
        values, indices = semantic_index.faiss_index.search(query.reshape(1, -1), nlimit)
        pairs = [
            (int(index), float(value))
            for index, value in zip(indices[0], values[0])
            if int(index) >= 0
        ]
    else:
        scores = semantic_index.matrix @ query
        indices = np.argsort(-scores)[:nlimit]
        pairs = [(int(index), float(scores[int(index)])) for index in indices]
    result = []
    for index, score in pairs:
        chunk = semantic_index.chunks[index]
        result.append(
            {
                "chunk_id": chunk.chunk_id,
                "path": chunk.path,
                "symbol": chunk.qualified_name or chunk.symbol,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "snippet": chunk.content[:1200],
                "score": score,
                "source": "semantic",
                "backend": semantic_index.backend,
            }
        )
    return result


def _aggregate(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    available = [row for row in rows if row.get("availability") == "AVAILABLE"]
    values: dict[str, Any] = {}
    for field in ("file_hit_at_1", "file_hit_at_3", "recall_at_10", "mrr", "latency_ms"):
        field_values = [row[field] for row in available if row.get(field) is not None]
        values[field] = sum(field_values) / len(field_values) if field_values else None
    return {
        "n": len(rows),
        "available_n": len(available),
        "unavailable_n": sum(row.get("availability") == "UNAVAILABLE" for row in rows),
        "failed_n": sum(row.get("availability") == "FAILED" for row in rows),
        "file_hit@1": values["file_hit_at_1"],
        "file_hit@3": values["file_hit_at_3"],
        f"file_recall@{top_k}": values["recall_at_10"],
        "mrr": values["mrr"],
        "avg_latency_ms": values["latency_ms"],
    }


def _deltas(
    rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    top_k: int,
    expected_case_count: int,
) -> dict[str, Any]:
    by_mode = {
        mode: {row["instance_id"]: row for row in rows if row.get("mode") == mode}
        for mode in modes
    }
    result: dict[str, Any] = {}
    recall_field = f"file_recall@{top_k}"
    for mode in ("semantic", "hybrid", "hybrid_ast"):
        paired_ids = sorted(
            instance_id
            for instance_id, row in by_mode.get("lexical", {}).items()
            if row.get("availability") == "AVAILABLE"
            and by_mode.get(mode, {}).get(instance_id, {}).get("availability") == "AVAILABLE"
        )
        baseline = _aggregate(
            [by_mode["lexical"][instance_id] for instance_id in paired_ids], top_k
        )
        current = _aggregate(
            [by_mode[mode][instance_id] for instance_id in paired_ids], top_k
        )
        result[mode] = {
            "comparison_scope": "paired_available_cases",
            "paired_n": len(paired_ids),
            "full_manifest_comparable": len(paired_ids) == expected_case_count,
            "deltas": {},
        }
        for field in ("file_hit@1", "file_hit@3", "mrr", recall_field, "avg_latency_ms"):
            old, new = baseline.get(field), current.get(field)
            result[mode]["deltas"][field] = None if old is None or new is None else new - old
    return result


def _available_row(
    *,
    task_id: str,
    mode: str,
    meta: dict[str, Any],
    workspace: Path,
    gold_files: set[str],
    paths: list[str],
    latency_ms: float | None,
    semantic_available: bool,
    ast_available: bool,
    ast_status: str,
    diagnostics: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    return {
        "instance_id": task_id,
        "task_id": task_id,
        "repo": str(meta.get("repo", "")),
        "base_commit": str(meta.get("base_commit", "")),
        "workspace": str(workspace),
        "mode": mode,
        "availability": "AVAILABLE",
        "failure_reason": "",
        "gold_files": sorted(gold_files),
        "top10_predicted_files": paths[:top_k],
        "paths": paths[:top_k],
        **_metric_values(paths[:top_k], gold_files, top_k),
        "latency_ms": latency_ms,
        "semantic_available": semantic_available,
        "ast_available": ast_available,
        "ast_status": ast_status,
        "diagnostics": diagnostics,
    }


def evaluate_retrieval(
    tasks_root,
    config,
    modes=RETRIEVAL_MODES,
    top_k=10,
):
    root = Path(tasks_root).resolve()
    requested_modes = tuple(str(mode).lower() for mode in modes)
    unknown = set(requested_modes) - set(RETRIEVAL_MODES)
    if unknown:
        raise ValueError(f"unsupported retrieval modes: {sorted(unknown)}")

    manifest_info = _fixed_manifest(root)
    manifest_path, manifest_data = manifest_info if manifest_info else (None, None)
    if manifest_data:
        task_ids = [str(value) for value in manifest_data["instance_ids"]]
        task_dirs = [root / task_id for task_id in task_ids]
        missing = [str(path) for path in task_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"fixed retrieval manifest task directories missing: {missing}")
    else:
        task_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "task.json").exists() and (path / "ground_truth.json").exists()
        )

    rows: list[dict[str, Any]] = []
    semcfg = config.harness.semantic_search
    preflight, provider = _provider_preflight(semcfg)
    cache: EmbeddingCache | None = None
    if provider is not None:
        cache_path = Path(semcfg.cache_path)
        if not cache_path.is_absolute():
            cache_path = root.parent.parent / cache_path
        cache = EmbeddingCache(cache_path)

    # Query text is runtime-visible issue text only. Batch the one query vector
    # per case so the four modes compare identical semantic inputs without
    # multiplying external Provider latency by three.
    query_vectors: dict[str, list[float]] = {}
    query_errors: dict[str, str] = {}
    query_batch_latency_ms: float | None = None
    if provider is not None:
        query_started = time.monotonic()
        # A single input per request avoids large-payload stalls while preserving
        # one real query vector per Case. Failures stay Case-local so one malformed
        # issue cannot erase valid results for the remaining fixed subset.
        for task_dir in task_dirs:
            task_id = task_dir.name
            try:
                issue = (task_dir / "issue.md").read_text(encoding="utf-8")
                vectors = provider.embed_documents([issue])
                if len(vectors) != 1:
                    raise ValueError("query embedding count mismatch")
                query_vectors[task_id] = vectors[0]
            except Exception as exc:
                query_errors[task_id] = f"semantic_query_failed:{type(exc).__name__}"
        query_batch_latency_ms = (time.monotonic() - query_started) * 1000
        preflight["query_batch"] = {
            "status": "AVAILABLE" if not query_errors else "PARTIAL",
            "count": len(query_vectors),
            "failed_count": len(query_errors),
            "latency_ms": query_batch_latency_ms,
            "error_type": "" if not query_errors else "case_local_query_failures",
        }

    try:
        for task_dir in task_dirs:
            task_id = task_dir.name
            task_file = task_dir / "task.json"
            issue_file = task_dir / "issue.md"
            gold_file = task_dir / "ground_truth.json"
            meta = json.loads(task_file.read_text(encoding="utf-8"))
            workspace = _resolve_workspace(root, meta.get("workspace"))
            issue = issue_file.read_text(encoding="utf-8")
            gold = json.loads(gold_file.read_text(encoding="utf-8"))
            gold_files = _gold_files(gold)
            snapshot_head = _git_head(workspace)
            snapshot_match = bool(
                meta.get("base_commit") and snapshot_head and snapshot_head == meta.get("base_commit")
            )

            index: RepositoryIndex | None = None
            try:
                fs = SafeRepositoryFS(workspace)
                with tempfile.TemporaryDirectory(prefix="debug-retrieval-") as temp_dir:
                    index = RepositoryIndex(workspace, Path(temp_dir) / "lexical.sqlite", fs=fs)
                    index.build()
                    engine = RepositorySearchEngine(index, None, rrf_k=semcfg.rrf_k)

                    lex_started = time.monotonic()
                    lexical_rows, lexical_diag = engine.search(
                        issue, mode="lexical", limit=max(top_k, 20)
                    )
                    lexical_latency = (time.monotonic() - lex_started) * 1000
                    lexical_paths = _unique_paths([_norm(row.get("path")) for row in lexical_rows])
                    lexical_diagnostics = asdict(lexical_diag)
                    lexical_diagnostics["snapshot_match"] = snapshot_match
                    rows.append(
                        _available_row(
                            task_id=task_id,
                            mode="lexical",
                            meta=meta,
                            workspace=workspace,
                            gold_files=gold_files,
                            paths=lexical_paths,
                            latency_ms=lexical_latency,
                            semantic_available=False,
                            ast_available=False,
                            ast_status="not_requested",
                            diagnostics=lexical_diagnostics,
                            top_k=top_k,
                        )
                    )

                    semantic_available = False
                    semantic_rows: list[dict[str, Any]] = []
                    semantic_diag: dict[str, Any] = {}
                    semantic_failure = ""
                    semantic_latency: float | None = None
                    if provider is None:
                        semantic_failure = str(
                            preflight.get("error_type") or "semantic_preflight_unavailable"
                        )
                    elif task_id in query_errors:
                        semantic_failure = query_errors[task_id]
                    else:
                        from debug_assistant.repository.semantic_index import SemanticIndex

                        manifest = build_chunk_manifest(
                            fs,
                            max_embedding_tokens=semcfg.max_embedding_tokens,
                        )
                        semantic_index = SemanticIndex(manifest, provider, cache)
                        build_stats = semantic_index.build()
                        semantic_available = semantic_index.available
                        semantic_diag["build"] = asdict(build_stats)
                        if not semantic_available:
                            semantic_failure = "semantic_index_build_failed"
                        else:
                            sem_started = time.monotonic()
                            try:
                                semantic_rows = _semantic_search_with_vector(
                                    semantic_index,
                                    query_vectors[task_id],
                                    max(top_k, 20),
                                )
                                semantic_latency = (time.monotonic() - sem_started) * 1000
                                semantic_latency += (query_batch_latency_ms or 0.0) / max(1, len(task_dirs))
                                semantic_diag.update(
                                    asdict(
                                        RetrievalDiagnostics(
                                            "semantic",
                                            "semantic",
                                            len(semantic_rows),
                                            True,
                                            semantic_candidates=len(semantic_rows),
                                            rrf_k=semcfg.rrf_k,
                                        )
                                    )
                                )
                                semantic_diag["query_embedding_reused_for_evaluation"] = True
                                semantic_diag["query_embedding_batch_latency_ms"] = query_batch_latency_ms
                            except Exception as exc:
                                semantic_failure = f"semantic_query_failed:{type(exc).__name__}"
                                semantic_diag["error"] = str(exc)

                    if semantic_available and not semantic_failure:
                        semantic_paths = _unique_paths([_norm(row.get("path")) for row in semantic_rows])
                        rows.append(
                            _available_row(
                                task_id=task_id,
                                mode="semantic",
                                meta=meta,
                                workspace=workspace,
                                gold_files=gold_files,
                                paths=semantic_paths,
                                latency_ms=semantic_latency,
                                semantic_available=True,
                                ast_available=False,
                                ast_status="not_requested",
                                diagnostics=semantic_diag,
                                top_k=top_k,
                            )
                        )

                        hybrid_started = time.monotonic()
                        hybrid_rows = reciprocal_rank_fusion(
                            [lexical_rows, semantic_rows],
                            k=semcfg.rrf_k,
                            limit=max(top_k, 20),
                        )
                        hybrid_fusion_latency = (time.monotonic() - hybrid_started) * 1000
                        hybrid_latency = lexical_latency + (semantic_latency or 0.0) + hybrid_fusion_latency
                        hybrid_paths = _unique_paths([_norm(row.get("path")) for row in hybrid_rows])
                        hybrid_diag = {
                            "requested_mode": "hybrid",
                            "effective_mode": "hybrid",
                            "result_count": len(hybrid_rows),
                            "semantic_available": True,
                            "degraded": False,
                            "reason": "",
                            "lexical_candidates": len(lexical_rows),
                            "semantic_candidates": len(semantic_rows),
                            "rrf_k": semcfg.rrf_k,
                            "query_embedding_reused_for_evaluation": True,
                        }
                        rows.append(
                            _available_row(
                                task_id=task_id,
                                mode="hybrid",
                                meta=meta,
                                workspace=workspace,
                                gold_files=gold_files,
                                paths=hybrid_paths,
                                latency_ms=hybrid_latency,
                                semantic_available=True,
                                ast_available=False,
                                ast_status="not_requested",
                                diagnostics=hybrid_diag,
                                top_k=top_k,
                            )
                        )

                        ast_started = time.monotonic()
                        ast_rows, ast_refinement = _refine_with_ast(
                            index,
                            hybrid_rows,
                            issue,
                            max(top_k, 20),
                        )
                        ast_available = bool(ast_refinement.get("ast_available"))
                        ast_status = str(ast_refinement.get("ast_status") or "unavailable")
                        ast_index_latency = int(ast_refinement.get("elapsed_ms") or 0)
                        ast_latency = hybrid_latency + (time.monotonic() - ast_started) * 1000
                        ast_paths = _unique_paths([_norm(row.get("path")) for row in ast_rows])
                        ast_diag = dict(hybrid_diag)
                        ast_diag.update(
                            {
                                "requested_mode": "hybrid_ast",
                                "ast_symbol_candidates": sum(
                                    1
                                    for row in ast_rows
                                    if row.get("matched_symbols")
                                ),
                                "ast_matched_symbols": sum(
                                    len(row.get("matched_symbols") or []) for row in ast_rows
                                ),
                                "ast_relations_used": sum(
                                    int(row.get("caller_count") or 0)
                                    + int(row.get("callee_count") or 0)
                                    for row in ast_rows
                                ),
                                "ast_supported_candidates": int(
                                    ast_refinement.get("supported_candidate_count") or 0
                                ),
                                "ast_unsupported_candidates": int(
                                    ast_refinement.get("unsupported_candidate_count") or 0
                                ),
                                "ast_index_latency_ms": ast_index_latency,
                                "ast_status": ast_status,
                            }
                        )
                        rows.append(
                            _available_row(
                                task_id=task_id,
                                mode="hybrid_ast",
                                meta=meta,
                                workspace=workspace,
                                gold_files=gold_files,
                                paths=ast_paths,
                                latency_ms=ast_latency,
                                semantic_available=True,
                                ast_available=ast_available,
                                ast_status=ast_status,
                                diagnostics=ast_diag,
                                top_k=top_k,
                            )
                        )
                    else:
                        status = "UNAVAILABLE" if provider is None else "FAILED"
                        for mode in ("semantic", "hybrid", "hybrid_ast"):
                            rows.append(
                                _unavailable_row(
                                    task_id=task_id,
                                    mode=mode,
                                    gold_files=gold_files,
                                    repo=meta.get("repo"),
                                    base_commit=meta.get("base_commit"),
                                    workspace=workspace,
                                    availability=status,
                                    reason=semantic_failure,
                                    semantic_available=False,
                                    ast_available=False,
                                    ast_status="not_available",
                                )
                            )
            except Exception as exc:
                reason = f"retrieval_case_failed:{type(exc).__name__}"
                for mode in requested_modes:
                    if any(row.get("instance_id") == task_id and row.get("mode") == mode for row in rows):
                        continue
                    rows.append(
                        _unavailable_row(
                            task_id=task_id,
                            mode=mode,
                            gold_files=gold_files,
                            repo=meta.get("repo"),
                            base_commit=meta.get("base_commit"),
                            workspace=workspace,
                            availability="FAILED",
                            reason=reason,
                            semantic_available=False,
                            ast_available=False,
                            ast_status="not_available",
                        )
                    )
            finally:
                if index is not None:
                    index.close()
    finally:
        if cache is not None:
            cache.close()

    aggregate = {
        mode: _aggregate([row for row in rows if row.get("mode") == mode], top_k)
        for mode in requested_modes
    }
    return {
        "metadata": {
            "manifest": str(manifest_path) if manifest_path else None,
            "seed": manifest_data.get("seed") if manifest_data else None,
            "selection": manifest_data.get("selection") if manifest_data else "directory_order",
            "instance_ids": [row.name for row in task_dirs],
            "instance_count": len(task_dirs),
            "query_source": "issue.md runtime-visible issue text only",
            "ground_truth_used_for": "final file metrics only",
            "top_k": top_k,
            "modes": list(requested_modes),
            "embedding_preflight": preflight,
            "fairness": {
                "same_manifest": bool(manifest_data),
                "same_query_source": True,
                "same_top_k": True,
                "base_snapshot_checked": True,
            },
        },
        "aggregate": aggregate,
        "deltas_vs_lexical": _deltas(
            rows,
            requested_modes,
            top_k,
            len(task_dirs),
        ),
        "status_counts": {
            mode: {
                status: sum(
                    row.get("availability") == status
                    for row in rows
                    if row.get("mode") == mode
                )
                for status in ("AVAILABLE", "UNAVAILABLE", "FAILED")
            }
            for mode in requested_modes
        },
        "cases": [row for row in rows if row.get("mode") in requested_modes],
    }
