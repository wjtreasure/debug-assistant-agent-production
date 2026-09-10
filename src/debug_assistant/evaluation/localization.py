from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from debug_assistant.datasets.ground_truth import (
    GROUND_TRUTH_SCHEMA_VERSION,
    normalize_repo_path,
    validate_ground_truth,
)


def _norm(path: str | None) -> str:
    """Return a safe repository-relative spelling for prediction comparison."""
    try:
        value = normalize_repo_path(path)
    except (TypeError, ValueError):
        return ""
    return value or ""


def _legacy_to_v2(gold: dict[str, Any]) -> dict[str, Any]:
    """Read pre-v2 gold without treating its hunk ranges as edit ranges."""
    if gold.get("schema_version") == GROUND_TRUTH_SCHEMA_VERSION:
        validate_ground_truth(gold)
        return gold
    locations = []
    for row in gold.get("patch_ranges") or []:
        if not isinstance(row, dict):
            continue
        old_path = _norm(row.get("old_path") or row.get("path")) or None
        new_path = _norm(row.get("path") or row.get("new_path")) or None
        hunk_ranges = []
        for hunk in row.get("modified_ranges") or []:
            if not isinstance(hunk, dict):
                continue
            hunk_ranges.append({
                "old": {"start_line": hunk.get("old_start"), "end_line": hunk.get("old_end")},
                "new": {"start_line": hunk.get("new_start"), "end_line": hunk.get("new_end")},
            })
        symbols = []
        for symbol in gold.get("symbols") or []:
            if isinstance(symbol, dict) and _norm(symbol.get("file")) == old_path:
                symbols.append({
                    "symbol": symbol.get("symbol", ""),
                    "kind": symbol.get("kind", ""),
                    "start_line": symbol.get("start_line", symbol.get("line_start")),
                    "end_line": symbol.get("end_line", symbol.get("line_end")),
                })
        locations.append({
            "old_path": old_path,
            "new_path": new_path,
            "status": "renamed" if old_path != new_path else "modified",
            "old_edit_ranges": [],
            "new_edit_ranges": [],
            "insertion_anchors": [],
            "hunk_ranges": hunk_ranges,
            "symbols": symbols,
            "new_symbols": [],
        })
    if not locations:
        legacy_symbols = [
            {
                "symbol": symbol.get("symbol", ""),
                "qualified_symbol": symbol.get("qualified_symbol", symbol.get("symbol", "")),
                "start_line": symbol.get("start_line", symbol.get("line_start")),
                "end_line": symbol.get("end_line", symbol.get("line_end")),
            }
            for symbol in gold.get("symbols") or []
            if isinstance(symbol, dict) and symbol.get("symbol")
        ]
        for path in gold.get("files") or []:
            locations.append({
                "old_path": _norm(path) or None,
                "new_path": _norm(path) or None,
                "status": "modified",
                "old_edit_ranges": [], "new_edit_ranges": [],
                "insertion_anchors": [], "hunk_ranges": [],
                "symbols": legacy_symbols, "new_symbols": [],
            })
    return {"schema_version": 2, "instance_id": gold.get("instance_id", "legacy"), "fix_locations": locations}


def _gold_locations(gold: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_legacy_to_v2(gold).get("fix_locations") or [])


def _location_paths(location: dict[str, Any]) -> set[str]:
    return {
        value for value in (_norm(location.get("old_path")), _norm(location.get("new_path")))
        if value
    }


def _predicted_points(pred: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for point in pred.get("recommended_change_points") or []:
        if not isinstance(point, dict):
            continue
        symbol = point.get("symbol")
        points.append({
            "file": _norm(point.get("file")),
            "symbol": str(symbol).strip() if symbol is not None else None,
            "line_start": point.get("line_start"),
            "line_end": point.get("line_end"),
        })
    return points


def _predicted_symbol_names(pred: dict[str, Any]) -> list[str]:
    names = [str(x).strip() for x in pred.get("likely_symbols") or [] if str(x).strip()]
    names.extend(x["symbol"] for x in _predicted_points(pred) if x.get("symbol"))
    return names


def _gold_symbols(locations: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result = []
    for location in locations:
        for symbol in location.get("symbols") or []:
            if isinstance(symbol, dict) and symbol.get("symbol"):
                result.append((location, symbol))
    return result


def _symbol_matches(point: dict[str, Any], location: dict[str, Any], symbol: dict[str, Any]) -> bool:
    if not point.get("file") or point["file"] not in _location_paths(location):
        return False
    predicted = point.get("symbol") or ""
    accepted = {str(symbol.get("symbol") or "")}
    if symbol.get("qualified_symbol"):
        accepted.add(str(symbol["qualified_symbol"]))
    return predicted in accepted


def _range_distance(start: int, end: int, gold_start: int, gold_end: int) -> int:
    if start <= gold_end and end >= gold_start:
        return 0
    return gold_start - end if end < gold_start else start - gold_end


def _point_range_distance(point: dict[str, Any], location: dict[str, Any]) -> int | None:
    start, end = point.get("line_start"), point.get("line_end")
    if start is None or end is None:
        return None
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if start < 1 or end < start:
        return None
    if point.get("file") not in _location_paths(location):
        return None
    distances = []
    for row in location.get("old_edit_ranges") or []:
        distances.append(_range_distance(start, end, int(row["start_line"]), int(row["end_line"])))
    for anchor in location.get("insertion_anchors") or []:
        # ChangePoint coordinates are 1-based. Represent an insertion before the
        # first line (old-side anchor 0) by line 1 for prediction matching.
        anchor_line = max(1, int(anchor))
        distances.append(0 if start <= anchor_line <= end else min(abs(start - anchor_line), abs(end - anchor_line)))
    return min(distances) if distances else None


def _point_overlaps_edit(point: dict[str, Any], location: dict[str, Any]) -> bool:
    start, end = point.get("line_start"), point.get("line_end")
    if start is None or end is None or point.get("file") not in _location_paths(location):
        return False
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return False
    return any(
        start <= int(row["end_line"]) and end >= int(row["start_line"])
        for row in location.get("old_edit_ranges") or []
    )


def _point_contains_anchor(point: dict[str, Any], location: dict[str, Any]) -> bool:
    start, end = point.get("line_start"), point.get("line_end")
    if start is None or end is None or point.get("file") not in _location_paths(location):
        return False
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return False
    return any(start <= max(1, int(anchor)) <= end for anchor in location.get("insertion_anchors") or [])


def _file_rank_and_recall(locations: list[dict[str, Any]], files: list[str]) -> tuple[int | None, list[int], dict[int, int]]:
    matched = []
    seen = set()
    location_ranks = {}
    for index, path in enumerate(files, 1):
        for location_index, location in enumerate(locations):
            if location_index in seen:
                continue
            if path in _location_paths(location):
                seen.add(location_index)
                matched.append(location_index)
                location_ranks[location_index] = index
                break
    first = min(location_ranks.values(), default=None)
    return first, matched, location_ranks


def _aggregate(rows: list[dict[str, Any]], *, field_names: tuple[str, ...]) -> dict[str, Any]:
    result = {"n": len(rows)}
    for name in field_names:
        values = [row[name] for row in rows if row.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


def evaluate_one(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    locations = _gold_locations(gold)
    points = _predicted_points(pred)
    fix_files = list(dict.fromkeys(_norm(x.get("file")) for x in points if x.get("file")))
    fix_file_rank, matched_locations, location_ranks = _file_rank_and_recall(locations, fix_files)
    gold_symbols = _gold_symbols(locations)
    symbol_ranks = [
        index for index, point in enumerate(points, 1)
        if any(_symbol_matches(point, location, symbol) for location, symbol in gold_symbols)
    ]
    distances = [
        distance for point in points for location in locations
        if (distance := _point_range_distance(point, location)) is not None
    ]
    range_overlap = int(any(
        _point_overlaps_edit(point, location)
        for point in points for location in locations
    ))
    insertion_anchor_hit = int(any(
        _point_contains_anchor(point, location)
        for point in points for location in locations
    ))
    relaxed_5 = int(any(distance <= 5 for distance in distances)) if distances else 0
    relaxed_10 = int(any(distance <= 10 for distance in distances)) if distances else 0
    qualified_symbol_hit = int(bool(symbol_ranks)) if gold_symbols else None
    fix_location_hit = int(any(
        any(_symbol_matches(point, location, symbol) for point in points for symbol in location.get("symbols") or [])
        or any(_point_range_distance(point, location) == 0 for point in points)
        for location in locations
    )) if locations else 0

    likely_files = list(dict.fromkeys(_norm(x) for x in pred.get("likely_files") or []))
    exploration_rank = next((i + 1 for i, path in enumerate(likely_files) if any(path in _location_paths(x) for x in locations)), None)
    symbol_name_hit = int(any(name in {str(s.get("symbol")) for _, s in gold_symbols} for name in _predicted_symbol_names(pred))) if gold_symbols else None
    symbol_name_ranks = [
        index for index, name in enumerate(_predicted_symbol_names(pred), 1)
        if any(name in {str(symbol.get("symbol")), str(symbol.get("qualified_symbol") or "")} for _, symbol in gold_symbols)
    ]
    return {
        "fix_file_hit_at_1": int(fix_file_rank == 1),
        "fix_file_hit_at_3": int(any(rank <= 3 for rank in location_ranks.values())),
        "fix_file_hit_at_5": int(any(rank <= 5 for rank in location_ranks.values())),
        "fix_file_mrr": 0.0 if fix_file_rank is None else 1.0 / fix_file_rank,
        "fix_file_recall_at_5": (len(matched_locations) / len(locations)) if locations else 0.0,
        "qualified_symbol_hit": qualified_symbol_hit,
        "qualified_symbol_mrr": (0.0 if not symbol_ranks else 1.0 / symbol_ranks[0]) if gold_symbols else None,
        "symbol_name_hit": symbol_name_hit,
        "range_overlap_hit": range_overlap,
        "insertion_anchor_hit": insertion_anchor_hit,
        "range_hit_at_5_lines": relaxed_5,
        "range_hit_at_10_lines": relaxed_10,
        "line_distance": min(distances) if distances else None,
        "fix_location_hit": fix_location_hit,
        "gold_fix_files": sorted({path for location in locations for path in _location_paths(location)}),
        "pred_fix_files": fix_files[:5],
        "exploration_file_hit_at_1": int(exploration_rank == 1),
        "exploration_file_hit_at_3": int(exploration_rank is not None and exploration_rank <= 3),
        "exploration_file_hit_at_5": int(exploration_rank is not None and exploration_rank <= 5),
        "exploration_file_mrr": 0.0 if exploration_rank is None else 1.0 / exploration_rank,
        # Deprecated aliases: these retain the former likely_files semantics.
        "file_hit1": int(exploration_rank == 1),
        "file_hit3": int(exploration_rank is not None and exploration_rank <= 3),
        "file_hit5": int(exploration_rank is not None and exploration_rank <= 5),
        "file_hit@1": int(exploration_rank == 1),
        "file_hit@3": int(exploration_rank is not None and exploration_rank <= 3),
        "file_hit@5": int(exploration_rank is not None and exploration_rank <= 5),
        "file_mrr": 0.0 if exploration_rank is None else 1.0 / exploration_rank,
        "symbol_hit": symbol_name_hit,
        "gold_file_rank": exploration_rank,
        "gold_symbol_rank": (symbol_name_ranks[0] if symbol_name_ranks else None) if gold_symbols else None,
    }


def _aggregate_compat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate(rows, field_names=("file_hit@1", "file_hit@3", "file_hit@5", "file_mrr", "symbol_hit"))


def _trace_gold_support(trace_path: str | None, gold_files: list[str]):
    if not trace_path or not Path(trace_path).exists():
        return {}
    events = [json.loads(x) for x in Path(trace_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    evidence_file = {}
    cumulative_tokens = 0
    first_step = None
    tokens_at = None
    gold = set(gold_files)
    for event in events:
        if event["type"] == "LLM_CALL_USAGE":
            cumulative_tokens += int(event["payload"].get("total_tokens", 0) or 0)
        elif event["type"] == "EVIDENCE_ADDED":
            payload = event["payload"]
            evidence_file[payload.get("evidence_id")] = _norm(payload.get("file")) if payload.get("file") else None
        elif event["type"] == "HYPOTHESIS_UPDATED" and first_step is None:
            support = event["payload"].get("supporting_evidence_ids") or []
            if any(evidence_file.get(item) in gold for item in support):
                first_step = event["payload"].get("updated_step")
                tokens_at = cumulative_tokens
    total = sum(int(event["payload"].get("total_tokens", 0) or 0) for event in events if event["type"] == "LLM_CALL_USAGE")
    post = max(0, total - int(tokens_at)) if tokens_at is not None else None
    return {
        "first_gold_support_step": first_step,
        "tokens_at_first_gold_support": tokens_at,
        "post_gold_support_tokens": post,
        "post_gold_support_ratio": post / total if post is not None and total else None,
    }


def evaluate_dataset(gold_root, predictions_path):
    preds = {}
    meta = {}
    path = Path(predictions_path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("predictions", [])
    duplicate_ids = []
    for record in records:
        task_id = record["task_id"]
        if task_id in preds:
            duplicate_ids.append(task_id)
        preds[task_id] = record.get("report") or {}
        meta[task_id] = {
            "status": record.get("status"),
            "report_source": record.get("report_source") or (record.get("report") or {}).get("report_source"),
            "state": record.get("state") or {},
            "trace": record.get("trace") or {},
        }

    rows = []
    expected = 0
    missing = []
    for directory in sorted(Path(gold_root).iterdir()):
        ground_truth_path = directory / "ground_truth.json"
        if not ground_truth_path.exists():
            continue
        expected += 1
        gold = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        if directory.name not in preds:
            missing.append(directory.name)
            metrics = evaluate_one(gold, {})
            metadata = {}
        else:
            metrics = evaluate_one(gold, preds[directory.name])
            metadata = meta.get(directory.name, {})
        metrics.update({
            "task_id": directory.name,
            "status": metadata.get("status"),
            "report_source": metadata.get("report_source"),
            "forced_finalization": bool((metadata.get("state") or {}).get("forced_finalization")),
        })
        metrics.update(_trace_gold_support(metadata.get("trace", {}).get("trace_path"), metrics["gold_fix_files"]))
        rows.append(metrics)

    fix_fields = (
        "fix_file_hit_at_1", "fix_file_hit_at_3", "fix_file_hit_at_5", "fix_file_mrr",
        "fix_file_recall_at_5", "qualified_symbol_hit", "qualified_symbol_mrr", "symbol_name_hit",
        "range_overlap_hit", "range_hit_at_5_lines", "range_hit_at_10_lines", "line_distance", "fix_location_hit",
        "insertion_anchor_hit",
    )
    exploration_fields = (
        "exploration_file_hit_at_1", "exploration_file_hit_at_3", "exploration_file_hit_at_5",
        "exploration_file_mrr",
    )
    llm = [row for row in rows if row.get("report_source") == "llm"]
    fallback = [row for row in rows if row.get("report_source") == "fallback"]
    forced = [row for row in rows if row.get("forced_finalization")]
    return {
        "schema_version": 2,
        "fix_localization": _aggregate(rows, field_names=fix_fields),
        "exploration": _aggregate(rows, field_names=exploration_fields),
        "execution": {
            "fallback_count": len(fallback),
            "fallback_rate": len(fallback) / (len(rows) or 1),
            "partial_success_count": sum(row.get("status") == "partial_success" for row in rows),
            "forced_finalization_count": len(forced),
        },
        "aggregate": _aggregate_compat(rows),
        "by_report_source": {
            "llm": _aggregate(llm, field_names=fix_fields),
            "fallback": _aggregate(fallback, field_names=fix_fields),
        },
        "forced_finalization": _aggregate(forced, field_names=fix_fields),
        "coverage": {
            "expected_tasks": expected,
            "predicted_tasks": expected - len(missing),
            "missing_predictions": len(missing),
            "coverage": (expected - len(missing)) / (expected or 1),
            "missing_task_ids": missing,
            "duplicate_prediction_task_ids": sorted(set(duplicate_ids)),
            "extra_prediction_task_ids": sorted(set(preds) - {row["task_id"] for row in rows}),
        },
        "cases": rows,
    }
