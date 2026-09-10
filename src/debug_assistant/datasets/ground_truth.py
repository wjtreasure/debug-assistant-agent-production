from __future__ import annotations

import posixpath
from typing import NotRequired, TypedDict


GROUND_TRUTH_SCHEMA_VERSION = 2


class LineRange(TypedDict):
    start_line: int
    end_line: int


class SymbolLocation(TypedDict):
    symbol: str
    qualified_symbol: NotRequired[str]
    kind: NotRequired[str]
    start_line: int
    end_line: int


class FixLocation(TypedDict):
    old_path: str | None
    new_path: str | None
    status: str
    old_edit_ranges: list[LineRange]
    new_edit_ranges: list[LineRange]
    insertion_anchors: list[int]
    hunk_ranges: list[dict[str, LineRange | None]]
    symbols: list[SymbolLocation]
    new_symbols: list[SymbolLocation]


class GroundTruthV2(TypedDict):
    schema_version: int
    instance_id: str
    fix_locations: list[FixLocation]


def normalize_repo_path(path: str | None) -> str | None:
    """Normalize a repository-relative path without erasing traversal."""
    if path is None:
        return None
    value = str(path).replace("\\", "/")
    if value == "/dev/null":
        return None
    if value.startswith("/") or (len(value) >= 3 and value[1:3] == ":/"):
        raise ValueError(f"absolute path is not repository-relative: {path!r}")
    value = posixpath.normpath(value)
    if value == ".." or value.startswith("../"):
        raise ValueError(f"parent traversal is not a repository path: {path!r}")
    if value.startswith("./"):
        value = value[2:]
    return value or "."


def validate_ground_truth(value: object, *, instance_id: str | None = None) -> GroundTruthV2:
    """Validate the small shared wire schema used by prepare and evaluation."""
    if not isinstance(value, dict):
        raise ValueError("ground truth must be a JSON object")
    if value.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ground-truth schema: {value.get('schema_version')!r}"
        )
    actual_id = value.get("instance_id")
    if not isinstance(actual_id, str) or not actual_id:
        raise ValueError("ground truth instance_id must be a non-empty string")
    if instance_id is not None and actual_id != instance_id:
        raise ValueError(f"ground truth instance_id mismatch: {actual_id!r}")
    locations = value.get("fix_locations")
    if not isinstance(locations, list):
        raise ValueError("ground truth fix_locations must be a list")
    for location in locations:
        if not isinstance(location, dict):
            raise ValueError("each fix location must be an object")
        for field in (
            "old_edit_ranges",
            "new_edit_ranges",
            "insertion_anchors",
            "hunk_ranges",
            "symbols",
            "new_symbols",
        ):
            if not isinstance(location.get(field), list):
                raise ValueError(f"fix location field {field!r} must be a list")
        for field in ("old_path", "new_path"):
            path = location.get(field)
            if path is not None:
                normalize_repo_path(path)
    return value  # type: ignore[return-value]
