from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable


_RANGE_RE = re.compile(r"(?i)\blines?\s+(\d+)(?:\s*[-–]\s*(\d+))?")
_SYMBOL_RANGE_RE = re.compile(
    r"(?<![\w/.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w+)*)\s+lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?"
)
_CLASS_RE = re.compile(r"(?<![\w/.])([A-Za-z_]\w*)\s+class\b", re.I)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(getattr(value, "__dict__", {}) or {})


def _clean_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r"(?i)(?:#L\d+(?:-L?\d+)?|:\d+(?:-\d+)?|\s*\(\s*lines?\s+\d+(?:-\d+)?\s*\))$", "", text)
    if not text or " " in PurePosixPath(text).name:
        return ""
    return PurePosixPath(text).as_posix()


def _path_matches(hint: Any, path: str) -> bool:
    value = _clean_path(hint)
    if not value:
        return False
    return value == path or value.endswith("/" + path) or path.endswith("/" + value)


def _source_files(evidence: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in evidence:
        if getattr(item, "source", None) not in {"read_file", "git_show"}:
            continue
        path = _clean_path(getattr(item, "file", None))
        if path and path not in result:
            result.append(path)
    return result


def _explicit_symbol_and_range(location: str) -> tuple[str | None, tuple[int, int] | None]:
    matches = list(_SYMBOL_RANGE_RE.finditer(location or ""))
    if matches:
        match = matches[-1]
        return match.group(1), (int(match.group(2)), int(match.group(3) or match.group(2)))
    ranges = list(_RANGE_RE.finditer(location or ""))
    if ranges:
        match = ranges[-1]
        return None, (int(match.group(1)), int(match.group(2) or match.group(1)))
    class_match = _CLASS_RE.search(location or "")
    return (class_match.group(1) if class_match else None), None


def _target_symbol(target: str) -> str | None:
    value = str(target or "").strip()
    if not value or "/" in value or "\\" in value:
        return None
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w+)*", value):
        return value
    return None


def build_runtime_change_points(hypothesis: Any, evidence: Iterable[Any], *, max_points: int = 3) -> list[dict[str, Any]]:
    """Serialize only source-backed structured diagnosis into change-point candidates.

    This helper never searches the repository and never invents a location. A candidate
    requires a hypothesis file hint that matches immutable source evidence. Line ranges
    are copied only from the structured hypothesis location; otherwise they remain null.
    """
    hyp = _as_dict(hypothesis)
    status = str(hyp.get("status") or "")
    try:
        confidence = float(hyp.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    # A low-confidence partial diagnosis is still useful as exploration context,
    # but must not be promoted into a formal fix location during fallback.
    if status == "partial" and confidence < 0.7:
        return []
    source = list(evidence)
    source_files = _source_files(source)
    target = str(hyp.get("root_cause_target") or "")
    location = str(hyp.get("root_cause_location") or "")
    hints = (target, location)
    matching_file = next((path for path in source_files if any(_path_matches(hint, path) for hint in hints)), None)
    if not matching_file:
        return []

    symbol, line_range = _explicit_symbol_and_range(location)
    symbol = _target_symbol(target) or symbol
    supporting_ids = [
        str(item)
        for item in hyp.get("supporting_evidence_ids") or []
        if any(str(getattr(ev, "evidence_id", "")) == str(item) and _clean_path(getattr(ev, "file", None)) == matching_file for ev in source)
    ]
    if not supporting_ids:
        supporting_ids = [
            str(getattr(ev, "evidence_id"))
            for ev in source
            if getattr(ev, "source", None) == "read_file" and _clean_path(getattr(ev, "file", None)) == matching_file
        ][:3]
    hypothesis_id = str(hyp.get("hypothesis_id") or "")
    point = {
        "file": matching_file,
        "symbol": symbol,
        "line_start": line_range[0] if line_range else None,
        "line_end": line_range[1] if line_range else None,
        "reason": "Source-backed runtime candidate copied from the structured root-cause target/location; no new diagnosis was inferred.",
        "confidence": confidence,
        "supporting_evidence_ids": supporting_ids,
        "supporting_hypothesis_ids": [hypothesis_id] if hypothesis_id else [],
    }
    return [point][: max(1, int(max_points))]
