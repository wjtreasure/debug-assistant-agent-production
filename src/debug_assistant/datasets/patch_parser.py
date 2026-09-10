from __future__ import annotations

import re
import shlex
from typing import Any


_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?(?: .*)?@@"
)


def _decode_path(value: str, *, shell_decode: bool = True) -> str | None:
    """Decode one git/unified-diff path and remove only its git prefix."""
    value = value.strip()
    if not value:
        return None
    if shell_decode:
        try:
            parts = shlex.split(value, posix=True)
        except ValueError:
            parts = [value]
        value = parts[0] if parts else value
    if value == "/dev/null":
        return None
    return value.replace("\\", "/")


def _strip_side_prefix(path: str | None, side: str) -> str | None:
    if path is None:
        return None
    prefix = f"{side}/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def _header_paths(line: str) -> tuple[str | None, str | None]:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        raise ValueError(f"invalid diff header: {line!r}") from exc
    if len(parts) < 4 or parts[0] != "diff" or parts[1] != "--git":
        raise ValueError(f"invalid diff header: {line!r}")
    return _strip_side_prefix(_decode_path(parts[2], shell_decode=False), "a"), _strip_side_prefix(
        _decode_path(parts[3], shell_decode=False), "b"
    )


def _file_header_path(line: str) -> str | None:
    value = line[4:]
    # Unified diff timestamps are tab-separated. Spaces are valid in paths.
    if "\t" in value:
        value = value.split("\t", 1)[0]
    return _decode_path(value)


def _add_range(ranges: list[dict[str, int]], start: int, end: int) -> None:
    if ranges and ranges[-1]["end_line"] + 1 == start:
        ranges[-1]["end_line"] = end
    else:
        ranges.append({"start_line": start, "end_line": end})


def _finish_hunk(current: dict[str, Any], hunk: dict[str, Any] | None) -> None:
    if hunk is None:
        return
    current["hunk_ranges"].append(
        {
            "old": (
                {
                    "start_line": hunk["old_start"],
                    "end_line": hunk["old_start"] + hunk["old_count"] - 1,
                }
                if hunk["old_count"]
                else None
            ),
            "new": (
                {
                    "start_line": hunk["new_start"],
                    "end_line": hunk["new_start"] + hunk["new_count"] - 1,
                }
                if hunk["new_count"]
                else None
            ),
        }
    )
    current["old_edit_ranges"].extend(hunk["old_edit_ranges"])
    current["new_edit_ranges"].extend(hunk["new_edit_ranges"])
    if not hunk["old_edit_ranges"] and hunk["new_edit_ranges"]:
        current["insertion_anchors"].append(hunk["insertion_anchor"])


def _finish_file(
    files: list[dict[str, Any]],
    current: dict[str, Any] | None,
    hunk: dict[str, Any] | None,
) -> None:
    if current is None:
        return
    _finish_hunk(current, hunk)
    old_path, new_path = current["old_path"], current["new_path"]
    if old_path is None:
        status = "added"
    elif new_path is None:
        status = "deleted"
    elif old_path != new_path:
        status = "renamed"
    else:
        status = "modified"
    current["status"] = status
    # Compatibility aliases for callers of the old parser API. The v2 gold
    # writer intentionally does not emit these ambiguous names.
    current["path"] = new_path or old_path
    current["modified_ranges"] = [
        {
            "old_start": (x["old"] or {}).get("start_line", 0),
            "old_end": (x["old"] or {}).get("end_line", 0),
            "new_start": (x["new"] or {}).get("start_line", 0),
            "new_end": (x["new"] or {}).get("end_line", 0),
        }
        for x in current["hunk_ranges"]
    ]
    files.append(current)


def parse_unified_patch(patch: str) -> dict[str, list[dict[str, Any]]]:
    """Parse a unified git patch into exact edit and hunk coordinates.

    Line numbers are one-based. ``old_edit_ranges`` and ``new_edit_ranges``
    contain only added/removed lines, while ``hunk_ranges`` includes context.
    A pure insertion has no old edit range and is represented by an old-side
    ``insertion_anchor``.
    """
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    hunk: dict[str, Any] | None = None

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            _finish_file(files, current, hunk)
            old_path, new_path = _header_paths(line)
            current = {
                "old_path": old_path,
                "new_path": new_path,
                "hunk_ranges": [],
                "old_edit_ranges": [],
                "new_edit_ranges": [],
                "insertion_anchors": [],
                "hunks": [],
            }
            hunk = None
            continue

        if current is None:
            continue
        if line.startswith("--- "):
            current["old_path"] = _strip_side_prefix(_file_header_path(line), "a")
            continue
        if line.startswith("+++ "):
            current["new_path"] = _strip_side_prefix(_file_header_path(line), "b")
            continue
        if line.startswith("@@"):
            _finish_hunk(current, hunk)
            match = _HUNK_RE.match(line)
            if not match:
                raise ValueError(f"invalid hunk header: {line!r}")
            old_start, old_count, new_start, new_count = match.groups()
            hunk = {
                "old_start": int(old_start),
                "old_count": int(old_count or 1),
                "new_start": int(new_start),
                "new_count": int(new_count or 1),
                "old_edit_ranges": [],
                "new_edit_ranges": [],
                "insertion_anchor": int(old_start),
                "lines": [],
            }
            current["hunks"].append(hunk)
            continue
        if hunk is None or line.startswith("\\ No newline at end of file"):
            continue

        hunk["lines"].append(line)
        if line.startswith(" "):
            hunk["old_cursor"] = hunk.get("old_cursor", hunk["old_start"]) + 1
            hunk["new_cursor"] = hunk.get("new_cursor", hunk["new_start"]) + 1
        elif line.startswith("-"):
            old_cursor = hunk.get("old_cursor", hunk["old_start"])
            _add_range(hunk["old_edit_ranges"], old_cursor, old_cursor)
            hunk["old_cursor"] = old_cursor + 1
        elif line.startswith("+"):
            old_cursor = hunk.get("old_cursor", hunk["old_start"])
            new_cursor = hunk.get("new_cursor", hunk["new_start"])
            _add_range(hunk["new_edit_ranges"], new_cursor, new_cursor)
            if not hunk["old_edit_ranges"] and "insertion_anchor_set" not in hunk:
                hunk["insertion_anchor"] = (
                    old_cursor - 1 if old_cursor > hunk["old_start"] else hunk["old_start"]
                )
                hunk["insertion_anchor_set"] = True
            hunk["new_cursor"] = new_cursor + 1

    _finish_file(files, current, hunk)
    return {"files": files}


def apply_file_patch(old_text: str, file_data: dict[str, Any]) -> str:
    """Apply one parsed file patch in memory, for new-side AST inspection."""
    old_lines = old_text.splitlines(keepends=True)
    output: list[str] = []
    old_pos = 0
    for hunk in file_data.get("hunks", []):
        start = max(int(hunk["old_start"]) - 1, 0)
        output.extend(old_lines[old_pos:start])
        old_pos = start
        for line in hunk["lines"]:
            if line.startswith(" "):
                if old_pos < len(old_lines):
                    output.append(old_lines[old_pos])
                else:
                    output.append(line[1:] + "\n")
                old_pos += 1
            elif line.startswith("-"):
                old_pos += 1
            elif line.startswith("+"):
                output.append(line[1:] + "\n")
    output.extend(old_lines[old_pos:])
    return "".join(output)
