from __future__ import annotations

from hashlib import sha1

from debug_assistant.context.indexes import extract_numbered_range
from debug_assistant.context.models import ContextItem, ContextProjection


class CodeProjectionPolicy:
    """Existing repository/read-file projection semantics."""

    catalog_mode = "source_ranges"
    supports_source_ranges = True

    def project(self, obs, item: ContextItem, step: int, *, requests=(), rehydrate_requested=False):
        path = (obs.metadata or {}).get("path")
        source_start = (obs.metadata or {}).get("start_line")
        source_end = (obs.metadata or {}).get("end_line")
        if obs.tool == "read_file" and requests and path:
            parts, visible = [], []
            for start, end, _ in requests:
                text, visible_start, visible_end = extract_numbered_range(obs.content, start, end)
                if text:
                    parts.append(text)
                    visible.append((visible_start, visible_end))
            if parts:
                content = "\n".join(parts)
                display_start = min(start for start, _ in visible)
                display_end = max(end for _, end in visible)
                projection_id = "proj-" + sha1(
                    f"{obs.observation_id}|{visible}".encode()
                ).hexdigest()[:10]
                return ContextProjection(
                    projection_id, obs.observation_id, path, source_start, source_end,
                    display_start, display_end, content, item.priority, "active", True,
                    step, "rehydrated_exact_range",
                )

        display = item.full_content
        if obs.tool == "read_file" and path:
            numbers = []
            for line in display.splitlines():
                if "|" not in line:
                    continue
                head = line.split("|", 1)[0].strip()
                if head.isdigit():
                    numbers.append(int(head))
            display_start = numbers[0] if numbers else None
            display_end = numbers[-1] if numbers else None
        else:
            display_start = display_end = None
        projection_id = "proj-" + sha1(
            f"{obs.observation_id}|default|{len(display)}".encode()
        ).hexdigest()[:10]
        return ContextProjection(
            projection_id, obs.observation_id, path, source_start, source_end,
            display_start, display_end, display, item.priority, item.lifecycle,
            item.pinned, step, item.metadata.get("selection_reason", ""),
        )


class IncidentProjectionPolicy:
    """Fault-agnostic structural projection for incident observations."""

    catalog_mode = "items"
    # Source reads use the same exact-range rehydration contract as repository
    # diagnosis. Structured incident observations continue to use item projection.
    supports_source_ranges = True

    _COMMON_TERMS = (
        "name:", "namespace:", "status:", "phase:", "condition", "ready",
        "restart", "reason:", "message", "warning", "error", "failed",
    )
    _SERVICE_TERMS = (
        "selector", "port:", "ports:", "targetport", "protocol", "endpoint",
        "address", "clusterip", "containerport",
    )
    _RUNTIME_TERMS = (
        "state:", "last state:", "controlled by:", "replicas:", "pods status:",
        "limits:", "requests:", "cpu:", "memory:", "liveness:", "readiness:",
        "probe", "events:", "event", "killed", "evicted", "throttl",
        "containercreating", "crashloop", "pending",
    )
    _CONFIG_TERMS = (
        "kind:", "metadata:", "spec:", "containers:", "resources:", "limits:",
        "requests:", "cpu:", "memory:", "ports:", "port:", "containerport:",
        "targetport:", "selector:", "matchlabels:", "env:", "value:",
        "livenessprobe:", "readinessprobe:", "grpc:", "httpget:",
    )
    # Deliberately small extension seams for later slices. They define structural
    # retention only; they do not encode fault labels or diagnosis rules.
    _LOG_TERMS = (
        "timestamp", "level", "logger", "trace", "exception", "stack",
        "request", "response", "latency", "timeout", "status_code",
    )
    _METRIC_TERMS = (
        "metric", "value", "labels", "timestamp", "cpu", "memory",
        "latency", "duration", "rate", "count", "percent", "quantile",
    )
    _ALERT_TERMS = (
        "abnormal", "anomal", "alert", "resource_saturation", "latency",
        "normal", "current", "time_range", "throttl", "percent",
    )
    _CODE_INDEX_TERMS = (
        "service", "root", "files", "path", "description", ".go", ".py",
        ".js", ".java", ".cs",
    )
    _CODE_TERMS = (
        "func ", "function ", "class ", "type ", "interface ", "struct ",
        "return", "error", "exception", "for ", "while ", "go ", "grpc",
        "rpc", "context.", "time.", "latency", "timeout", "port",
    )
    _MAX_PROJECTED_CHARS = {
        "LOG": 3600,
        "ALERT": 3000,
        "METRIC": 3200,
        "CODE": 5200,
        "CODE_SEARCH": 2600,
        "CODE_INDEX": 2600,
        "POD": 4200,
        "DEPLOYMENT": 4200,
        "EVENT": 3600,
        "CONFIG": 4200,
        "SERVICE": 3200,
        "ENDPOINT": 3200,
    }

    def project(self, obs, item: ContextItem, step: int, *, requests=(), rehydrate_requested=False):
        kind = str((obs.metadata or {}).get("context_kind") or "OBSERVATION").upper()
        metadata = obs.metadata or {}
        if kind == "CODE" and metadata.get("information_source") == "source_read" and obs.tool == "read_file":
            # Reuse the tested source-range projection instead of maintaining a
            # second code-file context lifecycle for incidents.
            return CodeProjectionPolicy().project(
                obs, item, step, requests=requests,
                rehydrate_requested=rehydrate_requested,
            )
        if rehydrate_requested:
            content = obs.content
            reason = "rehydrated_raw_observation"
            lifecycle, pinned = "active", True
        else:
            content = self.compact_content(obs)
            reason = f"incident_{kind.lower()}_projection"
            lifecycle, pinned = item.lifecycle, item.pinned
        projection_id = "proj-" + sha1(
            f"{obs.observation_id}|{reason}|{len(content)}".encode()
        ).hexdigest()[:10]
        return ContextProjection(
            projection_id, obs.observation_id, None, None, None, None, None,
            content, item.priority, lifecycle, pinned, step, reason,
        )

    def compact_content(self, obs) -> str:
        content = obs.content or ""
        kind = str((obs.metadata or {}).get("context_kind") or "OBSERVATION").upper()
        # Small observations are already bounded semantic units; preserve them exactly.
        if len(content) <= 1600:
            return content

        terms = list(self._COMMON_TERMS)
        if kind in {"SERVICE", "ENDPOINT"}:
            terms.extend(self._SERVICE_TERMS)
        elif kind in {"POD", "DEPLOYMENT", "EVENT"}:
            terms.extend(self._RUNTIME_TERMS)
        elif kind == "CONFIG":
            terms.extend(self._SERVICE_TERMS)
            terms.extend(self._RUNTIME_TERMS)
            terms.extend(self._CONFIG_TERMS)
        elif kind == "LOG":
            terms.extend(self._LOG_TERMS)
        elif kind == "ALERT":
            terms.extend(self._ALERT_TERMS)
        elif kind == "CODE_INDEX":
            terms.extend(self._CODE_INDEX_TERMS)
        elif kind == "CODE":
            terms.extend(self._CODE_TERMS)
        elif kind == "CODE_SEARCH":
            terms.extend(("path", "line", "matches", "symbol", "function", "class"))
        elif kind == "METRIC":
            terms.extend(self._METRIC_TERMS)
        else:
            terms.extend(self._SERVICE_TERMS)
            terms.extend(self._RUNTIME_TERMS)

        lines = content.splitlines()
        selected: set[int] = set()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if any(term in lowered for term in terms):
                # Keep a structural neighborhood so YAML values and event table rows
                # retain their parent key/header.
                selected.update(range(max(0, index - 1), min(len(lines), index + 3)))
            if lowered.strip() == "events:":
                selected.update(range(index, len(lines)))

        if not selected:
            return content
        projected_lines = [lines[index] for index in sorted(selected)]
        # Do not let a large event table or repeated log block turn a compact
        # Evidence object back into an 11k-character prompt item. Raw content
        # remains immutable in ObservationStore; this is only model projection.
        if kind in {"LOG", "METRIC"}:
            deduped = []
            previous = None
            for line in projected_lines:
                if line == previous and line.strip():
                    continue
                deduped.append(line)
                previous = line
            projected_lines = deduped
        max_chars = self._MAX_PROJECTED_CHARS.get(kind, 4000)
        projected = "\n".join(projected_lines)
        if len(projected) > max_chars:
            # Keep both the structural header and the newest/terminal signal.
            head_budget = max(200, int(max_chars * 0.58))
            tail_budget = max(200, max_chars - head_budget - 80)
            head, tail = [], []
            used = 0
            for line in projected_lines:
                if used + len(line) + 1 > head_budget:
                    break
                head.append(line)
                used += len(line) + 1
            used = 0
            for line in reversed(projected_lines):
                if used + len(line) + 1 > tail_budget:
                    break
                tail.append(line)
                used += len(line) + 1
            projected = "\n".join(head + ["... [projection_middle_omitted] ..."] + list(reversed(tail)))
        return f"[incident projection kind={kind}]\n{projected}"
