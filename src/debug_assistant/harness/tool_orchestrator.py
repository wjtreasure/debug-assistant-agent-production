from __future__ import annotations

"""The typed execution boundary between a planner and repository tools."""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from debug_assistant.models import ToolObservation
from debug_assistant.tools.registry import PARALLEL_ALLOWED_TOOLS
from .tool_executor import execute_with_retry
from .retry import RetryPolicy


@dataclass(frozen=True, slots=True)
class RequestedToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    information_need_id: str | None = None
    obligation_ids: tuple[str, ...] = ()
    reason: str = ""
    expected_evidence: str = ""
    retain_context_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpandedToolCall:
    request: RequestedToolCall
    arguments: dict[str, Any]
    requested_range: dict[str, Any] | None = None
    expanded_range: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionPlan:
    parallel_groups: tuple[tuple[ExpandedToolCall, ...], ...] = ()
    serial_calls: tuple[ExpandedToolCall, ...] = ()
    expanded_count: int = 0
    ordered_steps: tuple[tuple[str, tuple[ExpandedToolCall, ...]], ...] = ()

    @property
    def calls(self) -> tuple[ExpandedToolCall, ...]:
        if self.ordered_steps:
            return tuple(call for _, group in self.ordered_steps for call in group)
        return tuple(x for group in self.parallel_groups for x in group) + self.serial_calls


class ToolPlanningError(ValueError):
    def __init__(self, message: str, *, error_type: str = "tool_planning_error", tool: str | None = None):
        self.error_type = error_type
        self.tool = tool
        super().__init__(message)


class ToolOrchestrator:
    """Compile untrusted tool requests into a bounded deterministic execution plan."""

    def __init__(self, registry, *, max_parallel_actions: int = 4,
                 max_tool_calls: int | None = None,
                 read_context_padding: int = 0):
        self.registry = registry
        self.max_parallel_actions = max(1, int(max_parallel_actions))
        self.max_tool_calls = None if max_tool_calls is None else max(0, int(max_tool_calls))
        # Runtime callers may request a bounded source context around a short
        # read. The original range remains in metadata for auditability; only
        # the execution range is widened, never beyond the tool's 200-line cap.
        self.read_context_padding = max(0, int(read_context_padding))

    @staticmethod
    def _split_read_file(call: RequestedToolCall) -> list[ExpandedToolCall]:
        args = dict(call.arguments or {})
        try:
            start, end = int(args.get("start_line", 1)), int(args.get("end_line", 200))
        except (TypeError, ValueError):
            # Keep malformed values intact so the normal Pydantic boundary returns
            # a structured schema error instead of the expander crashing.
            return [ExpandedToolCall(call, args)]
        if start < 1 or end < start:
            return [ExpandedToolCall(call, args)]
        requested = {"path": args.get("path"), "start_line": start, "end_line": end}
        out = []
        for chunk_start in range(start, end + 1, 200):
            chunk_end = min(end, chunk_start + 199)
            chunk_args = dict(args, start_line=chunk_start, end_line=chunk_end)
            out.append(ExpandedToolCall(
                call, chunk_args, requested_range=requested,
                expanded_range={"start_line": chunk_start, "end_line": chunk_end},
            ))
        return out

    def _expand_source_context(self, call: RequestedToolCall) -> RequestedToolCall:
        if self.read_context_padding <= 0 or call.name != "read_file":
            return call
        args = dict(call.arguments or {})
        try:
            start, end = int(args.get("start_line", 1)), int(args.get("end_line", 200))
        except (TypeError, ValueError):
            return call
        if start < 1 or end < start or end - start + 1 >= 200:
            return call
        padding = min(self.read_context_padding, max(0, (200 - (end - start + 1)) // 2))
        if padding <= 0:
            return call
        widened = dict(args, start_line=max(1, start - padding), end_line=end + padding)
        return RequestedToolCall(
            id=call.id, name=call.name, arguments=widened,
            information_need_id=call.information_need_id, obligation_ids=call.obligation_ids,
            reason=call.reason, expected_evidence=call.expected_evidence,
            retain_context_ids=call.retain_context_ids,
        )

    def expand(self, requests: Iterable[RequestedToolCall]) -> list[ExpandedToolCall]:
        expanded: list[ExpandedToolCall] = []
        for call in requests:
            if self.registry.get(call.name) is None:
                raise ToolPlanningError(f"unknown tool: {call.name}", error_type="unknown_tool", tool=call.name)
            if call.name == "read_file":
                widened = self._expand_source_context(call)
                chunks = self._split_read_file(widened)
                if widened is not call and chunks:
                    # Preserve the planner's original range separately from the
                    # context-expanded execution range.
                    original = dict(call.arguments or {})
                    try:
                        requested = {
                            "path": original.get("path"),
                            "start_line": int(original.get("start_line", 1)),
                            "end_line": int(original.get("end_line", 200)),
                        }
                    except (TypeError, ValueError):
                        requested = None
                    chunks = [ExpandedToolCall(x.request, x.arguments, requested, x.expanded_range) for x in chunks]
                expanded.extend(chunks)
            else:
                expanded.append(ExpandedToolCall(call, dict(call.arguments or {})))
        if self.max_tool_calls is not None and len(expanded) > self.max_tool_calls:
            raise ToolPlanningError(
                f"expanded tool calls ({len(expanded)}) exceed max_tool_calls ({self.max_tool_calls})",
                error_type="tool_budget_preflight",
            )
        return expanded

    def build_plan(self, requests: Iterable[RequestedToolCall]) -> ToolExecutionPlan:
        expanded = self.expand(requests)
        parallel: list[list[ExpandedToolCall]] = []
        serial: list[ExpandedToolCall] = []
        ordered: list[tuple[str, tuple[ExpandedToolCall, ...]]] = []
        safe_batch: list[ExpandedToolCall] = []
        def flush_safe():
            if safe_batch:
                group = tuple(safe_batch)
                parallel.append(list(group)); ordered.append(("parallel", group)); safe_batch.clear()
        for call in expanded:
            safe = getattr(self.registry, "is_parallel_safe", lambda n: n in PARALLEL_ALLOWED_TOOLS)(call.request.name)
            if safe:
                safe_batch.append(call)
                if len(safe_batch) == self.max_parallel_actions:
                    flush_safe()
            else:
                flush_safe()
                serial.append(call)
                ordered.append(("serial", (call,)))
        flush_safe()
        return ToolExecutionPlan(tuple(tuple(x) for x in parallel), tuple(serial), len(expanded), tuple(ordered))

    @staticmethod
    def _annotate(observation: ToolObservation, call: ExpandedToolCall) -> ToolObservation:
        metadata = dict(observation.metadata or {})
        metadata.update({
            "requested_tool_call_id": call.request.id,
            "information_need_id": call.request.information_need_id,
            "obligation_ids": list(call.request.obligation_ids),
        })
        if call.requested_range is not None:
            metadata["requested_range"] = dict(call.requested_range)
            metadata["expanded_range"] = dict(call.expanded_range or {})
        observation.metadata = metadata
        return observation

    def execute(self, plan: ToolExecutionPlan, *, deadline: float | None = None,
                retry_policy: RetryPolicy | None = None, execute_parallel=None) -> list[ToolObservation]:
        """Execute in plan order; an optional parallel executor may preserve the existing runtime."""
        observations: list[ToolObservation] = []
        steps = plan.ordered_steps or tuple(("parallel", group) for group in plan.parallel_groups) + tuple(("serial", (call,)) for call in plan.serial_calls)
        for kind, calls in steps:
            if kind == "parallel" and len(calls) > 1:
                if execute_parallel is not None:
                    rows = execute_parallel(calls)
                else:
                    # Results are collected in request order even though the bounded,
                    # read-only group executes concurrently.
                    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                        futures = [pool.submit(self._execute_one, call, deadline=deadline, retry_policy=retry_policy) for call in calls]
                        rows = [future.result() for future in futures]
                observations.extend(self._annotate(obs, call) for call, obs in zip(calls, rows))
            else:
                for call in calls:
                    observations.append(self._annotate(
                        self._execute_one(call, deadline=deadline, retry_policy=retry_policy), call
                    ))
        return observations

    def _execute_one(self, call: ExpandedToolCall, *, deadline: float | None,
                     retry_policy: RetryPolicy | None) -> ToolObservation:
        args, error = self.registry.validate_arguments(call.request.name, call.arguments)
        if error:
            return ToolObservation(call.request.name, False, error["message"], error, error["error_type"])
        tool = self.registry.get(call.request.name)
        absolute_deadline = getattr(deadline, "absolute_deadline", deadline)
        return execute_with_retry(tool, args, policy=retry_policy, absolute_deadline=absolute_deadline)
