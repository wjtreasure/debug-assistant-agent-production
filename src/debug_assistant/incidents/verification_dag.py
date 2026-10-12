"""Minimal Dynamic Verification DAG state and orchestration adapter.

The DAG is an orchestration contract, not a second tool runtime.  The
``VerificationDAGHarnessAdapter`` owns structural validation, evidence-ID
admission and legal task transitions.  An optional LangGraph adapter only
routes a fixed topology around that adapter; it never executes a Tool or
creates Evidence itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

VerificationTaskStatus = Literal[
    "PENDING", "READY", "RUNNING", "COMPLETED", "FAILED", "BLOCKED",
]
TaskExecutionStatus = Literal["COMPLETED", "FAILED", "BLOCKED"]


class VerificationDAGError(ValueError):
    """Raised when a graph proposal or transition violates the MVP contract."""

    def __init__(self, message: str, *, reason_code: str = "dag_contract_invalid"):
        super().__init__(message)
        self.reason_code = reason_code


class VerificationTask(BaseModel):
    """Execution state for one canonical VerificationObligation.

    Semantic claim, evidence requirement, criticality and Evidence links live
    exclusively in RCAState/EvidenceSet. A task carries only the Obligation
    reference and scheduling/execution metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    obligation_id: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    status: VerificationTaskStatus = "PENDING"
    result_ref: str = ""
    retry_count: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_task_shape(self) -> "VerificationTask":
        if self.task_id in self.dependencies:
            raise ValueError("verification task cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("verification task dependencies must be unique")
        return self

    @property
    def terminal(self) -> bool:
        return self.status in {"COMPLETED", "FAILED", "BLOCKED"}

class VerificationTransition(BaseModel):
    """Small append-only process event for deterministic trace metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal[
        "TASK_READY", "TASK_SELECTED", "TASK_EXECUTED", "BRANCH_BLOCKED", "LOCAL_REPLAN"
    ]
    task_id: str = ""
    from_status: VerificationTaskStatus | None = None
    to_status: VerificationTaskStatus | None = None
    reason: str = ""


class VerificationDAGState(BaseModel):
    """Serializable graph state; task data lives here, not in LangGraph topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[VerificationTask, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    selected_task_id: str | None = None
    local_replan_count: int = Field(default=0, ge=0)
    transitions: tuple[VerificationTransition, ...] = ()

    @model_validator(mode="after")
    def validate_state_shape(self) -> "VerificationDAGState":
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("verification task IDs must be unique")
        if len(set(self.obligation_ids)) != len(self.obligation_ids):
            raise ValueError("verification obligation IDs must be unique")
        missing_obligations = sorted(
            {task.obligation_id for task in self.tasks} - set(self.obligation_ids)
        )
        if missing_obligations:
            raise ValueError(
                "verification tasks referenced unknown obligations: "
                + ",".join(missing_obligations)
            )
        if self.selected_task_id is not None:
            selected = next(
                (task for task in self.tasks if task.task_id == self.selected_task_id),
                None,
            )
            if selected is None or selected.status != "READY":
                raise ValueError("selected task must be an existing READY task")
        return self

    def task(self, task_id: str) -> VerificationTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise VerificationDAGError(f"unknown verification task: {task_id}")

    @property
    def ready_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.status == "READY")

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        """Return non-terminal execution work without semantic meaning."""
        return tuple(task.task_id for task in self.tasks if not task.terminal)

    @property
    def terminal(self) -> bool:
        return not self.ready_task_ids and not any(
            task.status == "PENDING" for task in self.tasks
        )

    def metrics(self) -> dict[str, int]:
        counts = {
            "task_count": len(self.tasks),
            "pending_task_count": 0,
            "ready_task_count": 0,
            "running_task_count": 0,
            "completed_task_count": 0,
            "failed_task_count": 0,
            "blocked_task_count": 0,
            "local_replan_count": self.local_replan_count,
        }
        for task in self.tasks:
            counts[f"{task.status.lower()}_task_count"] += 1
        return counts

class VerificationTaskExecution(BaseModel):
    """Execution-only result accepted by the orchestration adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    status: TaskExecutionStatus
    result_ref: str = ""
    reason: str = ""


class VerificationGraphState(TypedDict, total=False):
    """JSON-compatible state shape passed through the optional LangGraph."""

    dag_state: dict[str, Any]
    selected_task_id: str | None
    execution: dict[str, Any] | None
    replan_tasks: list[dict[str, Any]]
    replan_requested: bool
    terminal_status: str
    failure_category: str
    error_type: str
    error_message: str


def _validate_dependency_graph(tasks: Sequence[VerificationTask]) -> None:
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise VerificationDAGError("verification task IDs must be unique")
    for task in tasks:
        unknown = sorted(set(task.dependencies) - task_ids)
        if unknown:
            raise VerificationDAGError(
                f"task {task.task_id} has unknown dependencies: {unknown}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {task.task_id: task for task in tasks}

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise VerificationDAGError("verification task dependencies contain a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].dependencies:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)


def _runtime_owned_task(task: VerificationTask) -> VerificationTask:
    """Reset provider-supplied lifecycle state before scheduling."""

    return task.model_copy(
        update={
            "status": "PENDING",
        }
    )


def _canonicalize_tasks(
    tasks: tuple[VerificationTask, ...],
) -> tuple[VerificationTask, ...]:
    """Validate task references without creating semantic state."""
    canonical_tasks: list[VerificationTask] = []
    for task in tasks:
        if not task.obligation_id:
            raise VerificationDAGError(
                f"task {task.task_id} must reference an admitted canonical obligation"
            )
        canonical_tasks.append(_runtime_owned_task(task))
    return tuple(canonical_tasks)


class VerificationDAGHarnessAdapter:
    """Harness-facing deterministic state machine for the DAG MVP.

    This adapter does not call a Tool.  The caller must execute the selected
    task through the existing Registry/Runtime and feed back a
    ``VerificationTaskExecution`` containing an immutable Observation reference.
    """

    def __init__(self, *, max_local_replans: int = 1):
        if max_local_replans < 1:
            raise ValueError("the MVP requires at least one local replan")
        self.max_local_replans = max_local_replans

    def initialize(
        self,
        tasks: Iterable[VerificationTask],
        *,
        obligation_ids: Iterable[str] = (),
    ) -> VerificationDAGState:
        canonical_tasks = _canonicalize_tasks(tuple(tasks))
        ids = tuple(dict.fromkeys(str(item) for item in obligation_ids if str(item)))
        state = VerificationDAGState(
            tasks=canonical_tasks,
            obligation_ids=ids,
        )
        _validate_dependency_graph(state.tasks)
        return self._refresh(state, reason="initialization")

    def register_obligation_ids(
        self,
        state: VerificationDAGState,
        obligation_ids: Iterable[str],
    ) -> VerificationDAGState:
        """Extend the execution projection with canonical Obligation IDs.

        RCAState remains the semantic authority.  The DAG stores only the
        IDs needed to validate future Task references; it never copies
        Obligation claims, evidence requirements, or lifecycle status.
        Existing task history is preserved.
        """

        additions = tuple(
            dict.fromkeys(str(item) for item in obligation_ids if str(item))
        )
        if not additions:
            return state
        registered = tuple(dict.fromkeys((*state.obligation_ids, *additions)))
        if registered == state.obligation_ids:
            return state
        return state.model_copy(update={"obligation_ids": registered})

    def add_tasks(
        self,
        state: VerificationDAGState,
        tasks: Iterable[VerificationTask],
    ) -> VerificationDAGState:
        """Register newly planned tasks without consuming the replan budget.

        Normal Planner turns append fresh task data.  ``local_replan`` remains
        reserved for the bounded recovery path after a blocked/contradicted
        branch, so the integration does not mistake every Planner turn for a
        replan.
        """
        raw_additions = tuple(tasks)
        additions = _canonicalize_tasks(raw_additions)
        if not additions:
            raise VerificationDAGError("task registration requires at least one task")
        unknown_obligation_ids = sorted(
            {task.obligation_id for task in additions} - set(state.obligation_ids)
        )
        if unknown_obligation_ids:
            raise VerificationDAGError(
                "task registration references unregistered obligation IDs: "
                + ",".join(unknown_obligation_ids)
            )
        existing_ids = {task.task_id for task in state.tasks}
        duplicate_ids = sorted(existing_ids & {task.task_id for task in additions})
        if duplicate_ids:
            raise VerificationDAGError(
                f"task registration cannot overwrite existing tasks: {duplicate_ids}"
            )
        candidate_tasks = (*state.tasks, *additions)
        _validate_dependency_graph(candidate_tasks)
        next_state = state.model_copy(
            update={
                "tasks": candidate_tasks,
                "obligation_ids": state.obligation_ids,
                "selected_task_id": None,
            }
        )
        return self._refresh(next_state, reason="planner task registration")

    def select_ready(
        self,
        state: VerificationDAGState,
        task_id: str | None = None,
    ) -> tuple[VerificationDAGState, VerificationTask | None]:
        ready_ids = state.ready_task_ids
        if task_id is not None:
            if task_id not in ready_ids:
                raise VerificationDAGError(f"task is not READY: {task_id}")
            selected_id = task_id
        else:
            if len(ready_ids) > 1:
                raise VerificationDAGError(
                    "multiple READY tasks require an explicit planning selection",
                    reason_code="dag_ready_ambiguous",
                )
            selected_id = ready_ids[0] if ready_ids else None
        if selected_id is None:
            return state.model_copy(update={"selected_task_id": None}), None
        selected = state.task(selected_id)
        transition = VerificationTransition(
            event="TASK_SELECTED", task_id=selected_id, from_status="READY", to_status="READY"
        )
        next_state = state.model_copy(
            update={
                "selected_task_id": selected_id,
                "transitions": (*state.transitions, transition),
            }
        )
        return next_state, selected

    def apply_execution(
        self,
        state: VerificationDAGState,
        execution: VerificationTaskExecution,
    ) -> VerificationDAGState:
        task = state.task(execution.task_id)
        if task.status != "READY":
            raise VerificationDAGError(
                f"task execution requires READY status: {task.task_id}={task.status}"
            )
        if state.selected_task_id not in {None, task.task_id}:
            raise VerificationDAGError(
                f"execution does not match selected task: {state.selected_task_id}"
            )
        updates = {"status": execution.status, "result_ref": execution.result_ref}
        transition = VerificationTransition(
            event="TASK_EXECUTED",
            task_id=task.task_id,
            from_status=task.status,
            to_status=execution.status,
            reason=execution.reason,
        )
        projected_tasks = tuple(
            item.model_copy(update=updates) if item.task_id == task.task_id else item
            for item in state.tasks
        )
        next_state = state.model_copy(
            update={
                "tasks": projected_tasks,
                "selected_task_id": None,
                "transitions": (*state.transitions, transition),
            }
        )
        return self._refresh(next_state, reason=f"{execution.status.lower()} execution")

    def local_replan(
        self,
        state: VerificationDAGState,
        tasks: Iterable[VerificationTask],
    ) -> VerificationDAGState:
        if state.local_replan_count >= self.max_local_replans:
            raise VerificationDAGError("local replan limit exceeded")
        raw_additions = tuple(tasks)
        additions = _canonicalize_tasks(raw_additions)
        if not additions:
            raise VerificationDAGError("local replan requires at least one task")
        unknown_obligation_ids = sorted(
            {task.obligation_id for task in additions} - set(state.obligation_ids)
        )
        if unknown_obligation_ids:
            raise VerificationDAGError(
                "local replan references unregistered obligation IDs: "
                + ",".join(unknown_obligation_ids)
            )
        existing_ids = {task.task_id for task in state.tasks}
        duplicate_ids = sorted(existing_ids & {task.task_id for task in additions})
        if duplicate_ids:
            raise VerificationDAGError(
                f"local replan cannot overwrite existing tasks: {duplicate_ids}"
            )
        candidate_tasks = (*state.tasks, *additions)
        _validate_dependency_graph(candidate_tasks)
        transition = VerificationTransition(
            event="LOCAL_REPLAN",
            reason="one bounded replan after branch invalidation",
        )
        next_state = state.model_copy(
            update={
                "tasks": candidate_tasks,
                "obligation_ids": state.obligation_ids,
                "local_replan_count": state.local_replan_count + 1,
                "selected_task_id": None,
                "transitions": (*state.transitions, transition),
            }
        )
        return self._refresh(next_state, reason="local replan")

    def _refresh(self, state: VerificationDAGState, *, reason: str) -> VerificationDAGState:
        by_id = {task.task_id: task for task in state.tasks}
        changed = True
        transitions = list(state.transitions)
        while changed:
            changed = False
            for task in state.tasks:
                current = by_id[task.task_id]
                if current.terminal:
                    continue
                dependencies = [by_id[item] for item in current.dependencies]
                if any(item.status in {"FAILED", "BLOCKED"} for item in dependencies):
                    updated = current.model_copy(update={"status": "BLOCKED"})
                    transitions.append(
                        VerificationTransition(
                            event="BRANCH_BLOCKED",
                            task_id=current.task_id,
                            from_status=current.status,
                            to_status="BLOCKED",
                            reason=reason,
                        )
                    )
                elif all(item.status == "COMPLETED" for item in dependencies):
                    updated = current.model_copy(update={"status": "READY"})
                    if current.status != "READY":
                        transitions.append(
                            VerificationTransition(
                                event="TASK_READY",
                                task_id=current.task_id,
                                from_status=current.status,
                                to_status="READY",
                                reason=reason,
                            )
                        )
                else:
                    updated = current.model_copy(update={"status": "PENDING"})
                if updated != current:
                    by_id[current.task_id] = updated
                    changed = True
        refreshed = tuple(by_id[task.task_id] for task in state.tasks)
        selected_id = state.selected_task_id
        if selected_id is not None and by_id[selected_id].status != "READY":
            selected_id = None
        return state.model_copy(
            update={
                "tasks": refreshed,
                "selected_task_id": selected_id,
                "transitions": tuple(transitions),
            }
        )


class LangGraphUnavailable(RuntimeError):
    """Raised when the optional orchestration dependency is not installed."""


class LangGraphVerificationAdapter:
    """Fixed-topology LangGraph wrapper around the Harness state machine.

    The graph topology is static.  Verification tasks and dependencies remain
    data in ``dag_state``; adding a task never recompiles the topology.
    ``execute_task`` must be a Harness callback, not a raw Tool callback.
    """

    def __init__(
        self,
        harness: VerificationDAGHarnessAdapter,
        execute_task: Callable[[VerificationTask], VerificationTaskExecution],
    ):
        self.harness = harness
        self.execute_task = execute_task

    def compile(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise LangGraphUnavailable(
                "LangGraph is optional; install debug-assistant-agent[orchestration]"
            ) from exc

        graph = StateGraph(VerificationGraphState)
        graph.add_node("plan_or_replan", self._plan_or_replan)
        graph.add_node("select_ready_task", self._select_ready_task)
        graph.add_node("execute_through_harness", self._execute_through_harness)
        graph.add_node("update_task_state", self._update_task_state)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("review", self._review)
        graph.add_edge(START, "plan_or_replan")
        graph.add_edge("plan_or_replan", "select_ready_task")
        graph.add_conditional_edges(
            "select_ready_task",
            self._route_selected,
            {"execute_through_harness": "execute_through_harness", "synthesize": "synthesize"},
        )
        graph.add_edge("execute_through_harness", "update_task_state")
        graph.add_conditional_edges(
            "update_task_state",
            self._route_after_update,
            {
                "plan_or_replan": "plan_or_replan",
                "select_ready_task": "select_ready_task",
                "synthesize": "synthesize",
            },
        )
        graph.add_edge("synthesize", "review")
        graph.add_edge("review", END)
        return graph.compile()

    def run(
        self,
        state: VerificationDAGState,
        *,
        replan_tasks: Sequence[VerificationTask] = (),
        replan_requested: bool = False,
    ) -> dict[str, Any]:
        compiled = self.compile()
        serialized_replans = []
        for task in replan_tasks:
            serialized_replans.append(task.model_dump(mode="json"))
        initial: VerificationGraphState = {
            "dag_state": state.model_dump(mode="json"),
            "replan_tasks": serialized_replans,
            "replan_requested": bool(replan_requested),
        }
        return compiled.invoke(initial)

    def run_fail_closed(
        self,
        state: VerificationDAGState,
        *,
        replan_tasks: Sequence[VerificationTask] = (),
        replan_requested: bool = False,
    ) -> dict[str, Any]:
        """Return an explicit FAILED boundary when orchestration cannot run.

        The existing DiagnosisHarness remains responsible for mapping this
        result into its full ``IncidentRunResult``.  This adapter does not
        turn an exception into a successful diagnosis.
        """

        try:
            return self.run(
                state,
                replan_tasks=replan_tasks,
                replan_requested=replan_requested,
            )
        except Exception as exc:
            return {
                "dag_state": state.model_dump(mode="json"),
                "terminal_status": "FAILED",
                "failure_category": "DAG_ADAPTER_FAILURE",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    def _plan_or_replan(self, state: VerificationGraphState) -> dict[str, Any]:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        if state.get("replan_requested") and state.get("replan_tasks"):
            additions = [
                VerificationTask.model_validate(item) for item in state["replan_tasks"]
            ]
            dag = self.harness.local_replan(dag, additions)
            return {
                "dag_state": dag.model_dump(mode="json"),
                "replan_tasks": [],
                "replan_requested": False,
            }
        return {"dag_state": dag.model_dump(mode="json"), "replan_requested": False}

    def _select_ready_task(self, state: VerificationGraphState) -> dict[str, Any]:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        dag, task = self.harness.select_ready(dag)
        return {
            "dag_state": dag.model_dump(mode="json"),
            "selected_task_id": task.task_id if task else None,
        }

    def _execute_through_harness(self, state: VerificationGraphState) -> dict[str, Any]:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        selected_id = state.get("selected_task_id")
        if not selected_id:
            raise VerificationDAGError("execution node entered without a selected task")
        execution = self.execute_task(dag.task(selected_id))
        if not isinstance(execution, VerificationTaskExecution):
            execution = VerificationTaskExecution.model_validate(execution)
        return {"execution": execution.model_dump(mode="json")}

    def _update_task_state(self, state: VerificationGraphState) -> dict[str, Any]:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        execution = VerificationTaskExecution.model_validate(state["execution"])
        # Evidence is admitted by the Harness callback. The DAG only receives
        # the resulting execution status; Evidence IDs never become DAG state.
        dag = self.harness.apply_execution(dag, execution)
        return {
            "dag_state": dag.model_dump(mode="json"),
            "selected_task_id": None,
            "replan_requested": execution.status in {"FAILED", "BLOCKED"},
            "execution": None,
        }

    @staticmethod
    def _route_selected(state: VerificationGraphState) -> str:
        return "execute_through_harness" if state.get("selected_task_id") else "synthesize"

    def _route_after_update(self, state: VerificationGraphState) -> str:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        if (
            state.get("replan_requested")
            and state.get("replan_tasks")
            and dag.local_replan_count < self.harness.max_local_replans
        ):
            return "plan_or_replan"
        if dag.ready_task_ids:
            return "select_ready_task"
        return "synthesize"

    @staticmethod
    def _synthesize(state: VerificationGraphState) -> dict[str, Any]:
        dag = VerificationDAGState.model_validate(state["dag_state"])
        return {
            "terminal_status": (
                "EXECUTION_COMPLETE" if dag.terminal else "EXECUTION_PENDING"
            )
        }

    @staticmethod
    def _review(state: VerificationGraphState) -> dict[str, Any]:
        # Review routing is represented in the fixed topology.  Semantic Review
        # remains the existing bounded Reviewer/Harness responsibility.
        return {}
