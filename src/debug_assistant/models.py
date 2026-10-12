from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time, json, uuid

class ActionKind(str, Enum):
    TOOL = "tool"
    REFLECT = "reflect"
    FINISH = "finish"
    PARALLEL = "parallel"

@dataclass(slots=True)
class TaskSpec:
    task_id: str
    issue: str
    repo_path: str
    repo_name: str = ""
    base_commit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ActionProposal:
    kind: ActionKind
    skill: str
    reason: str
    confidence: float = 0.5
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_evidence: str = ""
    information_need: str = ""
    information_need_structured: dict[str, Any] | None = None
    retain_context_ids: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def fingerprint(self) -> str:
        payload={"arguments":self.arguments,"actions":self.actions}
        stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return f"{self.kind.value}|{self.skill}|{self.tool}|{stable}"

@dataclass(slots=True)
class ToolObservation:
    tool: str
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    latency_ms: float = 0.0
    observation_id: str = field(default_factory=lambda: f"obs-{uuid.uuid4().hex[:12]}")

@dataclass(slots=True)
class Evidence:
    evidence_id: str
    kind: str
    source: str
    summary: str
    target: str | None = None
    excerpt: str = ""
    file: str | None = None
    # Truth-preserving source provenance and coverage.
    raw_observation_id: str | None = None
    source_start_line: int | None = None
    source_end_line: int | None = None
    excerpt_start_line: int | None = None
    excerpt_end_line: int | None = None
    excerpt_truncated: bool = False
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    # Structured source metadata is kept on the canonical Evidence record so
    # Incident Review can receive a faithful projection without a second store.
    provenance: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    task: TaskSpec
    step: int = 0
    tool_calls: int = 0
    status: str = "running"
    actions: list[ActionProposal] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    invalid_routes: int = 0
    recovered_routes: int = 0
    repeated_actions: int = 0
    reflection_count: int = 0
    reflection_failure_count: int = 0
    max_consecutive_reflection_failures_observed: int = 0
    started_at: float = field(default_factory=time.time)
    termination_advisory: str = ""
    observation_reuse_count: int = 0
    no_progress_count: int = 0
    redundant_request_count: int = 0
    rehydration_count: int = 0
    rehydration_saved_tool_calls: int = 0
    rehydration_saved_chars: int = 0
    convergence_mode: str = "normal"
    forced_finalization: bool = False
    budget_critical_entered: bool = False
    first_supported_hypothesis_step: int | None = None
    first_stable_diagnosis_step: int | None = None
    prompt_tokens_at_first_stable_diagnosis: int | None = None
    completion_tokens_at_first_stable_diagnosis: int | None = None
    tokens_at_first_stable_diagnosis: int | None = None
    planner_contract_failure_count: int = 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "step": self.step, "tool_calls": self.tool_calls, "status": self.status,
            "evidence": len(self.evidence),
            "invalid_routes": self.invalid_routes, "recovered_routes": self.recovered_routes,
            "repeated_actions": self.repeated_actions, "reflection_count": self.reflection_count,
            "reflection_failure_count": self.reflection_failure_count,
            "max_consecutive_reflection_failures_observed": self.max_consecutive_reflection_failures_observed,
            "errors": self.errors[-5:],
            "observation_reuse_count": self.observation_reuse_count,
            "no_progress_count": self.no_progress_count,
            "redundant_request_count": self.redundant_request_count, "rehydration_count": self.rehydration_count,
            "rehydration_saved_tool_calls": self.rehydration_saved_tool_calls, "rehydration_saved_chars": self.rehydration_saved_chars,
            "convergence_mode": self.convergence_mode, "forced_finalization": self.forced_finalization,
            "budget_critical_entered": self.budget_critical_entered,
            "first_supported_hypothesis_step": self.first_supported_hypothesis_step,
            "first_stable_diagnosis_step": self.first_stable_diagnosis_step,
            "prompt_tokens_at_first_stable_diagnosis": self.prompt_tokens_at_first_stable_diagnosis,
            "completion_tokens_at_first_stable_diagnosis": self.completion_tokens_at_first_stable_diagnosis,
            "tokens_at_first_stable_diagnosis": self.tokens_at_first_stable_diagnosis,
            "planner_contract_failure_count": self.planner_contract_failure_count,
        }
