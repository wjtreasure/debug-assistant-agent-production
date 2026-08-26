from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any
import time, json, uuid

class ActionKind(str, Enum):
    TOOL = "tool"
    REFLECT = "reflect"
    FINISH = "finish"

class RuntimeStage(str, Enum):
    SETUP = "setup"
    INDEX_BUILD = "index_build"
    PLANNER = "planner"
    ACTION_VALIDATION = "action_validation"
    ROUTE_VALIDATION = "route_validation"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_INGESTION = "memory_ingestion"
    REFLECTION = "reflection"
    REPORTER = "reporter"
    SERIALIZATION = "serialization"
    FINALIZATION = "finalization"

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
    retain_context_ids: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        stable = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)
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
    excerpt: str = ""
    file: str | None = None
    # Backward-compatible source coverage fields used by reporter/older callers.
    line_start: int | None = None
    line_end: int | None = None
    # V1.2-A.1 truth-preserving provenance/coverage.
    raw_observation_id: str | None = None
    source_start_line: int | None = None
    source_end_line: int | None = None
    excerpt_start_line: int | None = None
    excerpt_end_line: int | None = None
    excerpt_truncated: bool = False
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)

@dataclass(slots=True)
class Hypothesis:
    hypothesis_id: str
    claim: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "open"

@dataclass(slots=True)
class DiagnosisReport:
    task_id: str
    summary: str
    root_cause: str
    likely_files: list[str]
    likely_symbols: list[str]
    impact_scope: list[str]
    evidence: list[dict[str, Any]]
    recommended_change_points: list[dict[str, Any]]
    uncertainties: list[str]
    next_checks: list[str]
    confidence: float
    policy_note: str = "Read-only diagnosis: no code changes were made."
    report_source: str = "llm"
    evidence_ids: list[str] = field(default_factory=list)

@dataclass(slots=True)
class RuntimeFailure:
    stage: str
    error_type: str
    exception_type: str
    message: str
    retryable: bool | None = None
    step: int = 0
    tool_calls: int = 0
    evidence_count: int = 0
    last_action: dict[str, Any] | None = None

@dataclass
class AgentState:
    task: TaskSpec
    step: int = 0
    tool_calls: int = 0
    status: str = "running"
    actions: list[ActionProposal] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    report: DiagnosisReport | None = None
    errors: list[str] = field(default_factory=list)
    invalid_routes: int = 0
    recovered_routes: int = 0
    repeated_actions: int = 0
    reflection_count: int = 0
    started_at: float = field(default_factory=time.time)
    failure: RuntimeFailure | None = None
    report_source: str = ""
    current_hypothesis: dict[str, Any] = field(default_factory=dict)
    termination_advisory: str = ""
    observation_reuse_count: int = 0
    no_progress_count: int = 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "step": self.step, "tool_calls": self.tool_calls, "status": self.status,
            "evidence": len(self.evidence), "hypotheses": len(self.hypotheses),
            "invalid_routes": self.invalid_routes, "recovered_routes": self.recovered_routes,
            "repeated_actions": self.repeated_actions, "reflection_count": self.reflection_count,
            "errors": self.errors[-5:],
            "failure": asdict(self.failure) if self.failure else None,
            "report_source": self.report_source, "observation_reuse_count": self.observation_reuse_count,
            "no_progress_count": self.no_progress_count, "current_hypothesis": self.current_hypothesis,
        }
