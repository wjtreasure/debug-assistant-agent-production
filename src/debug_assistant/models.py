from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any
import time, json

class ActionKind(str, Enum):
    TOOL = "tool"
    REFLECT = "reflect"
    FINISH = "finish"

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

@dataclass(slots=True)
class Evidence:
    evidence_id: str
    kind: str
    source: str
    summary: str
    excerpt: str = ""
    file: str | None = None
    line_start: int | None = None
    line_end: int | None = None
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

    def to_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "step": self.step, "tool_calls": self.tool_calls, "status": self.status,
            "evidence": len(self.evidence), "hypotheses": len(self.hypotheses),
            "invalid_routes": self.invalid_routes, "recovered_routes": self.recovered_routes,
            "repeated_actions": self.repeated_actions, "reflection_count": self.reflection_count,
            "errors": self.errors[-5:],
        }
