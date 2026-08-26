from __future__ import annotations
from typing import Any, Literal, Type
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class AgentActionContract(StrictModel):
    kind: Literal["tool", "reflect", "finish"]
    skill: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_evidence: str = ""
    information_need: str = ""
    retain_context_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_kind_fields(self):
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool is required when kind='tool'")
        if self.kind != "tool" and self.tool is not None:
            raise ValueError("tool must be null unless kind='tool'")
        return self

class ReflectionContract(StrictModel):
    decision: Literal["continue", "finish"]
    reason: str = Field(min_length=1)
    current_diagnosis: str = ""
    evidence_sufficient: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_next_goal: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis_changed: bool | None = None

class ChangePoint(StrictModel):
    file: str = ""
    symbol: str = ""
    reason: str = ""

class DiagnosisReportContract(StrictModel):
    summary: str = ""
    root_cause: str = ""
    likely_files: list[str] = Field(default_factory=list)
    likely_symbols: list[str] = Field(default_factory=list)
    impact_scope: list[str] = Field(default_factory=list)
    recommended_change_points: list[ChangePoint] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def render_contract(model: Type[BaseModel], title: str) -> str:
    """Single source of truth: the same Pydantic model drives prompt and parser."""
    schema=model.model_json_schema()
    return f"{title}: Return exactly one JSON object valid under this JSON Schema. Do not add fields not present in the schema.\n{json.dumps(schema,ensure_ascii=False,separators=(',',':'))}"


def compact_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    out=[]
    for err in exc.errors(include_url=False):
        out.append({
            "location": ".".join(str(x) for x in err.get("loc", ())),
            "message": err.get("msg", "validation error"),
            "type": err.get("type", "validation_error"),
        })
    return out
