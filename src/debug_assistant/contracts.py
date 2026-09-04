from __future__ import annotations
from typing import Any, Literal, Type
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class InformationNeedContract(StrictModel):
    target: str | None = None
    question_type: Literal["location", "behavior", "causality", "caller", "test", "history", "contradiction"] | None = None
    evidence_goal: str | None = None

    @field_validator("question_type", mode="before")
    @classmethod
    def canonicalize_question_type(cls, value):
        if value is None:
            return None
        raw=str(value).strip().lower().replace("-","_").replace(" ","_")
        aliases={
            "implementation":"behavior", "definition":"location", "where":"location", "symbol":"location",
            "mechanism":"causality", "cause":"causality", "root_cause":"causality", "why":"causality",
            "verification":"behavior", "validation":"behavior", "check":"behavior",
            "callsite":"caller", "call_site":"caller", "call_path":"caller", "callee":"caller",
            "tests":"test", "testing":"test",
            "history_diff":"history", "git_history":"history",
            "falsification":"contradiction", "conflict":"contradiction",
        }
        return aliases.get(raw,raw)


class PlannerIntent(StrictModel):
    """Optional semantic metadata accompanying native tool calls.

    This is deliberately separate from the execution contract: invalid intent metadata
    must never prevent otherwise valid provider-native tool calls from being executed.
    """
    information_need: str | None = None
    target: str | None = None
    question_type: Literal["location", "behavior", "causality", "caller", "test", "history", "contradiction"] | None = None
    evidence_goal: str | None = None
    reason: str | None = None

class ParallelChildActionContract(StrictModel):
    action_id: str | None = None
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_obligation_ids: list[str] = Field(default_factory=list)

class AgentActionContract(StrictModel):
    kind: Literal["tool", "parallel", "reflect", "finish"]
    skill: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_evidence: str = ""
    information_need: str = ""
    information_need_structured: InformationNeedContract | None = None
    retain_context_ids: list[str] = Field(default_factory=list)
    actions: list[ParallelChildActionContract] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_kind_fields(self):
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool is required when kind='tool'")
        if self.kind == "parallel":
            if self.tool is not None:
                raise ValueError("tool must be null when kind='parallel'")
            if len(self.actions) < 2:
                raise ValueError("parallel actions requires at least 2 child actions")
        elif self.actions:
            raise ValueError("actions are only allowed when kind='parallel'")
        if self.kind not in {"tool", "parallel"} and self.tool is not None:
            raise ValueError("tool must be null unless kind='tool'")
        return self

QuestionType = Literal["location", "behavior", "causality", "caller", "test", "history", "contradiction"]

class EvidenceRequirement(StrictModel):
    """Structured evidence requirement emitted by Reflection.

    ``goal_type`` is optional only for backward compatibility with V1.4.3/older
    model outputs. New reflections should populate it explicitly; the Harness may
    conservatively infer it when absent.
    """
    target: str = Field(min_length=1)
    location: str | None = None
    file: str | None = None
    symbol: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    goal_type: QuestionType | None = None
    reason: str = Field(min_length=1)
    information_need_root_id: str | None = None

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.line_start is None) ^ (self.line_end is None):
            raise ValueError("line_start and line_end must be provided together")
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self

    @field_validator("goal_type", mode="before")
    @classmethod
    def canonicalize_goal_type(cls, value):
        if value is None:
            return None
        raw=str(value).strip().lower().replace("-","_").replace(" ","_")
        aliases={
            "implementation":"behavior", "definition":"location", "where":"location", "symbol":"location",
            "mechanism":"causality", "cause":"causality", "root_cause":"causality", "why":"causality",
            "verification":"behavior", "validation":"behavior", "check":"behavior",
            "callsite":"caller", "call_site":"caller", "call_path":"caller", "callee":"caller",
            "tests":"test", "testing":"test",
            "history_diff":"history", "git_history":"history",
            "falsification":"contradiction", "conflict":"contradiction",
        }
        return aliases.get(raw,raw)

# Backward-compatible import name used by older code/tests.
MissingEvidenceItem = EvidenceRequirement


class ObligationReview(StrictModel):
    obligation_id: str = Field(min_length=1)
    decision: Literal["resolved", "still_open", "refine"]
    reason: str = Field(min_length=1)
    refined_requirement: EvidenceRequirement | None = None

    @model_validator(mode="after")
    def validate_refinement(self):
        if self.decision == "refine" and self.refined_requirement is None:
            raise ValueError("refined_requirement is required when decision='refine'")
        if self.decision != "refine" and self.refined_requirement is not None:
            raise ValueError("refined_requirement is only allowed when decision='refine'")
        return self

class ReflectionContract(StrictModel):
    @model_validator(mode="before")
    @classmethod
    def _legacy_reflection_fields(cls, data):
        if isinstance(data, dict):
            data=dict(data)
            if "required_missing_evidence" not in data and "missing" in data:
                data["required_missing_evidence"]=[{"target":str(x),"location":None,"reason":"legacy reflection field"} for x in (data.pop("missing") or [])]
            else:
                data.pop("missing",None)
            # `contradictions` was legacy free text; it never represented evidence IDs.
            data.pop("contradictions",None)
        return data

    decision: Literal["continue", "finish"]
    reason: str = Field(min_length=1)
    current_diagnosis: str = ""
    root_cause_target: str | None = None
    root_cause_location: str | None = None
    root_cause_mechanism: str | None = None
    evidence_sufficient: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    required_missing_evidence: list[EvidenceRequirement] = Field(default_factory=list, max_length=3)
    optional_validation: list[EvidenceRequirement] = Field(default_factory=list, max_length=2)
    obligation_reviews: list[ObligationReview] = Field(default_factory=list, max_length=5)
    recommended_next_goal: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis_changed: bool | None = None

    @model_validator(mode="after")
    def unique_obligation_reviews(self):
        ids=[x.obligation_id for x in self.obligation_reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation_reviews must contain unique obligation_id values")
        return self


class ReflectionDecision(StrictModel):
    """Typed semantic input to the reducer.

    Derived runtime facts (status, gaps, sufficiency and revisions) deliberately do
    not belong here; the Harness computes them from evidence and obligations.
    """
    decision: Literal["continue", "finish"] = "continue"
    diagnosis: str = ""
    root_cause_target: str | None = None
    root_cause_location: str | None = None
    root_cause_mechanism: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    obligation_reviews: list[ObligationReview] = Field(default_factory=list, max_length=5)
    new_requirements: list[EvidenceRequirement] = Field(default_factory=list, max_length=3)
    optional_validation: list[EvidenceRequirement] = Field(default_factory=list, max_length=2)
    recommended_next_goal: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="after")
    def unique_review_ids(self):
        ids = [x.obligation_id for x in self.obligation_reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation_reviews must contain unique obligation_id values")
        return self

class ReportClaim(StrictModel):
    text: str = Field(min_length=1)
    claim_type: Literal["source_fact", "causal_inference", "diagnosis"]
    status: Literal["observed", "supported_inference", "supported", "hypothesis", "acquired_unreviewed", "inferred"] = "hypothesis"
    evidence_ids: list[str] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)
    file: str | None = None
    symbol: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

class ChangePoint(StrictModel):
    file: str = ""
    symbol: str = ""
    reason: str = ""

class DiagnosisReportContract(StrictModel):
    summary: str = ""
    root_cause: str = ""
    likely_files: list[str] = Field(default_factory=list)
    likely_symbols: list[str] = Field(default_factory=list)
    likely_file_source: Literal["llm", "hypothesis", "partial_hypothesis", "evidence_fallback"] = "llm"
    impact_scope: list[str] = Field(default_factory=list)
    recommended_change_points: list[ChangePoint] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    claims: list[ReportClaim] = Field(default_factory=list, max_length=12)


def render_contract(model: Type[BaseModel], title: str) -> str:
    """Single source of truth: the same Pydantic model drives prompt and parser."""
    schema=model.model_json_schema()
    return f"{title}: Return exactly one JSON object valid under this JSON Schema. Do not add fields not present in the schema.\n{json.dumps(schema,ensure_ascii=False,separators=(',',':'))}"



def render_contract_compact(model: Type[BaseModel], title: str) -> str:
    """Compact prompt rendering derived from the same Pydantic SSOT.

    Runtime parsing/validation still uses the Pydantic model directly; this only removes
    JSON-Schema boilerplate from the model-facing prompt.
    """
    schema=model.model_json_schema()
    props=schema.get("properties",{})
    required=set(schema.get("required",[]))
    rows=[]
    for name,meta in props.items():
        enum=meta.get("enum")
        anyof=meta.get("anyOf") or []
        typ=meta.get("type")
        if enum:
            type_text="|".join(map(str,enum))
        elif anyof:
            vals=[]
            for part in anyof:
                if part.get("enum"): vals.extend(map(str,part["enum"]))
                elif part.get("type"): vals.append(part["type"])
                elif part.get("$ref"): vals.append(part["$ref"].split("/")[-1])
            type_text="|".join(dict.fromkeys(vals)) or "value"
        elif meta.get("$ref"):
            type_text=meta["$ref"].split("/")[-1]
        elif typ=="array":
            item=meta.get("items",{})
            inner=item.get("$ref","").split("/")[-1] or item.get("type","value")
            type_text=f"list[{inner}]"
        else:
            type_text=typ or "value"
        bounds=[]
        for k,label in (("minItems","min"),("maxItems","max")):
            if k in meta: bounds.append(f"{label}={meta[k]}")
        opt="" if name in required else "?"
        rows.append(f"  {name}{opt}: {type_text}" + (f" ({', '.join(bounds)})" if bounds else ""))
    defs=schema.get("$defs",{})
    def_rows=[]
    for dname,dmeta in defs.items():
        dprops=dmeta.get("properties",{})
        if not dprops: continue
        dreq=set(dmeta.get("required",[]))
        fields=[]
        for fname,fmeta in dprops.items():
            ftyp=fmeta.get("type") or (fmeta.get("$ref","").split("/")[-1] if fmeta.get("$ref") else "value")
            fields.append(f"{fname}{'' if fname in dreq else '?'}:{ftyp}")
        def_rows.append(f"  {dname} = {{{', '.join(fields)}}}")
    tail=("\n"+"\n".join(def_rows)) if def_rows else ""
    return f"{title}: Return exactly one JSON object with only these fields. Required fields omit '?'.\n{{\n"+"\n".join(rows)+"\n}"+tail

def compact_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    out=[]
    for err in exc.errors(include_url=False):
        out.append({
            "location": ".".join(str(x) for x in err.get("loc", ())),
            "message": err.get("msg", "validation error"),
            "type": err.get("type", "validation_error"),
        })
    return out
