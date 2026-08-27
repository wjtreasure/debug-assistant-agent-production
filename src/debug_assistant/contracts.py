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

class MissingEvidenceItem(StrictModel):
    target: str = Field(min_length=1)
    location: str | None = None
    reason: str = Field(min_length=1)

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
    root_cause_target: str = ""
    root_cause_location: str | None = None
    root_cause_mechanism: str = ""
    evidence_sufficient: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    required_missing_evidence: list[MissingEvidenceItem] = Field(default_factory=list, max_length=3)
    optional_validation: list[MissingEvidenceItem] = Field(default_factory=list, max_length=2)
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
