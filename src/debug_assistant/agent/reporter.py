from __future__ import annotations
from dataclasses import asdict, is_dataclass
import json
import re
from pydantic import ValidationError
from debug_assistant.models import DiagnosisReport
from debug_assistant.contracts import DiagnosisReportContract, compact_validation_error, render_contract, render_contract_compact

SYSTEM="""You are a senior software debugging assistant in FINAL REPORTING stage.
Produce a development decision report from the supplied structured hypothesis and evidence.
The investigation phase is complete. You MUST NOT request, call, or suggest executing any tool.
A tool-like response such as <read_file ...>, <grep ...>, a function_call, tool_call, or AgentAction is invalid in this stage.
Use only the supplied hypothesis, evidence summaries, and source projections. If evidence is incomplete, record that under uncertainties or next_checks.
Do not claim code was changed. Do not invent file names, symbols, line numbers or causal facts absent from evidence. Separate uncertainty from conclusions.
evidence_ids must contain only IDs explicitly available in the supplied context. Return ONLY one valid DiagnosisReport JSON object matching the supplied contract."""


class ReporterContractViolation(ValueError):
    pass


_XML_TOOL_LINE_RE = re.compile(r"(?im)^\s*<(?:read_file|grep|search|code_search|symbol_search|repo_tree|git_log|git_show|discover_tests)\b[^>]*?/?>\s*$")
_CALL_MARKER_RE = re.compile(r"(?im)^\s*(?:function_call|tool_call)\s*[:=]")
_JSON_TOOL_KEYS = {"tool_call", "function_call"}


def detect_tool_action(value) -> str | None:
    """Detect outputs that are predominantly tool actions without flagging normal prose.

    Matching is intentionally conservative: XML-like tool tags must occupy their own line,
    marker forms must begin an independent line, or a parsed object must structurally look
    like an AgentAction/tool-call rather than a DiagnosisReport.
    """
    if isinstance(value, str):
        text=value.strip()
        if _XML_TOOL_LINE_RE.search(text):
            return "xml_tool_action"
        if _CALL_MARKER_RE.search(text):
            return "tool_call_marker"
        return None
    if isinstance(value, dict):
        keys=set(value)
        if keys & _JSON_TOOL_KEYS:
            return "structured_tool_call"
        kind=str(value.get("kind","")).lower()
        if kind in {"tool","reflect","finish"} and ("tool" in value or "skill" in value or "arguments" in value):
            return "agent_action_object"
    return None


def _hypothesis_dict(hypothesis) -> dict:
    if hypothesis is None:
        return {}
    if isinstance(hypothesis, dict):
        return dict(hypothesis)
    if is_dataclass(hypothesis):
        return asdict(hypothesis)
    if hasattr(hypothesis,"model_dump"):
        return hypothesis.model_dump()
    return dict(getattr(hypothesis,"__dict__",{}) or {})


def _line_safe_head_tail(text: str, max_chars: int) -> str:
    """Bound a source projection while retaining both head and tail line coverage."""
    if len(text) <= max_chars:
        return text
    lines=text.splitlines()
    half=max(256,max_chars//2-64)
    head=[]; used=0
    for line in lines:
        add=len(line)+(1 if head else 0)
        if used+add > half: break
        head.append(line); used+=add
    tail=[]; used=0
    for line in reversed(lines):
        add=len(line)+(1 if tail else 0)
        if used+add > half: break
        tail.append(line); used+=add
    tail=list(reversed(tail))
    return "\n".join(head+["...[middle omitted for reporter context]..."]+tail)


def build_finalization_context(*, task_id: str, issue: str, state_summary: dict, hypothesis,
                               evidence, observation_store=None, default_source_projections: int=2,
                               hard_source_projection_cap: int=3, per_projection_chars: int=9000) -> tuple[str,dict]:
    """Build Reporter-only context independent of Planner/Reflection Active/Cold state.

    Invariants:
    - no KnownContextIndex is rendered here;
    - complete structured hypothesis is present, including contradictions/missing evidence;
    - every supporting/contradicting evidence item gets a compact summary;
    - only a bounded set of core supporting source bodies is projected from immutable observations;
    - this is finalization context assembly, not planner rehydration and does not execute tools.
    """
    hyp=_hypothesis_dict(hypothesis)
    by_id={e.evidence_id:e for e in evidence}
    support=[by_id[x] for x in (hyp.get("supporting_evidence_ids") or []) if x in by_id]
    contradict=[by_id[x] for x in (hyp.get("contradicting_evidence_ids") or []) if x in by_id]

    def compact(ev):
        loc=ev.file or ev.source
        if ev.source_start_line is not None and ev.source_end_line is not None:
            loc=f"{loc}:{ev.source_start_line}-{ev.source_end_line}"
        return f"- [{ev.evidence_id}] {ev.kind} {loc}: {ev.summary}"

    root_location=str(hyp.get("root_cause_location") or "")
    ranked=[]
    for idx,ev in enumerate(support):
        root_match=0 if (root_location and ev.file and (ev.file==root_location or root_location.endswith('/'+ev.file) or ev.file.endswith('/'+root_location))) else 1
        ranked.append((root_match,idx,ev))
    ranked.sort(key=lambda x:(x[0],x[1]))
    # Default 2, allow 3 only when the first two cannot cover all distinct root-cause source files.
    limit=max(0,min(default_source_projections,hard_source_projection_cap))
    if len(ranked)>limit and limit < hard_source_projection_cap:
        first_files={x[2].file for x in ranked[:limit] if x[2].file}
        remaining_root=[x for x in ranked[limit:] if x[0]==0 and x[2].file not in first_files]
        if remaining_root:
            limit+=1

    projections=[]
    obs_get=getattr(observation_store,"get",lambda _id: None) if observation_store is not None else (lambda _id: None)
    for _,_,ev in ranked[:limit]:
        body=""
        obs=obs_get(ev.raw_observation_id) if ev.raw_observation_id else None
        if obs is not None and getattr(obs,"content",None):
            body=_line_safe_head_tail(obs.content,per_projection_chars)
        elif ev.excerpt:
            body=_line_safe_head_tail(ev.excerpt,per_projection_chars)
        if not body:
            continue
        loc=ev.file or ev.source
        projections.append(f"SOURCE_PROJECTION evidence_id={ev.evidence_id} location={loc}\n{body}\nEND_SOURCE_PROJECTION")

    missing=hyp.get("required_missing_evidence") or []
    optional=hyp.get("optional_validation") or []
    text=(
        f"FINALIZATION_TASK_ID: {task_id}\n"
        f"ISSUE:\n{issue}\n\n"
        f"FINAL_RUNTIME_STATE: {json.dumps(state_summary,ensure_ascii=False,default=str)}\n\n"
        f"COMPLETE_HYPOTHESIS:\n{json.dumps(hyp,ensure_ascii=False,default=str)}\n\n"
        "SUPPORTING_EVIDENCE_SUMMARIES:\n" + ("\n".join(compact(e) for e in support) or "(none)") + "\n\n"
        "CONTRADICTING_EVIDENCE_SUMMARIES:\n" + ("\n".join(compact(e) for e in contradict) or "(none)") + "\n\n"
        f"REQUIRED_MISSING_EVIDENCE (report only as uncertainties/next_checks; do not investigate):\n{json.dumps(missing,ensure_ascii=False)}\n\n"
        f"OPTIONAL_VALIDATION (report only as uncertainties/next_checks when useful; do not investigate):\n{json.dumps(optional,ensure_ascii=False)}\n\n"
        "CORE_SOURCE_PROJECTIONS:\n" + ("\n\n".join(projections) or "(none)")
    )
    telemetry={
        "reporter_projection_count":len(projections),
        "reporter_context_chars":len(text),
        "reporter_supporting_evidence_count":len(support),
        "reporter_contradicting_evidence_count":len(contradict),
        "reporter_required_missing_count":len(missing),
        "reporter_optional_validation_count":len(optional),
        "known_context_included":False,
    }
    return text,telemetry


class Reporter:
    def __init__(self,llm,model='',compact_prompt=False):
        self.llm=llm; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}; self.last_contract_violation=None

    def build(self,task_id,context,evidence):
        available={e.evidence_id for e in evidence}
        contract=(render_contract_compact(DiagnosisReportContract,"FINAL_REPORT_SCHEMA") if self.compact_prompt else render_contract(DiagnosisReportContract,"FINAL_REPORT_SCHEMA"))
        available_text=f"AVAILABLE_EVIDENCE_IDS: {sorted(available)}"
        user=f"{context}\n\n{available_text}\n\n{contract}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'evidence_id_chars':len(available_text),'contract_chars':len(contract)}
        self.last_contract_violation=None
        before_raw=getattr(self.llm,"last_raw_content",None)
        try:
            raw=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        except Exception as exc:
            raw_text=getattr(self.llm,"last_raw_content",None)
            if raw_text and raw_text != before_raw:
                reason=detect_tool_action(raw_text)
                if reason:
                    self.last_contract_violation={"reason":reason,"raw_excerpt":raw_text[:500]}
                    raise ReporterContractViolation(f"reporter attempted tool action: {reason}") from exc
            raise
        raw_text=getattr(self.llm,"last_raw_content",None)
        reason=detect_tool_action(raw_text) if raw_text and raw_text != before_raw else detect_tool_action(raw)
        if reason:
            excerpt=(raw_text[:500] if isinstance(raw_text,str) and raw_text != before_raw else str(raw)[:500])
            self.last_contract_violation={"reason":reason,"raw_excerpt":excerpt}
            raise ReporterContractViolation(f"reporter attempted tool action: {reason}")
        try:
            d=DiagnosisReportContract.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"report schema validation failed: {compact_validation_error(exc)}") from exc
        unknown=[x for x in d.evidence_ids if x not in available]
        if unknown:
            raise ValueError(f"report validation failed: unknown evidence_ids: {unknown}")
        ev=[{"evidence_id":e.evidence_id,"source":e.source,"file":e.file,"line_start":e.line_start,"line_end":e.line_end,"summary":e.summary,
             "raw_observation_id":e.raw_observation_id,"source_start_line":e.source_start_line,"source_end_line":e.source_end_line,
             "excerpt_start_line":e.excerpt_start_line,"excerpt_end_line":e.excerpt_end_line,"excerpt_truncated":e.excerpt_truncated} for e in evidence]
        return DiagnosisReport(task_id=task_id,summary=d.summary,root_cause=d.root_cause,
            likely_files=d.likely_files,likely_symbols=d.likely_symbols,impact_scope=d.impact_scope,
            evidence=ev,recommended_change_points=[x.model_dump() for x in d.recommended_change_points],uncertainties=d.uncertainties,
            next_checks=d.next_checks,confidence=d.confidence,report_source="llm",evidence_ids=list(d.evidence_ids))
