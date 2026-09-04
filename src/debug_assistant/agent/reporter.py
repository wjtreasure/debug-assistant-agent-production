from __future__ import annotations
from dataclasses import asdict, is_dataclass
import copy
import json
import re
import time
from typing import Any
from pydantic import ValidationError
from debug_assistant.models import DiagnosisReport
from debug_assistant.contracts import DiagnosisReportContract, ReportClaim, compact_validation_error, render_contract, render_contract_compact
from debug_assistant.llm.base import complete_json_compat
from debug_assistant.security.redaction import redact_sensitive

SYSTEM="""You are a senior software debugging assistant in FINAL REPORTING stage.
Produce a development decision report from the supplied structured hypothesis and evidence.
The investigation phase is complete. You MUST NOT request, call, or suggest executing any tool.
A tool-like response such as <read_file ...>, <grep ...>, a function_call, tool_call, or AgentAction is invalid in this stage.
Use only the supplied hypothesis, evidence summaries, and source projections. If evidence is incomplete, record that under uncertainties or next_checks.
Do not claim code was changed. Do not invent file names, symbols, line numbers or causal facts absent from evidence. Separate uncertainty from conclusions. For important statements, prefer structured claims with claim_type and evidence_ids; the Harness will deterministically derive final claim status.
evidence_ids must contain only IDs explicitly available in the supplied context. Valid claim_type values are source_fact, causal_inference, diagnosis. Valid status values are observed, supported_inference, supported, hypothesis, acquired_unreviewed, inferred. Never invent enum values or evidence IDs. acquired_unreviewed cannot become supported, and a partial hypothesis cannot become a definitive diagnosis. Return ONLY one valid DiagnosisReport JSON object matching the supplied contract."""

REPAIR_SYSTEM="""You are formatting an existing grounded report. You are NOT performing new diagnosis.
Change only field shape, an invalid enum into an allowed enum, removal of an invalid
claim, removal of an unknown evidence_id, or missing optional fields. Do not add a
claim, evidence, file, symbol, root cause, certainty, or semantic explanation.
Never upgrade hypothesis or acquired_unreviewed to supported. If a claim cannot be
repaired without adding semantic content, drop the claim. Return exactly one
DiagnosisReport JSON object and nothing else."""

_REPORT_FIELDS = {
    "summary", "root_cause", "likely_files", "likely_symbols", "likely_file_source",
    "impact_scope", "recommended_change_points", "uncertainties", "next_checks",
    "evidence_ids", "confidence", "claims",
}


def _safe_repair_skeleton(value: Any, *, depth: int = 0) -> Any:
    """Keep the bounded repair prompt structural and free of raw durable output."""
    value = redact_sensitive(value)
    if depth > 4:
        return str(value)[:1000]
    if isinstance(value, dict):
        return {str(k): _safe_repair_skeleton(v, depth=depth + 1)
                for k, v in value.items() if depth or k in _REPORT_FIELDS}
    if isinstance(value, list):
        return [_safe_repair_skeleton(x, depth=depth + 1) for x in value[:12]]
    if isinstance(value, str):
        return value[:4000]
    return value


def _claim_count(raw: Any) -> int:
    return len(raw.get("claims") or []) if isinstance(raw, dict) and isinstance(raw.get("claims"), list) else 0


def _invalid_claim_count(errors: list[dict[str, Any]]) -> int:
    indexes = set()
    for error in errors:
        loc = error.get("location") or ""
        match = re.match(r"claims\.(\d+)", str(loc))
        if match:
            indexes.add(match.group(1))
    return len(indexes)


def _schema_enum(model, field: str) -> tuple[Any, ...]:
    prop=(model.model_json_schema().get('properties') or {}).get(field,{})
    if prop.get('enum'):
        return tuple(prop['enum'])
    return tuple(value for part in prop.get('anyOf',[]) for value in part.get('enum',[]))


_CLAIM_TYPES=frozenset(_schema_enum(ReportClaim,'claim_type'))
_CLAIM_STATUSES=frozenset(_schema_enum(ReportClaim,'status'))
_CLAIM_IDENTITY_FIELDS=frozenset({'text','evidence_ids','obligation_ids','file','symbol','line_start','line_end'})


def _constrain_repaired_report(primary: Any, repaired: Any, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply one bounded report repair without accepting semantic additions."""
    if not isinstance(primary,dict) or not isinstance(repaired,dict):
        return {}, [{'claim_index':None,'reason':'invalid_repair_shape'}]
    invalid_locations={str(x.get('location') or '') for x in errors}
    result=copy.deepcopy(primary)
    dropped=[]
    # Optional fields can be supplied when the primary omitted them. Existing
    # values are authoritative except for a field explicitly reported invalid.
    for field in _REPORT_FIELDS - {'claims'}:
        if field not in primary and field in repaired:
            result[field]=copy.deepcopy(repaired[field])
        elif field in repaired and field in invalid_locations and field == 'likely_file_source':
            result[field]=copy.deepcopy(repaired[field])

    primary_claims=primary.get('claims') if isinstance(primary.get('claims'),list) else []
    repaired_claims=repaired.get('claims') if isinstance(repaired.get('claims'),list) else []
    kept=[]
    for index, original in enumerate(primary_claims):
        if not isinstance(original,dict):
            dropped.append({'claim_index':index,'reason':'invalid_primary_claim'})
            continue
        if index >= len(repaired_claims) or not isinstance(repaired_claims[index],dict):
            dropped.append({'claim_index':index,'reason':'claim_not_repaired'})
            continue
        candidate=repaired_claims[index]
        if any(candidate.get(field)!=original.get(field) for field in _CLAIM_IDENTITY_FIELDS):
            dropped.append({'claim_index':index,'reason':'semantic_change_attempted'})
            continue
        merged=copy.deepcopy(original)
        if original.get('claim_type') not in _CLAIM_TYPES and candidate.get('claim_type') in _CLAIM_TYPES:
            merged['claim_type']=candidate['claim_type']
        if original.get('status') not in _CLAIM_STATUSES and candidate.get('status') in _CLAIM_STATUSES:
            merged['status']=candidate['status']
        # A format repair cannot upgrade an already valid status/type or add text.
        if original.get('status') in _CLAIM_STATUSES and candidate.get('status') != original.get('status'):
            merged['status']=original.get('status')
        if original.get('claim_type') in _CLAIM_TYPES and candidate.get('claim_type') != original.get('claim_type'):
            merged['claim_type']=original.get('claim_type')
        if 'evidence_ids' not in merged: merged['evidence_ids']=[]
        if 'obligation_ids' not in merged: merged['obligation_ids']=[]
        kept.append(merged)
    for index in range(len(primary_claims),len(repaired_claims)):
        dropped.append({'claim_index':index,'reason':'new_claim_forbidden'})
    result['claims']=kept
    return result,dropped


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


def _candidate_file_ranking(evidence) -> tuple[list[str],dict[str,dict]]:
    stats={}
    for ev in evidence:
        if ev.source != 'read_file' or not ev.file:
            continue
        st=stats.setdefault(ev.file,{'source_evidence_count':0,'need_ids':set(),'observation_ids':set(),'exact_location_hits':0,'last_evidence_gain_step':0})
        st['source_evidence_count']+=1
        if ev.raw_observation_id: st['observation_ids'].add(ev.raw_observation_id)
        for tag in ev.tags or []:
            if tag.startswith('need:'): st['need_ids'].add(tag.split(':',1)[1])
            elif tag.startswith('step:'):
                try: st['last_evidence_gain_step']=max(st['last_evidence_gain_step'],int(tag.split(':',1)[1]))
                except ValueError: pass
            elif tag=='exact_location': st['exact_location_hits']+=1
    ordered=sorted(stats.items(),key=lambda kv:(
        kv[1]['source_evidence_count'],len(kv[1]['need_ids']),len(kv[1]['observation_ids']),kv[1]['exact_location_hits'],kv[1]['last_evidence_gain_step']
    ),reverse=True)
    serial={path:{**st,'need_ids':sorted(st['need_ids']),'observation_ids':sorted(st['observation_ids'])} for path,st in ordered}
    return [path for path,_ in ordered],serial


def build_finalization_context(*, task_id: str, issue: str, state_summary: dict, hypothesis,
                               evidence, observation_store=None, default_source_projections: int=2,
                               hard_source_projection_cap: int=3, per_projection_chars: int=9000,
                               max_evidence_per_file: int=3, max_trace_summary_items: int=12,
                               max_snippet_lines: int=120,
                               max_context_chars: int | None = None) -> tuple[str,dict]:
    """Build Reporter-only context independent of Planner/Reflection Active/Cold state.

    If a supported hypothesis does not exist, source-backed evidence is still projected
    conservatively so finalization can report candidate files rather than pretending the
    repository was never inspected.
    """
    hyp=_hypothesis_dict(hypothesis)
    by_id={e.evidence_id:e for e in evidence}
    support=[by_id[x] for x in (hyp.get("supporting_evidence_ids") or []) if x in by_id]
    contradict=[by_id[x] for x in (hyp.get("contradicting_evidence_ids") or []) if x in by_id]
    candidate_files,candidate_stats=_candidate_file_ranking(evidence)
    fallback_source=[]
    if not support:
        # One immutable source observation contributes once even if it was rehydrated later.
        seen_obs=set()
        file_order={p:i for i,p in enumerate(candidate_files)}
        candidates=[]
        for ev in evidence:
            if ev.source!='read_file' or not ev.file: continue
            key=ev.raw_observation_id or ev.evidence_id
            if key in seen_obs: continue
            seen_obs.add(key); candidates.append(ev)
        fallback_source=sorted(candidates,key=lambda e:(file_order.get(e.file,10**6),-(next((int(t.split(':',1)[1]) for t in e.tags if t.startswith('step:') and t.split(':',1)[1].isdigit()),0))))

    def cap_per_file(rows, limit):
        counts={}; out=[]
        for ev in rows:
            key=ev.file or ev.source
            if counts.get(key,0) >= max(1,int(limit)):
                continue
            counts[key]=counts.get(key,0)+1; out.append(ev)
        return out

    support=cap_per_file(support, max_evidence_per_file)
    contradict=cap_per_file(contradict, max_evidence_per_file)
    fallback_source=cap_per_file(fallback_source, max_evidence_per_file)
    fallback_source=fallback_source[:max(1,int(max_trace_summary_items))]

    def compact(ev):
        loc=ev.file or ev.source
        if ev.source_start_line is not None and ev.source_end_line is not None:
            loc=f"{loc}:{ev.source_start_line}-{ev.source_end_line}"
        return f"- [{ev.evidence_id}] {ev.kind} {loc}: {ev.summary}"

    projection_pool=support or fallback_source
    root_location=str(hyp.get("root_cause_location") or "")
    ranked=[]
    for idx,ev in enumerate(projection_pool):
        root_match=0 if (root_location and ev.file and (ev.file==root_location or root_location.endswith('/'+ev.file) or ev.file.endswith('/'+root_location))) else 1
        ranked.append((root_match,idx,ev))
    ranked.sort(key=lambda x:(x[0],x[1]))
    limit=max(0,min(default_source_projections,hard_source_projection_cap))
    if len(ranked)>limit and limit < hard_source_projection_cap:
        first_files={x[2].file for x in ranked[:limit] if x[2].file}
        remaining_root=[x for x in ranked[limit:] if x[0]==0 and x[2].file not in first_files]
        if remaining_root: limit+=1

    projections=[]
    obs_get=getattr(observation_store,"get",lambda _id: None) if observation_store is not None else (lambda _id: None)
    for _,_,ev in ranked[:limit]:
        body=""
        obs=obs_get(ev.raw_observation_id) if ev.raw_observation_id else None
        if obs is not None and getattr(obs,"content",None): body=_line_safe_head_tail(obs.content,per_projection_chars)
        elif ev.excerpt: body=_line_safe_head_tail(ev.excerpt,per_projection_chars)
        if not body: continue
        body_lines=body.splitlines()
        if len(body_lines) > max(1,int(max_snippet_lines)):
            body="\n".join(body_lines[:max(1,int(max_snippet_lines))]) + "\n...[snippet lines omitted]"
        loc=ev.file or ev.source
        projections.append(f"SOURCE_PROJECTION evidence_id={ev.evidence_id} location={loc}\n{body}\nEND_SOURCE_PROJECTION")

    missing=hyp.get("required_missing_evidence") or []
    optional=(hyp.get("optional_validation") or [])[:max(1,int(max_trace_summary_items))]
    fallback_summary="\n".join(compact(e) for e in fallback_source) or "(none)"
    text=(
        f"FINALIZATION_TASK_ID: {task_id}\n"
        f"ISSUE:\n{issue}\n\n"
        f"FINAL_RUNTIME_STATE: {json.dumps(state_summary,ensure_ascii=False,default=str)}\n\n"
        f"COMPLETE_HYPOTHESIS:\n{json.dumps(hyp,ensure_ascii=False,default=str)}\n\n"
        "SUPPORTING_EVIDENCE_SUMMARIES:\n" + ("\n".join(compact(e) for e in support) or "(none)") + "\n\n"
        "SOURCE_EVIDENCE_FALLBACK_SUMMARIES (use only when hypothesis support is absent; these imply candidate files, not confirmed root cause):\n" + fallback_summary + "\n\n"
        "CONTRADICTING_EVIDENCE_SUMMARIES:\n" + ("\n".join(compact(e) for e in contradict) or "(none)") + "\n\n"
        f"CANDIDATE_FILES_FROM_SOURCE_EVIDENCE: {json.dumps(candidate_files,ensure_ascii=False)}\n\n"
        f"REQUIRED_MISSING_EVIDENCE (report only as uncertainties/next_checks; do not investigate):\n{json.dumps(missing,ensure_ascii=False)}\n\n"
        f"OPTIONAL_VALIDATION (report only as uncertainties/next_checks when useful; do not investigate):\n{json.dumps(optional,ensure_ascii=False)}\n\n"
        "CORE_SOURCE_PROJECTIONS:\n" + ("\n\n".join(projections) or "(none)")
    )
    if max_context_chars is not None and len(text) > int(max_context_chars):
        marker="CORE_SOURCE_PROJECTIONS:\n"
        prefix, _, source_text=text.partition(marker)
        budget=max(0,int(max_context_chars)-len(prefix)-len(marker))
        text=prefix+marker+_line_safe_head_tail(source_text,budget) if budget else prefix+marker+"(source projections omitted by context budget)"
    telemetry={
        "reporter_projection_count":len(projections),"reporter_context_chars":len(text),
        "reporter_supporting_evidence_count":len(support),"reporter_source_fallback_evidence_count":len(fallback_source),
        "reporter_contradicting_evidence_count":len(contradict),"reporter_required_missing_count":len(missing),
        "reporter_optional_validation_count":len(optional),"known_context_included":False,
        "evidence_fallback_used":bool(not support and fallback_source),"fallback_candidate_files":candidate_files,
        "fallback_candidate_stats":candidate_stats,
        "max_reporter_context_chars":max_context_chars,
        "max_evidence_per_file":max_evidence_per_file,
        "max_snippet_lines":max_snippet_lines,
        "max_trace_summary_items":max_trace_summary_items,
    }
    return text,telemetry


class Reporter:
    def __init__(self,llm,model='',compact_prompt=False):
        self.llm=llm; self.model=model; self.compact_prompt=compact_prompt
        self.last_prompt_breakdown={}; self.last_contract_violation=None
        self.last_contract_diagnostics=None; self.last_events=[]; self.last_fallback_reason=None; self.repair_admission=None

    def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.last_events.append((event_type, dict(payload)))
        if event_type == 'REPORTER_FALLBACK_TRIGGERED':
            self.last_fallback_reason=payload.get('reason')

    @staticmethod
    def _ground_contract(contract, available: set[str]) -> tuple[Any, list[dict[str, Any]]]:
        """Remove unsupported report references without changing their meaning."""
        corrections=[]
        top_ids=list(contract.evidence_ids)
        valid_top=[x for x in top_ids if x in available]
        if valid_top != top_ids:
            corrections.append({'kind':'unknown_evidence_id_dropped','location':'evidence_ids','count':len(top_ids)-len(valid_top)})
            contract.evidence_ids=valid_top
        grounded_claims=[]
        for idx, claim in enumerate(contract.claims):
            ids=list(claim.evidence_ids)
            valid=[x for x in ids if x in available]
            if ids and not valid:
                corrections.append({'kind':'claim_dropped_unknown_evidence','claim_index':idx})
                continue
            if valid != ids:
                corrections.append({'kind':'unknown_claim_evidence_id_dropped','claim_index':idx,'count':len(ids)-len(valid)})
                claim.evidence_ids=valid
            grounded_claims.append(claim)
        contract.claims=grounded_claims
        return contract, corrections

    def build(self,task_id,context,evidence,logical_timeout_seconds:float|None=None):
        available={e.evidence_id for e in evidence}
        contract=(render_contract_compact(DiagnosisReportContract,"FINAL_REPORT_SCHEMA") if self.compact_prompt else render_contract(DiagnosisReportContract,"FINAL_REPORT_SCHEMA"))
        available_text=f"AVAILABLE_EVIDENCE_IDS: {sorted(available)}"
        user=f"{context}\n\n{available_text}\n\n{contract}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'evidence_id_chars':len(available_text),'contract_chars':len(contract),'repair_attempted':False}
        self.last_contract_violation=None; self.last_contract_diagnostics=None; self.last_events=[]; self.last_fallback_reason=None
        before_raw=getattr(self.llm,"last_raw_content",None)
        started=time.monotonic()
        def remaining_timeout():
            if logical_timeout_seconds is None:
                return None
            remaining=float(logical_timeout_seconds)-(time.monotonic()-started)
            return remaining if remaining > 0 else 0.0

        def validate(raw):
            try:
                return DiagnosisReportContract.model_validate(raw), None
            except ValidationError as exc:
                return None, exc

        try:
            raw=complete_json_compat(self.llm,SYSTEM,user,model=self.model or None,logical_timeout_seconds=logical_timeout_seconds)
        except Exception as exc:
            raw_text=getattr(self.llm,"last_raw_content",None)
            if raw_text and raw_text != before_raw:
                reason=detect_tool_action(raw_text)
                if reason:
                    self.last_contract_violation={"reason":reason}
                    raise ReporterContractViolation(f"reporter attempted tool action: {reason}") from exc
            raise
        raw_text=getattr(self.llm,"last_raw_content",None)
        reason=detect_tool_action(raw_text) if raw_text and raw_text != before_raw else detect_tool_action(raw)
        if reason:
            self.last_contract_violation={"reason":reason}
            raise ReporterContractViolation(f"reporter attempted tool action: {reason}")
        d, validation_error = validate(raw)
        if validation_error is not None:
            errors=compact_validation_error(validation_error)
            self.last_contract_diagnostics={'validation_errors':errors,'claim_count':_claim_count(raw),'invalid_claim_count':_invalid_claim_count(errors)}
            self._event('REPORTER_CONTRACT_INVALID',self.last_contract_diagnostics)
            repair_allowed=self.repair_admission() if callable(self.repair_admission) else True
            repair_timeout=remaining_timeout()
            if not repair_allowed or (repair_timeout is not None and repair_timeout <= 0):
                reason='repair_not_admitted' if not repair_allowed else 'stage_deadline_insufficient'
                self._event('REPORTER_FALLBACK_TRIGGERED',{'reason':reason})
                raise ValueError(f"report schema validation failed: {errors}") from validation_error
            self.last_prompt_breakdown['repair_attempted']=True
            repair_user=(
                'REPORTER_FORMAT_REPAIR\n'
                f'PRIMARY_REPORT_SKELETON:\n{json.dumps(_safe_repair_skeleton(raw),ensure_ascii=False,default=str)}\n\n'
                f'VALIDATION_ERRORS:\n{json.dumps(errors,ensure_ascii=False)}\n\n'
                f'ALLOWED_CLAIM_TYPE_VALUES: [{", ".join(sorted(_CLAIM_TYPES))}]\n'
                f'ALLOWED_CLAIM_STATUS_VALUES: [{", ".join(sorted(_CLAIM_STATUSES))}]\n'
                'Repair existing grounded claims only. Added claims, files, symbols, root causes, '
                'confidence increases, unsupported diagnoses, and evidence upgrades are forbidden.\n'
                f'{available_text}\n\n{contract}'
            )
            self._event('REPORTER_REPAIR_STARTED',{'claim_count':_claim_count(raw)})
            repair_started=time.monotonic()
            try:
                repaired=complete_json_compat(self.llm,REPAIR_SYSTEM,repair_user,model=self.model or None,logical_timeout_seconds=repair_timeout)
            except Exception:
                self._event('REPORTER_REPAIR_FINISHED',{'success':False,'elapsed_ms':(time.monotonic()-repair_started)*1000.0})
                self._event('REPORTER_FALLBACK_TRIGGERED',{'reason':'repair_failed'})
                raise
            constrained,dropped_claims=_constrain_repaired_report(raw,repaired,errors)
            for dropped in dropped_claims:
                self._event('REPORTER_REPAIR_DROPPED_CLAIM',dropped)
            repaired_d, repair_error=validate(constrained)
            self._event('REPORTER_REPAIR_FINISHED',{'success':repair_error is None,'elapsed_ms':(time.monotonic()-repair_started)*1000.0})
            if repair_error is not None:
                self._event('REPORTER_FALLBACK_TRIGGERED',{'reason':'contract_invalid_after_repair'})
                raise ValueError(f"report schema validation failed after one repair: {compact_validation_error(repair_error)}") from repair_error
            d=repaired_d
            self._event('REPORTER_CORRECTED',{'repair':'bounded_format_only'})

        d, grounding_corrections=self._ground_contract(d,available)
        if grounding_corrections:
            self._event('REPORTER_CORRECTED',{'grounding_corrections':grounding_corrections})
        source_files={e.file for e in evidence if e.file}
        grounded_files=[path for path in d.likely_files if path in source_files]
        if grounded_files != d.likely_files:
            self._event('REPORTER_CORRECTED',{'likely_files_dropped':len(d.likely_files)-len(grounded_files)})
            d.likely_files=grounded_files
        ev=[{"evidence_id":e.evidence_id,"source":e.source,"file":e.file,"line_start":e.line_start,"line_end":e.line_end,"summary":e.summary,
             "raw_observation_id":e.raw_observation_id,"source_start_line":e.source_start_line,"source_end_line":e.source_end_line,
             "excerpt_start_line":e.excerpt_start_line,"excerpt_end_line":e.excerpt_end_line,"excerpt_truncated":e.excerpt_truncated} for e in evidence]
        return DiagnosisReport(task_id=task_id,summary=d.summary,root_cause=d.root_cause,
            likely_files=d.likely_files,likely_symbols=d.likely_symbols,impact_scope=d.impact_scope,
            evidence=ev,recommended_change_points=[x.model_dump() for x in d.recommended_change_points],uncertainties=d.uncertainties,
            next_checks=d.next_checks,confidence=d.confidence,report_source="llm",evidence_ids=list(d.evidence_ids),likely_file_source=d.likely_file_source,claims=[x.model_dump() for x in d.claims])
