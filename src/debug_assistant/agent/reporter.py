from __future__ import annotations
from pydantic import ValidationError
from debug_assistant.models import DiagnosisReport
from debug_assistant.contracts import DiagnosisReportContract, compact_validation_error, render_contract
SYSTEM="""You are a senior software debugging assistant. Produce a development decision report from the supplied evidence. Do not claim code was changed. Do not invent file names, symbols, line numbers or causal facts absent from evidence. Separate uncertainty from conclusions. evidence_ids must contain only IDs explicitly available in the supplied context."""

class Reporter:
    def __init__(self,llm,model=''): self.llm=llm; self.model=model; self.last_prompt_breakdown={}
    def build(self,task_id,context,evidence):
        available={e.evidence_id for e in evidence}
        contract=render_contract(DiagnosisReportContract,"FINAL_REPORT_SCHEMA")
        available_text=f"AVAILABLE_EVIDENCE_IDS: {sorted(available)}"
        user=f"{context}\n\n{available_text}\n\n{contract}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'evidence_id_chars':len(available_text),'contract_chars':len(contract)}
        raw=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        try:
            d=DiagnosisReportContract.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"report schema validation failed: {compact_validation_error(exc)}") from exc
        unknown=[x for x in d.evidence_ids if x not in available]
        if unknown:
            raise ValueError(f"report validation failed: unknown evidence_ids: {unknown}")
        selected_ids=set(d.evidence_ids)
        # Keep report evidence complete for backward compatibility, while explicit evidence_ids preserve report-level provenance.
        ev=[{"evidence_id":e.evidence_id,"source":e.source,"file":e.file,"line_start":e.line_start,"line_end":e.line_end,"summary":e.summary,
             "raw_observation_id":e.raw_observation_id,"source_start_line":e.source_start_line,"source_end_line":e.source_end_line,
             "excerpt_start_line":e.excerpt_start_line,"excerpt_end_line":e.excerpt_end_line,"excerpt_truncated":e.excerpt_truncated} for e in evidence]
        return DiagnosisReport(task_id=task_id,summary=d.summary,root_cause=d.root_cause,
            likely_files=d.likely_files,likely_symbols=d.likely_symbols,impact_scope=d.impact_scope,
            evidence=ev,recommended_change_points=[x.model_dump() for x in d.recommended_change_points],uncertainties=d.uncertainties,
            next_checks=d.next_checks,confidence=d.confidence,report_source="llm",evidence_ids=list(d.evidence_ids))
