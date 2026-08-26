from __future__ import annotations
from debug_assistant.models import DiagnosisReport
SYSTEM="""You are a senior software debugging assistant. Produce a development decision report from the supplied evidence. Do not claim code was changed. Do not invent file names, symbols, line numbers or causal facts absent from evidence. Separate uncertainty from conclusions."""

class Reporter:
    def __init__(self,llm,model=''): self.llm=llm; self.model=model
    def build(self,task_id,context,evidence):
        user=f"""{context}\n\nFINAL_REPORT_SCHEMA: Return JSON with keys summary, root_cause, likely_files(list), likely_symbols(list), impact_scope(list), recommended_change_points(list objects with file/symbol/reason), uncertainties(list), next_checks(list), confidence(0..1). Evidence IDs available: {[e.evidence_id for e in evidence]}"""
        d=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        ev=[{"evidence_id":e.evidence_id,"source":e.source,"file":e.file,"line_start":e.line_start,"line_end":e.line_end,"summary":e.summary} for e in evidence]
        return DiagnosisReport(task_id=task_id,summary=str(d.get('summary','')),root_cause=str(d.get('root_cause','')),
            likely_files=list(d.get('likely_files') or []),likely_symbols=list(d.get('likely_symbols') or []),impact_scope=list(d.get('impact_scope') or []),
            evidence=ev,recommended_change_points=list(d.get('recommended_change_points') or []),uncertainties=list(d.get('uncertainties') or []),
            next_checks=list(d.get('next_checks') or []),confidence=max(0.0,min(1.0,float(d.get('confidence',0.5)))))
