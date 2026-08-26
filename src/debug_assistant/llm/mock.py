from __future__ import annotations
from typing import Any
from .base import LLMClient

class MockLLMClient(LLMClient):
    """Deterministic smoke-test provider. Production diagnosis must use a real model."""
    def __init__(self): self.n=0; self.calls=[]
    def complete_json(self, system: str, user: str, *, model: str | None=None) -> dict[str, Any]:
        self.n+=1
        self.calls.append({
            "model":"mock","prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
            "input_tokens":0,"output_tokens":0,"cached_tokens":None,"reasoning_tokens":None,
            "prompt_chars":len(system)+len(user),"completion_chars":0,"latency_ms":0.0,
        })
        low=user.lower()
        if 'final_report_schema' in low:
            return {"summary":"Smoke-test diagnosis", "root_cause":"Evidence suggests the fixture parser boundary check is the likely defect.",
                    "likely_files":["src/parser.py"], "likely_symbols":["parse_value"], "impact_scope":["parser callers"],
                    "recommended_change_points":[{"file":"src/parser.py","symbol":"parse_value","reason":"boundary handling"}],
                    "uncertainties":["Mock provider does not perform semantic reasoning"], "next_checks":["Review boundary tests"], "confidence":0.62}
        if 'reflection_schema' in low:
            return {"decision":"continue","reason":"collect one more targeted source excerpt","current_diagnosis":"fixture parser boundary validation",
                    "evidence_sufficient":False,"supporting_evidence_ids":[],"missing":["target implementation"],"contradictions":[],
                    "recommended_next_goal":"inspect target implementation"}
        if self.n == 1:
            return {"kind":"tool","skill":"repository_exploration","reason":"find issue keywords","confidence":0.8,
                    "tool":"grep","arguments":{"query":"parse_value|boundary|invalid","glob":"*.py","max_results":20},"expected_evidence":"candidate implementation"}
        if self.n == 2:
            return {"kind":"tool","skill":"hypothesis_validation","reason":"read candidate source","confidence":0.9,
                    "tool":"read_file","arguments":{"path":"src/parser.py","start_line":1,"end_line":120},"expected_evidence":"implementation details"}
        return {"kind":"finish","skill":"report_synthesis","reason":"enough evidence for smoke test","confidence":0.75,"tool":None,"arguments":{},"expected_evidence":""}
