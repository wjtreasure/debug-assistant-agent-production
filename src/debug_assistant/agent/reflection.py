from __future__ import annotations
import json
import time
from pydantic import ValidationError
from debug_assistant.contracts import ReflectionContract, ReflectionDecision, ObligationReview, compact_validation_error, render_contract, render_contract_compact
from debug_assistant.llm.base import LLMDeadlineExceeded, complete_json_compat

SYSTEM="""You are a critical reviewer for a read-only debugging agent. Detect goal drift, premature certainty, unsupported claims and direct falsifying evidence. Do not invent evidence. Explicitly state the strongest current diagnosis and whether repository evidence is sufficient to support a specific causal mechanism and location. Evidence IDs must come from the context.

Use required_missing_evidence very narrowly: it contains only facts without which the current causal diagnosis cannot reasonably be supported. Put useful but nonessential confirmation in optional_validation instead. Optional validation must not by itself block finalization once the causal diagnosis is supported.

Also emit a compact structured root-cause identity. root_cause_target should be the most specific causal symbol/component you currently believe is responsible. root_cause_location should be a repository file path when known. root_cause_mechanism should be a concise mechanism statement. If target or mechanism is not yet known, return null rather than an object, list, or invented string. A partial diagnosis is valid and should not be discarded merely because one structured field is unknown. Keep known fields stable when the underlying diagnosis has not changed.

contradicting_evidence_ids contains only evidence that directly falsifies the proposed causal explanation. A buggy test expectation, an alternative implementation detail, incomplete information, missing validation, or a failing test consistent with the bug is NOT a contradiction unless it directly disproves the diagnosis. Keep the state concise."""

REPAIR_SYSTEM="""Repair one invalid reflection JSON object. Preserve its meaning and evidence IDs. Change only fields required to satisfy the supplied schema. For unknown scalar root-cause fields use null, never an object/list or invented content. Return exactly one corrected JSON object and nothing else."""


class Reflector:
    def __init__(self,llm,model='',compact_prompt=False):
        self.llm=llm; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}; self.last_repair_attempted=False


    @staticmethod
    def _sanitize_individual_reviews(data):
        """Drop only schema-invalid ObligationReview rows; preserve the rest of Reflection.

        This lets Runtime build a valid-subset candidate transaction without making one
        malformed review erase independent valid reviews from the same model response.
        """
        if not isinstance(data,dict):
            return data,[]
        rows=data.get('obligation_reviews')
        if not isinstance(rows,list):
            return data,[]
        valid=[]; invalid=[]
        for idx,row in enumerate(rows):
            try:
                valid.append(ObligationReview.model_validate(row).model_dump())
            except ValidationError as exc:
                oid=str(row.get('obligation_id') or '') if isinstance(row,dict) else ''
                invalid.append({'obligation_id':oid,'index':idx,'reason':'schema_invalid','errors':compact_validation_error(exc)})
        out=dict(data); out['obligation_reviews']=valid
        return out,invalid

    def review(self,context:str,logical_timeout_seconds:float|None=None,on_attempt_started=None):
        contract=(render_contract_compact(ReflectionContract,"REFLECTION_SCHEMA") if self.compact_prompt else render_contract(ReflectionContract,"REFLECTION_SCHEMA"))
        extra="Choose finish when repository evidence already supports a specific mechanism and location and no required causal gap remains. hypothesis_changed is telemetry only; the Harness independently fingerprints diagnostic state. root_cause_target/root_cause_location are the primary stability identity, while current_diagnosis may remain natural language. Return at most the most important required/optional items allowed by the schema. For every required_missing_evidence and optional_validation item, populate goal_type explicitly whenever possible: location, behavior, causality, caller, test, history, or contradiction. Use history only for actual version/commit/diff evidence needs; the mere word regression does not make a behavioral requirement historical. Every EvidenceRequirement must be atomic: one verifiable source scope (one file+symbol or one exact range), never a free-text A-and-B multi-source requirement. OPEN_CRITICAL_EVIDENCE_OBLIGATIONS may include obligation_id values. When source for an existing obligation is physically present in this Reflection context, emit exactly one obligation_reviews entry for that obligation: resolved if the shown source answers it, still_open if the shown source is insufficient, or refine with a more precise refined_requirement when the real causal question is delegated elsewhere. Do not review an obligation whose source is not shown in this Reflection prompt."
        user=f"{context}\n\n{contract}\n{extra}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'contract_chars':len(contract),'instruction_chars':len(extra),'repair_attempted':False}
        started=time.monotonic()
        def remaining_timeout():
            if logical_timeout_seconds is None:
                return None
            remaining=float(logical_timeout_seconds)-(time.monotonic()-started)
            if remaining <= 0:
                raise LLMDeadlineExceeded("Reflection logical deadline exhausted before schema repair")
            return remaining
        data=complete_json_compat(self.llm,SYSTEM,user,model=self.model or None,logical_timeout_seconds=remaining_timeout(),on_attempt_started=on_attempt_started)
        self.last_repair_attempted=False
        data,invalid_reviews=self._sanitize_individual_reviews(data)
        try:
            out=ReflectionContract.model_validate(data).model_dump(); out['_invalid_obligation_reviews']=invalid_reviews; return out
        except ValidationError as exc:
            # One bounded repair is cheaper and safer than discarding the entire
            # hypothesis transition because a nullable scalar was returned as an object.
            self.last_repair_attempted=True
            self.last_prompt_breakdown['repair_attempted']=True
            repair_user=(f"INVALID_REFLECTION:\n{json.dumps(data,ensure_ascii=False,default=str)}\n\n"
                         f"VALIDATION_ERRORS:\n{json.dumps(compact_validation_error(exc),ensure_ascii=False)}\n\n{contract}")
            repaired=complete_json_compat(self.llm,REPAIR_SYSTEM,repair_user,model=self.model or None,logical_timeout_seconds=remaining_timeout())
            repaired,repair_invalid_reviews=self._sanitize_individual_reviews(repaired)
            try:
                out=ReflectionContract.model_validate(repaired).model_dump(); out['_invalid_obligation_reviews']=invalid_reviews+repair_invalid_reviews; return out
            except ValidationError as exc2:
                raise ValueError(f"reflection schema validation failed after one repair: {compact_validation_error(exc2)}") from exc2


class TypedReflection:
    """Provider-capability-aware semantic reflection; state derivation lives in Reducer."""
    def __init__(self, llm, model=""):
        self.llm, self.model = llm, model

    def review(self, context: str, *, logical_timeout_seconds=None, on_attempt_started=None) -> ReflectionDecision:
        from debug_assistant.llm.base import ProviderCapabilities
        caps = getattr(self.llm, "capabilities", ProviderCapabilities())
        schema_prompt = render_contract(ReflectionDecision, "REFLECTION_DECISION_SCHEMA")
        system = "Interpret repository evidence semantically. Never emit derived status, gaps, or sufficiency."
        user = f"{context}\n\n{schema_prompt}"
        if caps.json_schema and hasattr(self.llm, "complete_structured"):
            response = self.llm.complete_structured(system, user, schema=ReflectionDecision,
                                                    model=self.model or None,
                                                    logical_timeout_seconds=logical_timeout_seconds,
                                                    on_attempt_started=on_attempt_started)
            data = response.structured
            if data is None:
                from debug_assistant.llm.base import extract_json
                data = extract_json(response.content)
        else:
            data = complete_json_compat(self.llm, system, user, model=self.model or None,
                                        logical_timeout_seconds=logical_timeout_seconds,
                                        on_attempt_started=on_attempt_started)
        # Native typed reflection must have the same per-review fault isolation as
        # the legacy reflector. One malformed refine row must not discard otherwise
        # valid diagnosis/evidence input or turn a recoverable model defect into a
        # whole reflection failure.
        data, _invalid_reviews = Reflector._sanitize_individual_reviews(data)
        return ReflectionDecision.model_validate(data)
