from __future__ import annotations
from pydantic import ValidationError
from debug_assistant.contracts import ReflectionContract, compact_validation_error, render_contract

SYSTEM="""You are a critical reviewer for a read-only debugging agent. Detect goal drift, premature certainty, unsupported claims and direct falsifying evidence. Do not invent evidence. Explicitly state the strongest current diagnosis and whether repository evidence is sufficient to support a specific causal mechanism and location. Evidence IDs must come from the context.

Use required_missing_evidence very narrowly: it contains only facts without which the current causal diagnosis cannot reasonably be supported. Put useful but nonessential confirmation in optional_validation instead. Optional validation must not by itself block finalization once the causal diagnosis is supported.

Also emit a compact structured root-cause identity. root_cause_target should be the most specific causal symbol/component you currently believe is responsible (for example a function or class). root_cause_location should be a repository file path when known, preferably repo-root relative and without line numbers. root_cause_mechanism should be a concise mechanism statement. Keep these fields stable when the underlying diagnosis has not changed; do not rewrite them merely for style.

contradicting_evidence_ids contains only evidence that directly falsifies the proposed causal explanation. A buggy test expectation, an alternative implementation detail, incomplete information, missing validation, or a failing test consistent with the bug is NOT a contradiction unless it directly disproves the diagnosis. Keep the state concise."""

class Reflector:
    def __init__(self,llm,model=''):
        self.llm=llm; self.model=model; self.last_prompt_breakdown={}
    def review(self,context:str):
        contract=render_contract(ReflectionContract,"REFLECTION_SCHEMA")
        extra="Choose finish when repository evidence already supports a specific mechanism and location and no required causal gap remains. hypothesis_changed is telemetry only; the Harness independently fingerprints diagnostic state. root_cause_target/root_cause_location are the primary stability identity, while current_diagnosis may remain natural language. Return at most the most important required/optional items allowed by the schema."
        user=f"{context}\n\n{contract}\n{extra}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'contract_chars':len(contract),'instruction_chars':len(extra)}
        data=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        try:
            return ReflectionContract.model_validate(data).model_dump()
        except ValidationError as exc:
            raise ValueError(f"reflection schema validation failed: {compact_validation_error(exc)}") from exc
