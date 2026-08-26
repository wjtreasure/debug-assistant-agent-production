from __future__ import annotations
from pydantic import ValidationError
from debug_assistant.contracts import ReflectionContract, compact_validation_error, render_contract

SYSTEM="""You are a critical reviewer for a read-only debugging agent. Detect goal drift, premature certainty, unsupported claims, contradictory evidence and missing falsification. Do not invent evidence. Explicitly state the strongest current diagnosis and whether existing evidence is sufficient to support a specific mechanism and location. Evidence IDs must come from the context."""

class Reflector:
    def __init__(self,llm,model=''): self.llm=llm; self.model=model
    def review(self,context:str):
        contract=render_contract(ReflectionContract,"REFLECTION_SCHEMA")
        user=f"""{context}\n\n{contract}\nChoose finish only if repository evidence supports a specific mechanism and location. hypothesis_changed is your own assessment and is telemetry only; the Harness independently fingerprints hypothesis state."""
        data=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        try:
            return ReflectionContract.model_validate(data).model_dump()
        except ValidationError as exc:
            raise ValueError(f"reflection schema validation failed: {compact_validation_error(exc)}") from exc
