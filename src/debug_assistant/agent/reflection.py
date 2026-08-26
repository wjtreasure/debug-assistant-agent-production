from __future__ import annotations
SYSTEM="""You are a critical reviewer for a read-only debugging agent. Detect goal drift, premature certainty, unsupported claims, contradictory evidence and missing falsification. Do not invent evidence."""

class Reflector:
    def __init__(self,llm,model=''): self.llm=llm; self.model=model
    def review(self,context:str):
        user=f"""{context}\n\nREFLECTION_SCHEMA: Return JSON: {{"decision":"continue|finish","reason":"...","missing":["..."],"contradictions":["..."]}}. Choose finish only if evidence supports a specific mechanism and location."""
        return self.llm.complete_json(SYSTEM,user,model=self.model or None)
