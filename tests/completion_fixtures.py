"""Explicit completion contracts for deterministic test scenarios."""
from debug_assistant.incidents.contracts import CompletionContract, FactRequirement


def completion(*slots: str, role: str = "OBSERVATION") -> CompletionContract:
    return CompletionContract(required_facts=tuple(
        FactRequirement(slot=slot, description=f"Establish {slot}", role=role)
        for slot in slots
    ))
