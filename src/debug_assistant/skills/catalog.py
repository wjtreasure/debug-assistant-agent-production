from __future__ import annotations

from dataclasses import dataclass

from .loader import LocalSkillSource


@dataclass(frozen=True, slots=True)
class SkillSpec:
    name: str
    objective: str
    suggested_tools: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    completion: str = ""
    available_evidence_sources: tuple[str, ...] = ()
    expected_evidence: tuple[str, ...] = ()
    typical_handoff: str = ""

    @property
    def description(self) -> str:
        return self.objective

    @property
    def preconditions(self) -> tuple[str, ...]:
        return self.prerequisites

    @property
    def completion_criteria(self) -> str:
        return self.completion


def _project_incident_skills() -> dict[str, SkillSpec]:
    """Build the incident catalogue from the project Skill files."""
    result: dict[str, SkillSpec] = {}
    for item in LocalSkillSource().discover().values():
        metadata = item.metadata
        projection = metadata.get("registry_projection")
        if not isinstance(projection, dict):
            continue
        completion = projection.get("completion_rule") or {}
        result[item.skill_id] = SkillSpec(
            name=item.skill_id,
            objective=str(metadata.get("description") or item.description),
            suggested_tools=tuple(str(value) for value in metadata.get(
                "runtime_allowed_tools", metadata.get("allowed_tools", ())
            )),
            prerequisites=tuple(str(value) for value in metadata.get("signals", ())),
            completion=str(completion.get("rule") or item.completion),
            available_evidence_sources=tuple(str(value) for value in metadata.get("expected_evidence", ())),
            expected_evidence=tuple(str(value) for value in metadata.get("expected_evidence", ())),
            typical_handoff="Runtime owns Obligation completion; Skill text is guidance only.",
        )
    return result


# Canonical incident names are the 12 project Skill IDs. This mapping is the
# only vocabulary exposed to the active CloudOps Planner/Runtime path.
CANONICAL_INCIDENT_SKILLS = _project_incident_skills()
INCIDENT_SKILLS = dict(CANONICAL_INCIDENT_SKILLS)


def render_skill_catalog(compact: bool = False, *, skills=None) -> str:
    """Render only the canonical CloudOps DomainSkillRegistry vocabulary."""
    skills = CANONICAL_INCIDENT_SKILLS if skills is None else skills
    if compact:
        return "\n".join(f"- {s.name}: {s.objective}" for s in skills.values())
    return "\n".join(
        f"- {s.name}\n"
        f"  description: {s.description}\n"
        f"  preconditions: {'; '.join(s.preconditions) or 'none'}\n"
        f"  available evidence sources: {'; '.join(s.available_evidence_sources) or 'none'}\n"
        f"  suggested tools: {','.join(s.suggested_tools) or 'none'} (guidance, not a permission boundary)\n"
        f"  expected evidence: {'; '.join(s.expected_evidence) or 'none'}\n"
        f"  completion criteria: {s.completion_criteria}\n"
        f"  typical handoff: {s.typical_handoff or 'none'}"
        for s in skills.values()
    )
