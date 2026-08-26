from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SkillSpec:
    name: str
    objective: str
    suggested_tools: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    completion: str = ""

SKILLS = {
 "issue_triage": SkillSpec("issue_triage", "Extract failure symptom, expected/actual behavior, constraints and search anchors from the issue.", ("repo_tree","grep","code_search"), (), "Issue has actionable search anchors."),
 "repository_exploration": SkillSpec("repository_exploration", "Map issue concepts to repository modules and candidate symbols.", ("repo_tree","grep","code_search","symbol_search","read_file","git_log"), (), "At least one plausible code location is grounded by repository evidence."),
 "hypothesis_generation": SkillSpec("hypothesis_generation", "Form falsifiable root-cause hypotheses tied to evidence.", ("read_file","symbol_search","grep","code_search","git_log","discover_tests"), (), "One or more hypotheses identify a mechanism, not merely a file name."),
 "hypothesis_validation": SkillSpec("hypothesis_validation", "Try to falsify leading hypotheses using implementation, call sites, tests and history.", ("read_file","grep","code_search","symbol_search","git_log","git_show","discover_tests"), ("evidence",), "Leading hypothesis has support and explicit uncertainty/contradiction handling."),
 "impact_analysis": SkillSpec("impact_analysis", "Estimate affected callers, modules, tests and behavior without modifying code.", ("grep","symbol_search","read_file","discover_tests"), ("evidence",), "Likely impact scope and change points are identified."),
 "report_synthesis": SkillSpec("report_synthesis", "Produce an evidence-backed development decision report.", (), ("evidence",), "Root cause, locations, evidence, uncertainty and next checks are explicit."),
}

def render_skill_catalog() -> str:
    return "\n".join(
        f"- {s.name}: {s.objective} Suggested tools={','.join(s.suggested_tools) or 'none'} (guidance only, not a hard permission). Completion={s.completion}"
        for s in SKILLS.values()
    )
