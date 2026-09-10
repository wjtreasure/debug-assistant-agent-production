from __future__ import annotations
from dataclasses import dataclass

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

INCIDENT_SKILLS = {
 "service_investigation": SkillSpec(
     name="service_investigation",
     objective="Investigate service discovery, dependency, reachability, selector, port and endpoint consistency without assuming a fault label.",
     suggested_tools=("get_service_topology", "get_service_dependencies", "check_service_connectivity", "get_resources", "get_app_yaml"),
     prerequisites=("The incident indicates partial or complete request-path unreachability, or a service routing evidence gap.",),
     available_evidence_sources=("Structured service topology", "Connectivity probes", "Kubernetes Service/Endpoint/workload snapshots"),
     expected_evidence=(
         "The affected request path and service candidate are grounded.",
         "Declared service routing configuration is compared with discovered endpoints and workload configuration.",
         "A routing hypothesis is supported or falsified by at least two independent observations.",
     ),
     completion="A component and causal service-routing mechanism are supported by independent observations.",
     typical_handoff="Hand off to Final Review when complete; otherwise hand off to the skill matching the remaining evidence gap.",
 ),
 "runtime_resource_investigation": SkillSpec(
     name="runtime_resource_investigation",
     objective="Investigate workload lifecycle, container state, restart reasons, resource requests/limits and runtime events without assuming a fault label.",
     suggested_tools=("get_alerts", "get_error_logs", "get_resources", "describe_resource", "get_app_yaml", "get_service_topology", "get_cluster_configuration", "check_node_service_status"),
     prerequisites=("The incident or current evidence indicates workload availability, startup, restart, scheduling, or resource-pressure symptoms.",),
     available_evidence_sources=("Kubernetes workload listings", "Resource descriptions and events", "Captured workload configuration", "Structured service topology"),
     expected_evidence=(
         "The abnormal workload and lifecycle state are grounded relative to peers.",
         "Runtime events or termination/waiting reasons identify the failing mechanism.",
         "Resource configuration is compared with the observed runtime behavior using independent observations.",
     ),
     completion="A workload component and causal runtime/resource mechanism are supported by independent observations.",
     typical_handoff="Hand off to Final Review when complete; otherwise hand off to the skill matching the remaining telemetry, service, or code evidence gap.",
 ),
 "telemetry_investigation": SkillSpec(
     name="telemetry_investigation",
     objective="Investigate captured alerts and error-log summaries, correlate their time, target and service context, and separate symptoms from causal evidence without assuming a fault label.",
     suggested_tools=("get_alerts", "get_error_logs", "get_service_topology", "get_resources", "describe_resource"),
     prerequisites=("The incident or current evidence contains a performance, latency, anomaly, or application-error information gap.",),
     available_evidence_sources=("Captured alert summaries", "Captured service error-log summaries", "Structured service topology", "Kubernetes workload snapshots"),
     expected_evidence=(
         "An alert or log signal is tied to a concrete service and time window.",
         "The signal is compared with workload, dependency, or configuration observations.",
         "Normal/empty telemetry results remain negative observations rather than fabricated causal facts.",
     ),
     completion="A service-level symptom and its evidence-supported causal direction are grounded, or the telemetry gap is explicitly recorded.",
     typical_handoff="Hand off to runtime, service, or code investigation when telemetry identifies the affected component but not the mechanism.",
 ),
 "code_investigation": SkillSpec(
     name="code_investigation",
     objective="Investigate application-code candidates from the captured file index and, when a source workspace is explicitly bound, verify behavior with bounded repository tree, search, symbol and file-read tools.",
     suggested_tools=("list_code_files", "repo_tree", "grep", "read_file", "symbol_search", "code_search", "inspect_symbol_context"),
     prerequisites=("The incident or current evidence indicates an application-code information gap and a code file inventory is available.",),
     available_evidence_sources=("Captured application source-file inventory", "Bound repository source workspace when provided"),
     expected_evidence=(
         "A concrete service and candidate source file are grounded by the runtime-visible code index.",
         "A source-backed causal claim is made only after a bound workspace provides the relevant source text.",
         "An index-only result is reported as an availability limitation, not as proof of implementation behavior.",
     ),
     completion="The relevant code location and behavior are source-backed when source is available; otherwise the missing source binding is explicit.",
     typical_handoff="Hand off to Final Review only when the candidate's causal claims are supported by cited runtime or source Evidence.",
 ),
}

SKILLS = {
 "issue_triage": SkillSpec("issue_triage", "Extract failure symptom, expected/actual behavior, constraints and search anchors from the issue.", ("repo_tree","grep","code_search"), (), "Issue has actionable search anchors."),
 "repository_exploration": SkillSpec("repository_exploration", "Map issue concepts to repository modules and candidate symbols.", ("repo_tree","grep","code_search","symbol_search","read_file","git_log"), (), "At least one plausible code location is grounded by repository evidence."),
 "hypothesis_generation": SkillSpec("hypothesis_generation", "Form falsifiable root-cause hypotheses tied to evidence.", ("read_file","symbol_search","grep","code_search","git_log","discover_tests"), (), "One or more hypotheses identify a mechanism, not merely a file name."),
 "hypothesis_validation": SkillSpec("hypothesis_validation", "Try to falsify leading hypotheses using implementation, call sites, tests and history.", ("read_file","grep","code_search","symbol_search","git_log","git_show","discover_tests"), ("evidence",), "Leading hypothesis has support and explicit uncertainty/contradiction handling."),
 "impact_analysis": SkillSpec("impact_analysis", "Estimate affected callers, modules, tests and behavior without modifying code.", ("grep","symbol_search","read_file","discover_tests"), ("evidence",), "Likely impact scope and change points are identified."),
 "report_synthesis": SkillSpec("report_synthesis", "Produce an evidence-backed development decision report.", (), ("evidence",), "Root cause, locations, evidence, uncertainty and next checks are explicit."),
}

def render_skill_catalog(compact: bool=False, *, skills=None) -> str:
    skills = SKILLS if skills is None else skills
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
