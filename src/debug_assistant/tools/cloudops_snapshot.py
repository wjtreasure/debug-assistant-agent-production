from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator
from jsonschema import Draft202012Validator
from debug_assistant._vendor.cloudops_snapshot import (
    RESOURCE_ALIASES_DB, CLUSTER_SCOPED_RESOURCE_TYPES,
    LABEL_SELECTOR_SUPPORTED_TYPES, YAML_SNAPSHOT_RESOURCES,
)

from debug_assistant.models import ToolObservation
from debug_assistant.incidents.contracts import CausalChainLink, ClaimEvidenceMapping, EvidenceId, SourceClaim
from debug_assistant.tools.base import Tool, ToolArgs, ToolSpec
from debug_assistant.tools.cloudops_source import CloudOpsSourceBinding, CloudOpsSourceToolRegistry, SourceCodeArgs, source_provenance_profile
from debug_assistant.tools.official_cloudops import (
    OfficialCloudOps, official_result_success,
)


class ServiceNameArgs(ToolArgs):
    service_name: str = Field(min_length=1)


class ConnectivityArgs(ServiceNameArgs):
    namespace: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class EmptyArgs(ToolArgs):
    pass


def resource_argument_contract(*, projections=False):
    """Project the pinned backend's protocol, without inspecting snapshot data."""
    branches = []
    for cluster, selector, yaml in sorted({(value in CLUSTER_SCOPED_RESOURCE_TYPES,
            value in LABEL_SELECTOR_SUPPORTED_TYPES, value in YAML_SNAPSHOT_RESOURCES)
            for value in RESOURCE_ALIASES_DB.values()}):
        types = [key for key, value in RESOURCE_ALIASES_DB.items() if
            (value in CLUSTER_SCOPED_RESOURCE_TYPES, value in LABEL_SELECTOR_SUPPORTED_TYPES,
             value in YAML_SNAPSHOT_RESOURCES) == (cluster, selector, yaml)]
        scope = {'resource_type': {'enum': types}}
        if not cluster:
            scope['namespace'] = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
        if not projections:
            branches.append({'properties': scope})
            continue
        ordinary = {**scope, 'output_yaml': {'enum': [False]}}
        ordinary['label_selector'] = ({'anyOf': [{'type': 'null'},
            {'type': 'string', 'pattern': r'^[^=]*[^\s=][^=]*=[^=]*[^\s=][^=]*$'}]}
            if selector else {'type': 'null'})
        branches.append({'properties': ordinary})
        if yaml:
            branches.append({'properties': {**scope, 'output_yaml': {'enum': [True]},
                'show_labels': {'enum': [False]}, 'output_wide': {'enum': [False]},
                'label_selector': {'type': 'null'}}})
    return {'anyOf': branches}


class GetResourcesArgs(ToolArgs):
    model_config = {**ToolArgs.model_config, 'json_schema_extra': resource_argument_contract(projections=True)}
    # Command choices describe the same legal projection as the validator.
    # Runtime derives low-level flags from an explicit choice, never repairs
    # an illegal model-generated flag combination.
    command_choices: ClassVar[dict[str, dict[str, dict[str, bool]]]] = {
        'presentation': {
            'STANDARD': {'show_labels': False, 'output_wide': False, 'output_yaml': False},
            'LABELS': {'show_labels': True, 'output_wide': False, 'output_yaml': False},
            'WIDE': {'show_labels': False, 'output_wide': True, 'output_yaml': False},
            'YAML': {'show_labels': False, 'output_wide': False, 'output_yaml': True},
        }
    }
    resource_type: str = Field(min_length=1)
    namespace: str | None = None
    name: str | None = Field(
        default=None,
        description=(
            "Exact captured Kubernetes resource name. This is not a logical "
            "service/app name; leave empty to list resources and obtain an exact name first."
        ),
    )
    show_labels: bool = Field(
        default=False,
        description="Add captured label columns; mutually exclusive with output_wide.",
    )
    output_wide: bool = Field(
        default=False,
        description="Add captured IP/node/placement columns; mutually exclusive with show_labels.",
    )
    output_yaml: bool = False
    label_selector: str | None = Field(
        default=None,
        description=(
            "Optional simple key=value selector for captured pods, services, "
            "deployments, or persistentvolumeclaims."
        ),
    )

    @model_validator(mode="after")
    def validate_output_projection(self) -> "GetResourcesArgs":
        if self.show_labels and self.output_wide:
            raise ValueError("show_labels and output_wide are mutually exclusive")
        if self.label_selector is not None and not self.label_selector.strip():
            raise ValueError("label_selector must be non-empty when provided")
        arguments = self.model_dump(mode='json')
        arguments['resource_type'] = self.resource_type.lower()
        error = next(Draft202012Validator(type(self).model_json_schema()).iter_errors(arguments), None)
        if error:
            raise ValueError(error.message)
        return self


class GetAppYAMLArgs(ToolArgs):
    app_name: str = Field(min_length=1)


class DescribeResourceArgs(ToolArgs):
    model_config = {**ToolArgs.model_config, 'json_schema_extra': resource_argument_contract()}
    resource_type: str = Field(min_length=1)
    namespace: str | None = None
    name: str = Field(
        min_length=1,
        description="Exact captured Kubernetes resource name returned by a list/query result; logical service/app names are not expanded.",
    )

    @model_validator(mode="after")
    def validate_scope(self):
        arguments = self.model_dump(mode='json')
        arguments['resource_type'] = self.resource_type.lower()
        error = next(Draft202012Validator(type(self).model_json_schema()).iter_errors(arguments), None)
        if error:
            raise ValueError(error.message)
        return self


class ErrorLogsArgs(ToolArgs):
    namespace: str = Field(min_length=1)
    service_name: str = Field(min_length=1)


class ListCodeFilesArgs(ToolArgs):
    app_name: str = Field(min_length=1)


class NodeServiceStatusArgs(ToolArgs):
    node_name: str = Field(min_length=1)
    service_name: str = Field(min_length=1)


class FinalizeDiagnosisArgs(ToolArgs):
    component: str = Field(
        min_length=1,
        description=(
            "Provider proposal field. Runtime ignores this value and copies the "
            "component from the finalized Incident Hypothesis."
        ),
    )
    fault: str = Field(
        min_length=1,
        description=(
            "Provider proposal field. Runtime ignores this value and copies the "
            "fault from the finalized Incident Hypothesis."
        ),
    )
    fault_code: str = Field(
        default="",
        description=(
            "Structured lowercase snake_case fault taxonomy projection. Runtime "
            "uses the already validated Incident Hypothesis as the authority."
        ),
    )
    fault_explanation: str = Field(
        default="",
        description=(
            "Evidence-grounded explanation of the fault. Runtime uses the "
            "already validated Incident Hypothesis as the authority."
        ),
    )
    mechanism: str = Field(
        min_length=1,
        description=(
            "Provider proposal field. Runtime ignores this value and copies the "
            "mechanism from the finalized Incident Hypothesis."
        ),
    )
    evidence_ids: list[EvidenceId] = Field(
        min_length=2,
        description=(
            "Provider Evidence projection. Runtime validates the IDs but "
            "uses the finalized Incident Hypothesis supporting Evidence as the "
            "Candidate authority. Final Review cannot see uncited Evidence."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    claim_evidence_mapping: list[ClaimEvidenceMapping] = Field(
        default_factory=list,
        description="Optional structured mapping from each material claim to supporting ev-* IDs.",
    )
    causal_chain_summary: list[CausalChainLink] = Field(
        default_factory=list,
        description="Optional concise cause/effect chain; never include raw chain-of-thought.",
    )
    source_claims: list[SourceClaim] = Field(
        default_factory=list,
        description=(
            "Optional source claims. Each claim must declare the canonical file "
            "and exact line range covered by cited read_file CODE Evidence."
        ),
    )


class _OfficialTool(Tool):
    """Executable adapter; all query semantics belong to KubernetesTools."""

    def __init__(self, spec: ToolSpec, official: OfficialCloudOps, method: str):
        self.spec, self.official, self.method = spec, official, method

    @property
    def available(self) -> bool:
        return self.method != "GetRecentLogs" or self.official.raw_logs_available

    def provenance_profiles(self):
        if self.spec.name == "get_source_code":
            return (source_provenance_profile(self.spec.name),)
        arguments = ({'resource_type': value} for value in RESOURCE_ALIASES_DB.values()) if (
            'resource_type' in self.spec.args_model.model_fields) else ({},)
        pairs = {tuple(_incident_context_metadata(self.spec.name, value)[key] for key in
                       ('context_kind', 'information_source')) for value in arguments}
        return tuple({'context_kind': kind, 'information_source': source} for kind, source in sorted(pairs))

    def provenance_for_arguments(self, arguments):
        if self.spec.name == 'get_source_code':
            return (source_provenance_profile(self.spec.name),)
        metadata = _incident_context_metadata(self.spec.name, arguments)
        return ({key: metadata[key] for key in ('context_kind', 'information_source')},)

    def provenance_argument_domains(self, constraints):
        """Finite backend resource aliases sharing the exact declared output type."""
        if 'resource_type' not in self.spec.args_model.model_fields:
            return {}
        from debug_assistant.tools.provenance import profile_matches_constraints
        return {'resource_type': tuple(alias for alias in sorted(RESOURCE_ALIASES_DB)
            if any(profile_matches_constraints({
                'context_kinds': (profile['context_kind'],),
                'information_sources': (profile['information_source'],),
                'source_tools': (self.spec.name,),
            }, constraints) for profile in self.provenance_for_arguments({'resource_type': alias})))}

    def provenance_categories(self):
        from debug_assistant.tools.provenance import profile_categories
        return profile_categories(self.provenance_profiles())

    def execute(self, **kwargs: Any) -> ToolObservation:
        arguments = dict(kwargs)
        method = self.method
        if method == "GetResources":
            yaml_output = bool(arguments.pop("output_yaml", False))
            if yaml_output:
                if arguments.get("label_selector"):
                    return ToolObservation(self.spec.name, False,
                                           "YAML and selector use distinct official contracts.",
                                           {}, "schema_validation")
                arguments.pop("label_selector", None)
                arguments["output_yaml"] = True
                method = "GetResources_v2"
        metadata = {"arguments": kwargs, "snapshot": True,
                    **_incident_context_metadata(self.spec.name, kwargs)}
        try:
            if method == "GetSourceCode":
                return ToolObservation(self.spec.name, False, "Source workspace unavailable.",
                                       {**metadata, "status": "UNAVAILABLE", "semantic_negative": False,
                                        "reason": "source_workspace_unavailable", "official_invoked": False},
                                       "source_workspace_unavailable")
            output, acquired, provenance = self.official.invoke(method, arguments)
            success = acquired and official_result_success(output)
            metadata.update(provenance)
            metadata.update({
                "status": "AVAILABLE" if success else "UNAVAILABLE",
                "semantic_negative": success and provenance.get("semantic_negative", False),
                "negative_observation": success and isinstance(output, str) and output.casefold() == "connection failed",
            })
            if not acquired:
                metadata["reason"] = provenance.get("reason", "observation_not_present_in_snapshot")
            return ToolObservation(
                self.spec.name, success, _serialize_snapshot_output(output), metadata,
                None if success else "snapshot_unavailable" if not acquired else "official_tool_error",
            )
        except (ValueError, OSError) as exc:
            return ToolObservation(self.spec.name, False, str(exc), metadata,
                                   type(exc).__name__)


class _TopologyTool(Tool):
    spec = ToolSpec(
        "get_service_topology",
        "Read structured Online Boutique responsibility, upstream, downstream, dependencies and typical signals.",
        ServiceNameArgs,
        capability="incident_read", parallel_safe=True,
    )

    def __init__(self, topology_path: Path):
        self._topology = json.loads(topology_path.read_text(encoding="utf-8"))["services"]

    _SOURCE_PROFILE = (('context_kind', 'SERVICE'), ('information_source', 'static_kg'))

    def provenance_profiles(self):
        return (dict(self._SOURCE_PROFILE),)

    def provenance_categories(self):
        from debug_assistant.tools.provenance import profile_categories
        return profile_categories(self.provenance_profiles())

    def execute(self, **kwargs: Any) -> ToolObservation:
        name = kwargs["service_name"]
        value = self._topology.get(name)
        if value is None:
            return ToolObservation(tool=self.spec.name, ok=False, content=f"Unknown service: {name}",
                                   metadata={"arguments": kwargs}, error_type="not_found")
        return ToolObservation(tool=self.spec.name, ok=True, content=json.dumps(value, ensure_ascii=False, sort_keys=True),
                               metadata={
                                   "arguments": kwargs, "structured_topology": True,
                                   **dict(self._SOURCE_PROFILE), "context_target": name,
                               })


class CloudOpsSnapshotToolRegistry:
    """Read-only replay registry for a runtime-visible Cloud-OpsBench snapshot."""

    # These CloudOps snapshot primitives are always available. ``code_search``
    # is exposed only when a bound application source workspace exists.
    PLANNER_HIDDEN_TOOLS = frozenset({
        "list_code_files", "inspect_symbol_context",
    })
    PLANNER_CODE_TOOLS = ("repo_tree", "grep", "read_file", "symbol_search", "code_search")

    def __init__(self, runtime_data_dir: str | Path, topology_path: str | Path, *, search_engine=None):
        self.official = OfficialCloudOps(runtime_data_dir)
        specs = {
            "check_service_connectivity": ToolSpec(
                "check_service_connectivity", "Probe a service and port from the captured incident environment.",
                ConnectivityArgs, capability="incident_read",
            ),
            "get_resources": ToolSpec(
                "get_resources",
                "Read captured Kubernetes state (status, labels, placement, and configuration). "
                "It does not provide live CPU/memory utilization metrics unless those fields "
                "are explicitly present in the captured output. Resource name is exact; omit "
                "it to list and discover a captured resource name.",
                GetResourcesArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_app_yaml": ToolSpec(
                "get_app_yaml", "Read captured workload and Service manifests for one application.",
                GetAppYAMLArgs, capability="incident_read", parallel_safe=True,
            ),
            "describe_resource": ToolSpec(
                "describe_resource", "Read a captured Kubernetes resource by its exact captured name, including state, configuration and events.",
                DescribeResourceArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_alerts": ToolSpec(
                "get_alerts", "Read captured anomaly and alert summaries for the incident environment.",
                EmptyArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_error_logs": ToolSpec(
                "get_error_logs", "Read captured error-log summaries for one service in a namespace.",
                ErrorLogsArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_recent_logs": ToolSpec(
                "get_recent_logs", "Read official captured raw recent logs; unavailable when raw logs are absent.",
                ErrorLogsArgs, capability="incident_read",
            ),
            "get_source_code": ToolSpec(
                "get_source_code", "Acquire app-scoped source through official GetSourceCode when source is bound.",
                SourceCodeArgs, capability="source_read",
            ),
            "list_code_files": ToolSpec(
                "list_code_files", "Read the captured application source-file inventory and file descriptions.",
                ListCodeFilesArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_service_dependencies": ToolSpec(
                "get_service_dependencies", "Read captured upstream and downstream service dependencies.",
                ServiceNameArgs, capability="incident_read", parallel_safe=True,
            ),
            "check_node_service_status": ToolSpec(
                "check_node_service_status", "Read captured node-level system service status.",
                NodeServiceStatusArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_cluster_configuration": ToolSpec(
                "get_cluster_configuration", "Read captured cluster node readiness and allocatable resources.",
                EmptyArgs, capability="incident_read", parallel_safe=True,
            ),
        }
        methods = {local: official for official, local in _OFFICIAL_TOOL_NAMES.items()}
        self.official_aliases = dict(_OFFICIAL_TOOL_NAMES)
        self._tools = {name: _OfficialTool(spec, self.official, methods[name])
                       for name, spec in specs.items()}
        topology = _TopologyTool(Path(topology_path))
        self._tools[topology.spec.name] = topology
        self.source_binding = CloudOpsSourceBinding(runtime_data_dir)
        if self.source_binding.available:
            self._tools.update(CloudOpsSourceToolRegistry(
                self.source_binding, search_engine=search_engine, official=self.official,
            ).tools())
        self._finalize_spec = ToolSpec(
            "finalize_diagnosis", "Submit the evidence-backed final root-cause candidate for review. Final Review sees only the cited evidence_ids, so include every Evidence item required to support the component and mechanism.",
            FinalizeDiagnosisArgs, capability="diagnosis_control",
        )

    def get(self, name: str):
        return self._tools.get(name)

    @property
    def decision_tool_aliases(self):
        # The existing completion/source-coverage path cites read_file. The
        # public official API adapter remains available for direct callers.
        return {**self.official_aliases, 'GetSourceCode': 'read_file'}

    def register(self, tool: Tool) -> None:
        """Register an optional capability-layer tool on this existing registry."""
        name = str(tool.spec.name)
        if name == self._finalize_spec.name or name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def specs(self):
        return [tool.spec for tool in self._tools.values()] + [self._finalize_spec]

    def function_schemas(self, visible_tools=None):
        visible = None if visible_tools is None else {str(name) for name in visible_tools}
        schemas = []
        for spec in self.specs():
            if spec.name in self.PLANNER_HIDDEN_TOOLS:
                continue
            if visible is not None and spec.name not in visible:
                continue
            schema = _compact_provider_schema(
                spec.function_schema(),
                keep_descriptions=spec.name == "finalize_diagnosis",
            )
            schemas.append(schema)
        return schemas

    def planner_visible_code_tools(self) -> tuple[str, ...]:
        """Return the code-investigation tools actually exposed to the Planner."""
        return tuple(
            name for name in self.PLANNER_CODE_TOOLS
            if self.get(name) is not None and name not in self.PLANNER_HIDDEN_TOOLS
        )

    def validate_arguments(self, name: str, arguments: dict[str, Any]):
        spec = next((item for item in self.specs() if item.name == name), None)
        if spec is None:
            return None, {"error_type": "unknown_tool", "message": f"unknown tool: {name}"}
        try:
            normalized = spec.args_model.model_validate(arguments).model_dump()
            if name == "get_resources":
                # Official cache keys omit false/None projection options but
                # retain the historical explicit ``name`` field.
                normalized = {
                    key: value for key, value in normalized.items()
                    if key not in {"show_labels", "output_wide", "label_selector"}
                    or value not in (False, None)
                }
            return normalized, None
        except Exception as exc:
            return None, {"error_type": "schema_validation", "message": str(exc)}


_OFFICIAL_TOOL_NAMES = {
    "CheckServiceConnectivity": "check_service_connectivity",
    "GetResources": "get_resources",
    "GetAppYAML": "get_app_yaml",
    "DescribeResource": "describe_resource",
    "GetAlerts": "get_alerts",
    "GetErrorLogs": "get_error_logs",
    "GetRecentLogs": "get_recent_logs",
    "GetSourceCode": "get_source_code",
    "ListCodeFiles": "list_code_files",
    "GetServiceDependencies": "get_service_dependencies",
    "CheckNodeServiceStatus": "check_node_service_status",
    "GetClusterConfiguration": "get_cluster_configuration",
}


def _incident_context_metadata(tool_name: str, arguments: dict[str, Any]) -> dict[str, str]:
    resource_type = str(arguments.get("resource_type") or "").lower()
    resource_type = RESOURCE_ALIASES_DB.get(resource_type, resource_type)
    resource_kinds = {
        "pods": "POD", "services": "SERVICE", "endpoints": "ENDPOINT",
        "deployments": "DEPLOYMENT", "events": "EVENT", "nodes": "NODE",
        "configmaps": "CONFIGMAP", "daemonsets": "DAEMONSET", "ingresses": "INGRESS",
        "jobs": "JOB", "namespaces": "NAMESPACE", "networkpolicies": "NETWORKPOLICY",
        "persistentvolumeclaims": "PVC", "persistentvolumes": "PV", "replicasets": "REPLICASET",
        "resourcequota": "RESOURCEQUOTA", "rolebindings": "ROLEBINDING",
        "secrets": "SECRET", "serviceaccounts": "SERVICEACCOUNT", "statefulsets": "STATEFULSET",
        "storageclasses": "STORAGECLASS", "horizontalpodautoscalers": "HPA",
    }
    if tool_name == "get_app_yaml":
        kind = "CONFIG"
    elif tool_name == "check_service_connectivity":
        kind = "ENDPOINT"
    elif tool_name == "get_alerts":
        kind = "ALERT"
    elif tool_name in {"get_error_logs", "get_recent_logs"}:
        kind = "LOG"
    elif tool_name == "list_code_files":
        kind = "CODE_INDEX"
    elif tool_name == "get_service_dependencies":
        kind = "SERVICE"
    elif tool_name == "check_node_service_status":
        kind = "NODE"
    elif tool_name == "get_cluster_configuration":
        kind = "CLUSTER"
    else:
        kind = resource_kinds.get(resource_type, "OBSERVATION")
    target = str(arguments.get("name") or arguments.get("app_name")
                  or arguments.get("service_name") or arguments.get("node_name")
                  or ("alerts" if tool_name == "get_alerts" else resource_type))
    return {"context_kind": kind, "context_target": target, "information_source": "incident_snapshot"}


def _serialize_snapshot_output(output: Any) -> str:
    """Keep structured official outputs JSON-shaped in model-visible evidence."""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def _compact_provider_schema(schema: dict[str, Any], *, keep_descriptions: bool = False) -> dict[str, Any]:
    """Remove non-semantic JSON-Schema decoration before provider transport.

    Pydantic remains the validation source of truth.  Provider-facing tool
    schemas only need names, types, enums, constraints and required fields;
    titles/defaults/examples and long field descriptions are repeated on every
    Planner call. The finalization schema keeps descriptions because its
    Evidence projection is a high-risk semantic boundary.
    """
    def compact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in {"title", "default", "examples", "deprecated", "$comment"}:
                    continue
                if key == "description" and not keep_descriptions:
                    continue
                result[key] = compact(item)
            return result
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    compacted = compact(copy.deepcopy(schema))
    if not keep_descriptions:
        # A few high-risk argument descriptions are part of the executable
        # contract from a Planner's perspective. Preserve those short cues
        # while still removing the large repeated prose from ordinary fields.
        original_properties = (
            schema.get("function", {}).get("parameters", {}).get("properties", {})
        )
        compacted_properties = (
            compacted.get("function", {}).get("parameters", {}).get("properties", {})
        )
        for field_name in ("name",):
            description = original_properties.get(field_name, {}).get("description")
            if description and field_name in compacted_properties:
                compacted_properties[field_name]["description"] = description
    return compacted
