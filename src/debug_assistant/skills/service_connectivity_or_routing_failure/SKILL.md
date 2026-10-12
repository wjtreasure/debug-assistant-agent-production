---
skill_id: service_connectivity_or_routing_failure
description: A caller cannot resolve or connect to a Kubernetes Service, or traffic
  fails to reach otherwise healthy backends.
source_sha256: d0d69988a9afcf5f2123f58d1f5eca74b1187060aae7e4ec8052e57374ba0a0f
derived_sha256: 5cb288b1b7e84e90d954e9fe3d5cba7469c1f1e134a36a3a3530ba04c5789ae0
signals:
- downstream Service is unreachable while pods are Running and Ready
- DNS lookup for a downstream service fails
- connection refused or connection timeout identifies a downstream target
- Service has empty or incorrect endpoints
- traffic to an existing Service fails despite healthy backend pods
entry:
- locate_service_connectivity_failure
allowed_tools: &id001
- get_service_dependencies
- get_error_logs
- get_app_yaml
- describe_resource
- get_resources
expected_evidence: &id002
- pod_health_status
- service_inventory
- entrypoint_service
- service_dependency_graph
- calling_service
- failed_dependency
- failing_service
- failing_service_port
- connectivity_failure_pattern
- dns_resolution_failure_pattern
- configuration_owner_service
- downstream_address_env_var
- env_var_value
- actual_service_name
- address_mismatch_confirmed
- service_selector
- service_endpoints
- service_port
- service_targetPort
- service_protocol
- candidate_pod_labels
- container_port
- selector_mismatch_candidate
- target_port_mismatch_candidate
- protocol_mismatch_candidate
- routing_configuration_consistent
- container_port_name
- port_mismatch_verdict
- container_protocol
- client_expected_protocol
- connectivity_failure
- protocol_mismatch_confirmed
- selector_value
- pod_label_value
- selector_label_mismatch
- empty_or_wrong_endpoints
- root_cause_confirmed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_service_dependencies
- get_error_logs
- get_app_yaml
- describe_resource
- get_resources
registry_projection:
registry_projection:
  objective: A caller cannot resolve or connect to a Kubernetes Service, or traffic
    fails to reach otherwise healthy backends.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_service_connectivity_failure:
    goal: Localize the failed downstream Service and calling path when pods are healthy
      but DNS, connection, endpoint, or service traffic still fails.
    actions:
    - tool: GetServiceDependencies
      args:
        service_name: <entrypoint_service>
    - tool: GetErrorLogs
      args:
        namespace: <namespace>
        service_name: <entrypoint_service>
    evidence:
    - pod_health_status
    - service_inventory
    - entrypoint_service
    - service_dependency_graph
    - calling_service
    - failed_dependency
    - failing_service
    - failing_service_port
    - connectivity_failure_pattern
    - dns_resolution_failure_pattern
    next:
    - confirm_service_address_configuration
    - inspect_service_routing_context
  confirm_service_address_configuration:
    goal: Confirm that the service owning a downstream address configuration uses
      an invalid hostname in its environment variables.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_service_name>
    evidence:
    - configuration_owner_service
    - downstream_address_env_var
    - env_var_value
    - actual_service_name
    - address_mismatch_confirmed
    next: []
  inspect_service_routing_context:
    goal: Inspect the localized Service's selector, endpoints, ports, protocol, and
      backing Pod labels/container ports once before selecting a routing confirmation
      branch.
    actions:
    - tool: DescribeResource
      args:
        resource_type: service
        name: <suspected_downstream_service>
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        show_labels: true
    - tool: GetAppYAML
      args:
        app_name: <suspected_downstream_service>
    evidence:
    - service_selector
    - service_endpoints
    - service_port
    - service_targetPort
    - service_protocol
    - candidate_pod_labels
    - container_port
    - selector_mismatch_candidate
    - target_port_mismatch_candidate
    - protocol_mismatch_candidate
    - routing_configuration_consistent
    next:
    - confirm_service_target_port_mapping
    - confirm_service_selector_mapping
    - confirm_service_protocol_compatibility
  confirm_service_target_port_mapping:
    goal: Compare the Service targetPort with the container's listening port to confirm
      a port mapping mismatch.
    actions:
    - tool: GetAppYAML
      args:
        app_name: suspected_downstream_service
    evidence:
    - container_port
    - container_port_name
    - service_targetPort
    - service_port
    - port_mismatch_verdict
    next: []
  confirm_service_protocol_compatibility:
    goal: Confirm that a service protocol mismatch (UDP vs TCP/gRPC) is the root cause.
    actions:
    - tool: GetAppYAML
      args:
        app_name: suspected_downstream_service
    evidence:
    - service_protocol
    - container_protocol
    - client_expected_protocol
    - connectivity_failure
    - protocol_mismatch_confirmed
    next: []
  confirm_service_selector_mapping:
    goal: Confirm service selector mismatch by proving the Service selector does not
      match backend pod labels.
    actions:
    - tool: DescribeResource
      args:
        resource_type: service
        name: <suspected_downstream_service>
        namespace: <application_namespace>
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <application_namespace>
        show_labels: true
    - tool: GetResources
      args:
        resource_type: endpoints
        namespace: <application_namespace>
    evidence:
    - selector_value
    - pod_label_value
    - selector_label_mismatch
    - empty_or_wrong_endpoints
    - root_cause_confirmed
    next: []
---

# service_connectivity_or_routing_failure

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_service_dependencies`, `get_error_logs`, `get_app_yaml`, `describe_resource`, `get_resources`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_service_connectivity_failure

Goal: Localize the failed downstream Service and calling path when pods are healthy but DNS, connection, endpoint, or service traffic still fails.

Actions:
- `GetServiceDependencies` `{"service_name": "<entrypoint_service>"}`
- `GetErrorLogs` `{"namespace": "<namespace>", "service_name": "<entrypoint_service>"}`

Expected evidence:
- `pod_health_status`
- `service_inventory`
- `entrypoint_service`
- `service_dependency_graph`
- `calling_service`
- `failed_dependency`
- `failing_service`
- `failing_service_port`
- `connectivity_failure_pattern`
- `dns_resolution_failure_pattern`

Next nodes: `confirm_service_address_configuration`, `inspect_service_routing_context`

### confirm_service_address_configuration

Goal: Confirm that the service owning a downstream address configuration uses an invalid hostname in its environment variables.

Actions:
- `GetAppYAML` `{"app_name": "<failing_service_name>"}`

Expected evidence:
- `configuration_owner_service`
- `downstream_address_env_var`
- `env_var_value`
- `actual_service_name`
- `address_mismatch_confirmed`

Next nodes: none

### inspect_service_routing_context

Goal: Inspect the localized Service's selector, endpoints, ports, protocol, and backing Pod labels/container ports once before selecting a routing confirmation branch.

Actions:
- `DescribeResource` `{"name": "<suspected_downstream_service>", "namespace": "<namespace>", "resource_type": "service"}`
- `GetResources` `{"namespace": "<namespace>", "resource_type": "pods", "show_labels": true}`
- `GetAppYAML` `{"app_name": "<suspected_downstream_service>"}`

Expected evidence:
- `service_selector`
- `service_endpoints`
- `service_port`
- `service_targetPort`
- `service_protocol`
- `candidate_pod_labels`
- `container_port`
- `selector_mismatch_candidate`
- `target_port_mismatch_candidate`
- `protocol_mismatch_candidate`
- `routing_configuration_consistent`

Next nodes: `confirm_service_target_port_mapping`, `confirm_service_selector_mapping`, `confirm_service_protocol_compatibility`

### confirm_service_target_port_mapping

Goal: Compare the Service targetPort with the container's listening port to confirm a port mapping mismatch.

Actions:
- `GetAppYAML` `{"app_name": "suspected_downstream_service"}`

Expected evidence:
- `container_port`
- `container_port_name`
- `service_targetPort`
- `service_port`
- `port_mismatch_verdict`

Next nodes: none

### confirm_service_protocol_compatibility

Goal: Confirm that a service protocol mismatch (UDP vs TCP/gRPC) is the root cause.

Actions:
- `GetAppYAML` `{"app_name": "suspected_downstream_service"}`

Expected evidence:
- `service_protocol`
- `container_protocol`
- `client_expected_protocol`
- `connectivity_failure`
- `protocol_mismatch_confirmed`

Next nodes: none

### confirm_service_selector_mapping

Goal: Confirm service selector mismatch by proving the Service selector does not match backend pod labels.

Actions:
- `DescribeResource` `{"name": "<suspected_downstream_service>", "namespace": "<application_namespace>", "resource_type": "service"}`
- `GetResources` `{"namespace": "<application_namespace>", "resource_type": "pods", "show_labels": true}`
- `GetResources` `{"namespace": "<application_namespace>", "resource_type": "endpoints"}`

Expected evidence:
- `selector_value`
- `pod_label_value`
- `selector_label_mismatch`
- `empty_or_wrong_endpoints`
- `root_cause_confirmed`

Next nodes: none
