---
skill_id: node_network_degradation
description: Network performance or reachability degrades in a node-correlated pattern,
  such as increased latency or differences between nodes.
source_sha256: c3b9335297a93a144062e27b9827ed231e5f81cd2690eacfb4e10c771993c6f9
derived_sha256: b40b99646f15c8d75c83af62be486606dbaa6fd4add05f93bc1ad933e5b780fc
signals:
- performance alert indicates node network degradation
- multiple services degrade on the same node
- traffic quality differs by node placement
- node-level latency or packet-loss symptom is reported
- the same service is reachable from some nodes but not from one node
- cross-node connectivity differs by source or destination node
entry:
- locate_node_network_scope
allowed_tools: &id001
- get_alerts
- get_resources
- get_cluster_configuration
- check_service_connectivity
- check_node_service_status
- describe_resource
- get_error_logs
expected_evidence: &id002
- performance_alert_summary
- anomalous_node
- affected_services
- pod_node_mapping
- node_health_status
- node_network_evidence
- node_taints
- affected_services_on_node
- node_connectivity_results
- suspect_node
- cross_node_difference
- ordinary_node_failure
- node_network_degradation_supported
- kube_proxy_status
- containerd_status
- node_network_component_status
- kube_proxy_unavailable_supported
- components_healthy_for_network_attribution
- failing_node
- kube_proxy_unavailable_confirmed
- node_correlated_latency_degradation
- no_high_magnitude_tcp_retransmission_on_suspect_node
- node_network_components_healthy
- node_network_delay_supported
- high_magnitude_tcp_retransmission
- retransmission_node_matches_suspect_node
- affected_services_on_suspect_node
- packet_loss_supporting_timeouts
- node_network_packet_loss_supported
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_alerts
- get_resources
- get_cluster_configuration
- check_service_connectivity
- check_node_service_status
- describe_resource
- get_error_logs
registry_projection:
registry_projection:
  objective: Network performance or reachability degrades in a node-correlated pattern,
    such as increased latency or differences between nodes.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_node_network_scope:
    goal: Identify the node and affected services associated with observable node-network
      degradation or cross-node traffic-quality differences.
    actions:
    - tool: GetAlerts
      args:
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    - tool: GetClusterConfiguration
      args: {}
    evidence:
    - performance_alert_summary
    - anomalous_node
    - affected_services
    - pod_node_mapping
    - node_health_status
    - node_network_evidence
    next:
    - localize_degraded_node
  localize_degraded_node:
    goal: Localize a node-correlated network degradation by comparing node health,
      affected pod placement, and cross-node service connectivity.
    actions:
    - tool: GetClusterConfiguration
      args: {}
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    - tool: CheckServiceConnectivity
      args:
        service_name: <affected_service>
        port: <service_port>
        namespace: <namespace>
    evidence:
    - node_health_status
    - node_taints
    - pod_node_mapping
    - affected_services_on_node
    - node_connectivity_results
    - suspect_node
    - cross_node_difference
    - ordinary_node_failure
    - node_network_degradation_supported
    next:
    - inspect_node_network_components
    - confirm_node_network_delay_from_node_status
    - inspect_service_logs_for_network_packet_loss
  inspect_node_network_components:
    goal: Inspect kube-proxy and containerd health once on the localized suspect node
      to distinguish component unavailability from component-healthy network delay
      or packet loss.
    actions:
    - tool: CheckNodeServiceStatus
      args:
        node_name: <node_name>
        service_name: kube-proxy
    - tool: CheckNodeServiceStatus
      args:
        node_name: <node_name>
        service_name: containerd
    evidence:
    - kube_proxy_status
    - containerd_status
    - node_network_component_status
    - kube_proxy_unavailable_supported
    - components_healthy_for_network_attribution
    next:
    - confirm_kube_proxy_unavailable
    - confirm_node_network_delay_from_node_status
    - inspect_service_logs_for_network_packet_loss
  confirm_kube_proxy_unavailable:
    goal: Confirm kube-proxy unavailability from the component status already collected
      by the shared node-network component inspection.
    actions: []
    evidence:
    - kube_proxy_status
    - failing_node
    - kube_proxy_unavailable_confirmed
    next: []
  confirm_node_network_delay_from_node_status:
    goal: Confirm node_network_delay only when latency or response-delay degradation
      is correlated with the same suspect node and that node does not show a meaningful
      TCP Retransmission increase. Small fluctuations alone are insufficient to establish
      packet loss.
    actions:
    - tool: GetAlerts
      args: {}
    - tool: DescribeResource
      args:
        resource_type: node
        name: <node_name>
    evidence:
    - suspect_node
    - node_correlated_latency_degradation
    - no_high_magnitude_tcp_retransmission_on_suspect_node
    - node_network_components_healthy
    - node_network_delay_supported
    next: []
  inspect_service_logs_for_network_packet_loss:
    goal: Confirm node_network_packet_loss from a meaningful TCP Retransmission increase
      on the same suspect node that hosts the affected services. Timeout, Unavailable,
      DeadlineExceeded, traffic-drop, or latency evidence is supportive but is not
      sufficient without the node-aligned retransmission signal.
    actions:
    - tool: GetAlerts
      args: {}
    - tool: GetErrorLogs
      args:
        namespace: <application_namespace>
        service_name: <service_name>
    evidence:
    - suspect_node
    - high_magnitude_tcp_retransmission
    - retransmission_node_matches_suspect_node
    - affected_services_on_suspect_node
    - packet_loss_supporting_timeouts
    - node_network_packet_loss_supported
    next: []
---

# node_network_degradation

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_alerts`, `get_resources`, `get_cluster_configuration`, `check_service_connectivity`, `check_node_service_status`, `describe_resource`, `get_error_logs`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_node_network_scope

Goal: Identify the node and affected services associated with observable node-network degradation or cross-node traffic-quality differences.

Actions:
- `GetAlerts` `{"namespace": "<namespace>"}`
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`
- `GetClusterConfiguration` `{}`

Expected evidence:
- `performance_alert_summary`
- `anomalous_node`
- `affected_services`
- `pod_node_mapping`
- `node_health_status`
- `node_network_evidence`

Next nodes: `localize_degraded_node`

### localize_degraded_node

Goal: Localize a node-correlated network degradation by comparing node health, affected pod placement, and cross-node service connectivity.

Actions:
- `GetClusterConfiguration` `{}`
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`
- `CheckServiceConnectivity` `{"namespace": "<namespace>", "port": "<service_port>", "service_name": "<affected_service>"}`

Expected evidence:
- `node_health_status`
- `node_taints`
- `pod_node_mapping`
- `affected_services_on_node`
- `node_connectivity_results`
- `suspect_node`
- `cross_node_difference`
- `ordinary_node_failure`
- `node_network_degradation_supported`

Next nodes: `inspect_node_network_components`, `confirm_node_network_delay_from_node_status`, `inspect_service_logs_for_network_packet_loss`

### inspect_node_network_components

Goal: Inspect kube-proxy and containerd health once on the localized suspect node to distinguish component unavailability from component-healthy network delay or packet loss.

Actions:
- `CheckNodeServiceStatus` `{"node_name": "<node_name>", "service_name": "kube-proxy"}`
- `CheckNodeServiceStatus` `{"node_name": "<node_name>", "service_name": "containerd"}`

Expected evidence:
- `kube_proxy_status`
- `containerd_status`
- `node_network_component_status`
- `kube_proxy_unavailable_supported`
- `components_healthy_for_network_attribution`

Next nodes: `confirm_kube_proxy_unavailable`, `confirm_node_network_delay_from_node_status`, `inspect_service_logs_for_network_packet_loss`

### confirm_kube_proxy_unavailable

Goal: Confirm kube-proxy unavailability from the component status already collected by the shared node-network component inspection.

Actions:
- none (semantic classification uses already-admitted observations)

Expected evidence:
- `kube_proxy_status`
- `failing_node`
- `kube_proxy_unavailable_confirmed`

Next nodes: none

### confirm_node_network_delay_from_node_status

Goal: Confirm node_network_delay only when latency or response-delay degradation is correlated with the same suspect node and that node does not show a meaningful TCP Retransmission increase. Small fluctuations alone are insufficient to establish packet loss.

Actions:
- `GetAlerts` `{}`
- `DescribeResource` `{"name": "<node_name>", "resource_type": "node"}`

Expected evidence:
- `suspect_node`
- `node_correlated_latency_degradation`
- `no_high_magnitude_tcp_retransmission_on_suspect_node`
- `node_network_components_healthy`
- `node_network_delay_supported`

Next nodes: none

### inspect_service_logs_for_network_packet_loss

Goal: Confirm node_network_packet_loss from a meaningful TCP Retransmission increase on the same suspect node that hosts the affected services. Timeout, Unavailable, DeadlineExceeded, traffic-drop, or latency evidence is supportive but is not sufficient without the node-aligned retransmission signal.

Actions:
- `GetAlerts` `{}`
- `GetErrorLogs` `{"namespace": "<application_namespace>", "service_name": "<service_name>"}`

Expected evidence:
- `suspect_node`
- `high_magnitude_tcp_retransmission`
- `retransmission_node_matches_suspect_node`
- `affected_services_on_suspect_node`
- `packet_loss_supporting_timeouts`
- `node_network_packet_loss_supported`

Next nodes: none
