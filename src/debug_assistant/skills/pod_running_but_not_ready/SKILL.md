---
skill_id: pod_running_but_not_ready
description: A pod is Running but does not become Ready, such as showing READY 0/1.
source_sha256: b86921b6b1ffaab3c612a9bdca6e149c13f65b6bbfb33e25dcef98df1fe325fe
derived_sha256: dea8532eec441b823a1208940d14ece346e501f9002145bd5473e4d1f3dd199c
signals:
- pod is Running but NotReady
- pod readiness is 0/1
- workload exists but readiness gate never becomes healthy
entry:
- locate_runtime_affected_pod
allowed_tools: &id001
- get_resources
- describe_resource
- get_app_yaml
expected_evidence: &id002
- pod_runtime_status
- pod_readiness_status
- pod_restart_counts
- failing_pod_identified
- abnormal_pod_identified
- no_failing_pod_identified
- alert_anomaly_status
- readiness_probe_config
- container_port
- probe_failure_events
- suspected_port_mismatch
- readiness_probe_protocol_mismatch_suspected
- probe_protocol_consistency
- readiness_probe_incorrect_protocol
- deployment_probe_config
- probe_port_consistency
- port_mismatch_confirmed
- root_cause_confirmed
- deployment_yaml
- liveness_probe_config
- protocol_consistency
- probe_protocol_mismatch_confirmed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_resources
- describe_resource
- get_app_yaml
registry_projection:
registry_projection:
  objective: A pod is Running but does not become Ready, such as showing READY 0/1.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_runtime_affected_pod:
    goal: Locate the pod-level runtime anomaly by listing pods and checking readiness,
      runtime status, and restart counts.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <application_namespace>
        output_wide: true
    evidence:
    - pod_runtime_status
    - pod_readiness_status
    - pod_restart_counts
    - failing_pod_identified
    - abnormal_pod_identified
    - no_failing_pod_identified
    - alert_anomaly_status
    next:
    - classify_readiness_probe_failure
  classify_readiness_probe_failure:
    goal: Classify readiness probe failure reason by inspecting NotReady pod events
      and probe configuration — port mismatch or protocol mismatch.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <failing_pod>
        namespace: <namespace>
    evidence:
    - readiness_probe_config
    - container_port
    - probe_failure_events
    - suspected_port_mismatch
    - readiness_probe_protocol_mismatch_suspected
    - probe_protocol_consistency
    - readiness_probe_incorrect_protocol
    next:
    - confirm_probe_port_mismatch
    - confirm_probe_protocol_mismatch
  confirm_probe_port_mismatch:
    goal: Confirm liveness or readiness probe port mismatch from deployment or pod
      configuration.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_app>
    evidence:
    - deployment_probe_config
    - readiness_probe_config
    - container_port
    - probe_port_consistency
    - port_mismatch_confirmed
    - root_cause_confirmed
    next: []
  confirm_probe_protocol_mismatch:
    goal: Confirm liveness or readiness probe protocol mismatch by inspecting workload
      YAML probe protocol settings.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_app>
    evidence:
    - deployment_yaml
    - liveness_probe_config
    - readiness_probe_config
    - protocol_consistency
    - probe_protocol_mismatch_confirmed
    - root_cause_confirmed
    next: []
---

# pod_running_but_not_ready

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_app_yaml`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_runtime_affected_pod

Goal: Locate the pod-level runtime anomaly by listing pods and checking readiness, runtime status, and restart counts.

Actions:
- `GetResources` `{"namespace": "<application_namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `pod_runtime_status`
- `pod_readiness_status`
- `pod_restart_counts`
- `failing_pod_identified`
- `abnormal_pod_identified`
- `no_failing_pod_identified`
- `alert_anomaly_status`

Next nodes: `classify_readiness_probe_failure`

### classify_readiness_probe_failure

Goal: Classify readiness probe failure reason by inspecting NotReady pod events and probe configuration — port mismatch or protocol mismatch.

Actions:
- `DescribeResource` `{"name": "<failing_pod>", "namespace": "<namespace>", "resource_type": "pod"}`

Expected evidence:
- `readiness_probe_config`
- `container_port`
- `probe_failure_events`
- `suspected_port_mismatch`
- `readiness_probe_protocol_mismatch_suspected`
- `probe_protocol_consistency`
- `readiness_probe_incorrect_protocol`

Next nodes: `confirm_probe_port_mismatch`, `confirm_probe_protocol_mismatch`

### confirm_probe_port_mismatch

Goal: Confirm liveness or readiness probe port mismatch from deployment or pod configuration.

Actions:
- `GetAppYAML` `{"app_name": "<failing_app>"}`

Expected evidence:
- `deployment_probe_config`
- `readiness_probe_config`
- `container_port`
- `probe_port_consistency`
- `port_mismatch_confirmed`
- `root_cause_confirmed`

Next nodes: none

### confirm_probe_protocol_mismatch

Goal: Confirm liveness or readiness probe protocol mismatch by inspecting workload YAML probe protocol settings.

Actions:
- `GetAppYAML` `{"app_name": "<failing_app>"}`

Expected evidence:
- `deployment_yaml`
- `liveness_probe_config`
- `readiness_probe_config`
- `protocol_consistency`
- `probe_protocol_mismatch_confirmed`
- `root_cause_confirmed`

Next nodes: none
