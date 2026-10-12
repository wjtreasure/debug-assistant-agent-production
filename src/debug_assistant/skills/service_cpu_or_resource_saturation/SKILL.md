---
skill_id: service_cpu_or_resource_saturation
description: An application service shows sustained CPU saturation or throttling with
  associated performance degradation.
source_sha256: 64bb5afd4265aad49af85833af5571151a6c4e1247157588a15bdcde50632cdf
derived_sha256: 0586b2a1969c65ea7eec6300783b966b3ee7f1fad02473fbc7e61d7f9aba8e41
signals:
- service has CPU throttling alert
- application service has CPU saturation
- pod CPU usage or throttling is abnormally high
- service performance degradation is dominated by CPU pressure
entry:
- locate_service_cpu_scope
allowed_tools: &id001
- get_alerts
- get_resources
- get_error_logs
- get_app_yaml
- list_code_files
- read_file_when_source_workspace_is_bound
expected_evidence: &id002
- performance_alert_summary
- affected_services
- anomalous_service
- pod_health_summary
- pod_node_mapping
- cpu_throttling_evidence
- affected_service
- pod_health_status
- pod_restart_count
- pod_node_placement
- error_log_pattern
- application_error_present
- cpu_saturation
- resource_limits
- configuration_normal
- code_level_performance_candidate
- local_failure_excluded
- source_file_list
- candidate_source_files
- primary_source_file
- source_code_pattern
- artificial_delay_candidate
- busy_loop_candidate
- excessive_file_read_candidate
- excessive_file_write_candidate
- no_obvious_code_defect
- source_code_evidence
- busy_loop_found
- non_busy_loop_code_pattern
- root_cause_confirmation
- code_defect_pattern
- observed_cpu_saturation
- source_code_review_summary
- no_supported_cpu_code_defect
- cpu_limit_context
- pod_cpu_overload_attributed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_alerts
- get_resources
- get_error_logs
- get_app_yaml
- list_code_files
- read_file
registry_projection:
registry_projection:
  objective: An application service shows sustained CPU saturation or throttling with
    associated performance degradation.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_service_cpu_scope:
    goal: Identify the service and pods affected by observable CPU throttling or CPU
      saturation without assigning resource-limit or code ownership.
    actions:
    - tool: GetAlerts
      args:
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    evidence:
    - performance_alert_summary
    - affected_services
    - anomalous_service
    - pod_health_summary
    - pod_node_mapping
    - cpu_throttling_evidence
    next:
    - inspect_cpu_service_context
  inspect_cpu_service_context:
    goal: Inspect the CPU-affected service's pod health, application logs, and resource
      configuration once to distinguish CPU-limit pressure, code-driven CPU usage,
      and ordinary application failure. A configured CPU limit together with cpu_cfs
      throttling proves quota pressure only; it does not establish that the limit
      configuration or pod_cpu_overload is the root cause, because code-driven demand
      such as a busy loop can produce the same evidence. Continue to source triage
      before attributing pod_cpu_overload.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        label_selector: app=<service_name>
        output_wide: true
    - tool: GetErrorLogs
      args:
        namespace: <namespace>
        service_name: <service_name>
    - tool: GetAppYAML
      args:
        app_name: <service_name>
    evidence:
    - affected_service
    - pod_health_status
    - pod_restart_count
    - pod_node_placement
    - error_log_pattern
    - application_error_present
    - cpu_saturation
    - resource_limits
    - configuration_normal
    - code_level_performance_candidate
    - local_failure_excluded
    next:
    - triage_performance_source
  triage_performance_source:
    goal: Locate the affected service's relevant source files and inspect the primary
      candidate code once before selecting a specific artificial-delay, busy-loop,
      excessive-read, or excessive-write confirmation.
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: <candidate_source_file>
    evidence:
    - source_file_list
    - candidate_source_files
    - primary_source_file
    - source_code_pattern
    - artificial_delay_candidate
    - busy_loop_candidate
    - excessive_file_read_candidate
    - excessive_file_write_candidate
    - no_obvious_code_defect
    next:
    - confirm_busy_loop_code
    - attribute_observed_cpu_overload
  confirm_busy_loop_code:
    goal: Confirm code_busy_loop when an active CPU-intensive loop or busy-wait explains
      the observed CPU saturation or latency. Concrete active-spin evidence remains
      a busy loop even if the same path also contains a sleep expression; a genuinely
      blocking delay without active spinning belongs to code_artificial_delay.
    actions:
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: main.go
    evidence:
    - source_code_evidence
    - busy_loop_found
    - non_busy_loop_code_pattern
    - root_cause_confirmation
    - code_defect_pattern
    next: []
  attribute_observed_cpu_overload:
    goal: Attribute the observed CPU saturation to pod_cpu_overload only after source
      inspection finds no busy-loop or other supported code-level CPU defect.
    actions: []
    evidence:
    - observed_cpu_saturation
    - source_code_review_summary
    - no_supported_cpu_code_defect
    - cpu_limit_context
    - pod_cpu_overload_attributed
    next: []
---

# service_cpu_or_resource_saturation

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_alerts`, `get_resources`, `get_error_logs`, `get_app_yaml`, `list_code_files`, `read_file`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_service_cpu_scope

Goal: Identify the service and pods affected by observable CPU throttling or CPU saturation without assigning resource-limit or code ownership.

Actions:
- `GetAlerts` `{"namespace": "<namespace>"}`
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `performance_alert_summary`
- `affected_services`
- `anomalous_service`
- `pod_health_summary`
- `pod_node_mapping`
- `cpu_throttling_evidence`

Next nodes: `inspect_cpu_service_context`

### inspect_cpu_service_context

Goal: Inspect the CPU-affected service's pod health, application logs, and resource configuration once to distinguish CPU-limit pressure, code-driven CPU usage, and ordinary application failure. A configured CPU limit together with cpu_cfs throttling proves quota pressure only; it does not establish that the limit configuration or pod_cpu_overload is the root cause, because code-driven demand such as a busy loop can produce the same evidence. Continue to source triage before attributing pod_cpu_overload.

Actions:
- `GetResources` `{"label_selector": "app=<service_name>", "namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`
- `GetErrorLogs` `{"namespace": "<namespace>", "service_name": "<service_name>"}`
- `GetAppYAML` `{"app_name": "<service_name>"}`

Expected evidence:
- `affected_service`
- `pod_health_status`
- `pod_restart_count`
- `pod_node_placement`
- `error_log_pattern`
- `application_error_present`
- `cpu_saturation`
- `resource_limits`
- `configuration_normal`
- `code_level_performance_candidate`
- `local_failure_excluded`

Next nodes: `triage_performance_source`

### triage_performance_source

Goal: Locate the affected service's relevant source files and inspect the primary candidate code once before selecting a specific artificial-delay, busy-loop, excessive-read, or excessive-write confirmation.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<candidate_source_file>"}`

Expected evidence:
- `source_file_list`
- `candidate_source_files`
- `primary_source_file`
- `source_code_pattern`
- `artificial_delay_candidate`
- `busy_loop_candidate`
- `excessive_file_read_candidate`
- `excessive_file_write_candidate`
- `no_obvious_code_defect`

Next nodes: `confirm_busy_loop_code`, `attribute_observed_cpu_overload`

### confirm_busy_loop_code

Goal: Confirm code_busy_loop when an active CPU-intensive loop or busy-wait explains the observed CPU saturation or latency. Concrete active-spin evidence remains a busy loop even if the same path also contains a sleep expression; a genuinely blocking delay without active spinning belongs to code_artificial_delay.

Actions:
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "main.go"}`

Expected evidence:
- `source_code_evidence`
- `busy_loop_found`
- `non_busy_loop_code_pattern`
- `root_cause_confirmation`
- `code_defect_pattern`

Next nodes: none

### attribute_observed_cpu_overload

Goal: Attribute the observed CPU saturation to pod_cpu_overload only after source inspection finds no busy-loop or other supported code-level CPU defect.

Actions:
- none (semantic classification uses already-admitted observations)

Expected evidence:
- `observed_cpu_saturation`
- `source_code_review_summary`
- `no_supported_cpu_code_defect`
- `cpu_limit_context`
- `pod_cpu_overload_attributed`

Next nodes: none
