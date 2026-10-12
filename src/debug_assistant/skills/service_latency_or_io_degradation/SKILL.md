---
skill_id: service_latency_or_io_degradation
description: An application service shows sustained response latency, throughput degradation,
  or abnormal filesystem-I/O activity.
source_sha256: da01a1d8620ab3ca43c743f0a9756a23f34446cbdc0b0fe2215dec59c57cae34
derived_sha256: 086355eddb826440761ab80ed621b66414f30dea6723221f084f17fa0f591928
signals:
- application service has severe latency or slow response
- service latency degradation is reported
- downstream timeout appears in service logs
- FILESYSTEM_WRITE_SURGE or abnormal fs_write_bytes is reported
entry:
- locate_service_latency_io_scope
allowed_tools: &id001
- get_alerts
- get_resources
- get_error_logs
- list_code_files
- read_file_when_source_workspace_is_bound
- get_service_dependencies
expected_evidence: &id002
- performance_alert_summary
- affected_services
- anomalous_service
- pod_health_summary
- pod_node_mapping
- latency_degradation_evidence
- filesystem_io_evidence
- affected_service
- pod_health_status
- pod_restart_count
- pod_node_placement
- error_log_pattern
- recent_log_pattern
- application_error_present
- downstream_timeout_found
- fixed_latency_pattern
- filesystem_write_surge
- connectivity_failure_pattern
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
- target_source_file
- artificial_delay_present
- non_delay_code_pattern
- root_cause_confirmed
- source_code_evidence
- busy_loop_found
- non_busy_loop_code_pattern
- root_cause_confirmation
- code_defect_pattern
- repeated_or_unbounded_file_write_path
- code_excessive_file_writes_confirmed
- observed_latency_degradation
- source_code_review_summary
- no_supported_latency_code_defect
- service_dependency_topology
- no_visible_downstream_bottleneck
- pod_network_delay_attributed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_alerts
- get_resources
- get_error_logs
- list_code_files
- read_file
- get_service_dependencies
registry_projection:
registry_projection:
  objective: An application service shows sustained response latency, throughput degradation,
    or abnormal filesystem-I/O activity.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_service_latency_io_scope:
    goal: Identify the service and pods affected by observable latency degradation
      or filesystem-I/O anomaly without assigning network or code ownership.
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
    - latency_degradation_evidence
    - filesystem_io_evidence
    next:
    - inspect_latency_io_service_context
  inspect_latency_io_service_context:
    goal: Inspect the latency- or I/O-affected service's pod health, logs, dependency
      context, and relevant configuration once to distinguish pod-network delay, artificial
      delay, file I/O code patterns, and ordinary application failure.
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
    - tool: GetRecentLogs
      args:
        namespace: <namespace>
        service_name: <service_name>
    evidence:
    - affected_service
    - pod_health_status
    - pod_restart_count
    - pod_node_placement
    - error_log_pattern
    - recent_log_pattern
    - application_error_present
    - downstream_timeout_found
    - fixed_latency_pattern
    - filesystem_write_surge
    - connectivity_failure_pattern
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
    - confirm_artificial_delay_code
    - confirm_busy_loop_code
    - confirm_excessive_file_writes_code
    - attribute_pod_network_delay_after_code_exclusion
  confirm_artificial_delay_code:
    goal: Confirm code_artificial_delay only when an explicit delay operation is active
      with a meaningful non-zero duration on the failing execution path. A dormant,
      zero-valued, or unreachable delay hook is insufficient. Active spin or busy-wait
      code remains code_busy_loop even if a sleep expression also exists.
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: main.go
    evidence:
    - source_file_list
    - target_source_file
    - artificial_delay_present
    - non_delay_code_pattern
    - root_cause_confirmed
    next: []
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
  confirm_excessive_file_writes_code:
    goal: Confirm whether repeated or unbounded file writes in the affected service's
      source code explain FILESYSTEM_WRITE_SURGE / abnormal fs_write_bytes.
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: <relevant_source_file>
    evidence:
    - source_code_evidence
    - repeated_or_unbounded_file_write_path
    - code_excessive_file_writes_confirmed
    next: []
  attribute_pod_network_delay_after_code_exclusion:
    goal: Attribute remaining service latency to pod_network_delay after supported
      code-level and dependency explanations are inspected and not found.
    actions:
    - tool: GetServiceDependencies
      args:
        service_name: <service_name>
    evidence:
    - observed_latency_degradation
    - source_code_review_summary
    - no_supported_latency_code_defect
    - service_dependency_topology
    - no_visible_downstream_bottleneck
    - pod_network_delay_attributed
    next: []
---

# service_latency_or_io_degradation

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_alerts`, `get_resources`, `get_error_logs`, `list_code_files`, `read_file`, `get_service_dependencies`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_service_latency_io_scope

Goal: Identify the service and pods affected by observable latency degradation or filesystem-I/O anomaly without assigning network or code ownership.

Actions:
- `GetAlerts` `{"namespace": "<namespace>"}`
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `performance_alert_summary`
- `affected_services`
- `anomalous_service`
- `pod_health_summary`
- `pod_node_mapping`
- `latency_degradation_evidence`
- `filesystem_io_evidence`

Next nodes: `inspect_latency_io_service_context`

### inspect_latency_io_service_context

Goal: Inspect the latency- or I/O-affected service's pod health, logs, dependency context, and relevant configuration once to distinguish pod-network delay, artificial delay, file I/O code patterns, and ordinary application failure.

Actions:
- `GetResources` `{"label_selector": "app=<service_name>", "namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`
- `GetErrorLogs` `{"namespace": "<namespace>", "service_name": "<service_name>"}`
- `GetRecentLogs` `{"namespace": "<namespace>", "service_name": "<service_name>"}`

Expected evidence:
- `affected_service`
- `pod_health_status`
- `pod_restart_count`
- `pod_node_placement`
- `error_log_pattern`
- `recent_log_pattern`
- `application_error_present`
- `downstream_timeout_found`
- `fixed_latency_pattern`
- `filesystem_write_surge`
- `connectivity_failure_pattern`
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

Next nodes: `confirm_artificial_delay_code`, `confirm_busy_loop_code`, `confirm_excessive_file_writes_code`, `attribute_pod_network_delay_after_code_exclusion`

### confirm_artificial_delay_code

Goal: Confirm code_artificial_delay only when an explicit delay operation is active with a meaningful non-zero duration on the failing execution path. A dormant, zero-valued, or unreachable delay hook is insufficient. Active spin or busy-wait code remains code_busy_loop even if a sleep expression also exists.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "main.go"}`

Expected evidence:
- `source_file_list`
- `target_source_file`
- `artificial_delay_present`
- `non_delay_code_pattern`
- `root_cause_confirmed`

Next nodes: none

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

### confirm_excessive_file_writes_code

Goal: Confirm whether repeated or unbounded file writes in the affected service's source code explain FILESYSTEM_WRITE_SURGE / abnormal fs_write_bytes.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<relevant_source_file>"}`

Expected evidence:
- `source_code_evidence`
- `repeated_or_unbounded_file_write_path`
- `code_excessive_file_writes_confirmed`

Next nodes: none

### attribute_pod_network_delay_after_code_exclusion

Goal: Attribute remaining service latency to pod_network_delay after supported code-level and dependency explanations are inspected and not found.

Actions:
- `GetServiceDependencies` `{"service_name": "<service_name>"}`

Expected evidence:
- `observed_latency_degradation`
- `source_code_review_summary`
- `no_supported_latency_code_defect`
- `service_dependency_topology`
- `no_visible_downstream_bottleneck`
- `pod_network_delay_attributed`

Next nodes: none
