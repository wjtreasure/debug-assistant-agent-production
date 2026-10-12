---
skill_id: pod_crash_or_restart
description: A pod enters CrashLoopBackOff, exits abnormally, or repeatedly restarts.
source_sha256: 5ba5a33f3dbd635ad85dfa680129f2dd989881fb24c5688a3b17e55985fb006e
derived_sha256: bdb14314b1b0e015de331379b9d7a1f422f3379bf017ea48fe2d9fa3fd0682ac
signals:
- pod status is CrashLoopBackOff
- pod repeatedly restarts before becoming Ready
- container exits during startup or runtime
- application pod crashes or restarts
- service logs show panic or runtime exception
entry:
- locate_crashing_workload
allowed_tools: &id001
- get_resources
- describe_resource
- get_error_logs
- get_app_yaml
- list_code_files
- read_file_when_source_workspace_is_bound
expected_evidence: &id002
- failing_pod_identified
- failing_pod_name
- affected_workload
- failing_service
- symptom_service
- entrypoint_service
- pod_status_summary
- pod_readiness_status
- pod_restart_counts
- active_alerts_status
- container_exit_code
- container_termination_reason
- pod_events
- volume_mount_details
- liveness_probe_config
- container_port
- probe_failure_events
- crash_log_message
- panic_message
- stack_trace_present
- permission_denied_indicator
- probe_port_mismatch_suspected
- probe_protocol_mismatch_suspected
- repeated_probe_timeout
- container_memory_limit_too_low
- container_error_log
- permission_denied_confirmed
- volume_permission_denied_error
- security_context_uid
- deployment_probe_config
- readiness_probe_config
- probe_port_consistency
- port_mismatch_confirmed
- root_cause_confirmed
- deployment_yaml
- protocol_consistency
- probe_protocol_mismatch_confirmed
- effective_probe_configuration
- liveness_probe_timing_configuration
- liveness_probe_timing_config_from_yaml
- root_cause_confirmation
- error_log_pattern
- error_type
- service_logs_clean
- missing_parameter_hint
- source_file_list
- candidate_source_file_identified
- primary_source_file
- candidate_handler_files
- candidate_source_files
- source_file_candidates
- relevant_source_files_identified
- source_code_evidence
- required_parameter
- parameter_cleared_or_omitted
- code_missing_parameter_confirmed
- request_or_call_site
- swapped_arguments
- code_wrong_argument_order_confirmed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_resources
- describe_resource
- get_error_logs
- get_app_yaml
- list_code_files
- read_file
registry_projection:
registry_projection:
  objective: A pod enters CrashLoopBackOff, exits abnormally, or repeatedly restarts.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_crashing_workload:
    goal: Locate the pod and workload showing a crash or abnormal restart, and retain
      alert or service context without assigning a root-cause category.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    evidence:
    - failing_pod_identified
    - failing_pod_name
    - affected_workload
    - failing_service
    - symptom_service
    - entrypoint_service
    - pod_status_summary
    - pod_readiness_status
    - pod_restart_counts
    - active_alerts_status
    next:
    - inspect_crash_evidence
  inspect_crash_evidence:
    goal: Classify the observable crash mechanism by inspecting the affected pod's
      last termination state, events, volume and liveness-probe configuration, and
      crash logs before exposing a root-cause confirmation branch.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <failing_pod>
        namespace: <namespace>
    - tool: GetRecentLogs
      args:
        namespace: <namespace>
        service_name: <failing_service>
    evidence:
    - container_exit_code
    - container_termination_reason
    - pod_events
    - pod_restart_counts
    - volume_mount_details
    - liveness_probe_config
    - container_port
    - probe_failure_events
    - crash_log_message
    - panic_message
    - stack_trace_present
    - permission_denied_indicator
    - probe_port_mismatch_suspected
    - probe_protocol_mismatch_suspected
    - repeated_probe_timeout
    - container_memory_limit_too_low
    next:
    - confirm_volume_mount_permission_denied
    - confirm_probe_port_mismatch
    - confirm_probe_protocol_mismatch
    - confirm_liveness_probe_timing
    - classify_missing_parameter_logs
    - locate_relevant_source_files
  confirm_volume_mount_permission_denied:
    goal: Confirm whether container logs contain a permission denied error on the
      mounted volume that explains the pod crash.
    actions:
    - tool: GetRecentLogs
      args:
        namespace: the affected namespace
        service_name: the backing datastore service
    evidence:
    - container_error_log
    - permission_denied_confirmed
    - volume_permission_denied_error
    - security_context_uid
    next: []
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
  confirm_liveness_probe_timing:
    goal: Confirm whether liveness probe timing misconfiguration causes pod restarts
      by inspecting pod events, container state, and probe timing configuration.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <failing_pod>
        namespace: <namespace>
    evidence:
    - probe_failure_events
    - effective_probe_configuration
    - container_termination_reason
    - pod_events
    - liveness_probe_timing_configuration
    - liveness_probe_timing_config_from_yaml
    - root_cause_confirmation
    next: []
  classify_missing_parameter_logs:
    goal: Inspect runtime and error logs of a suspected service to classify error
      pattern or exclude the service.
    actions:
    - tool: GetRecentLogs
      args:
        namespace: <application_namespace>
        service_name: <service_name>
    - tool: GetErrorLogs
      args:
        namespace: <application_namespace>
        service_name: <service_name>
    evidence:
    - error_log_pattern
    - error_type
    - service_logs_clean
    - failing_service
    - missing_parameter_hint
    next:
    - locate_relevant_source_files
    - confirm_missing_parameter_code
  locate_relevant_source_files:
    goal: List source files for a localized service to identify candidate files for
      code-level defect inspection.
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    evidence:
    - source_file_list
    - candidate_source_file_identified
    - primary_source_file
    - candidate_handler_files
    - candidate_source_files
    - source_file_candidates
    - relevant_source_files_identified
    next:
    - confirm_missing_parameter_code
    - confirm_wrong_argument_order_code
  confirm_missing_parameter_code:
    goal: 'Confirm code_missing_parameter when required request data is made absent
      before use: omitted from a request, assigned nil, overwritten with an empty
      string, or replaced by an empty/default value. This includes code such as currency
      = "" or productID = "" even when the downstream consequence is an error response.
      Do not reinterpret that downstream error as code_wrong_return, and require an
      actual swap of existing meaningful values before considering code_wrong_argument_order.'
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: <candidate_source_file>
    evidence:
    - source_code_evidence
    - required_parameter
    - parameter_cleared_or_omitted
    - code_missing_parameter_confirmed
    next: []
  confirm_wrong_argument_order_code:
    goal: Confirm code_wrong_argument_order only when two or more existing meaningful
      values are supplied to the wrong parameter or request-field positions, such
      as value A being passed where B belongs and value B where A belongs. A value
      merely cleared, omitted, nil, empty, or defaulted is code_missing_parameter,
      not wrong argument order.
    actions:
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: <candidate_source_file>
    evidence:
    - source_code_evidence
    - request_or_call_site
    - swapped_arguments
    - code_wrong_argument_order_confirmed
    next: []
---

# pod_crash_or_restart

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_error_logs`, `get_app_yaml`, `list_code_files`, `read_file`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_crashing_workload

Goal: Locate the pod and workload showing a crash or abnormal restart, and retain alert or service context without assigning a root-cause category.

Actions:
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `failing_pod_identified`
- `failing_pod_name`
- `affected_workload`
- `failing_service`
- `symptom_service`
- `entrypoint_service`
- `pod_status_summary`
- `pod_readiness_status`
- `pod_restart_counts`
- `active_alerts_status`

Next nodes: `inspect_crash_evidence`

### inspect_crash_evidence

Goal: Classify the observable crash mechanism by inspecting the affected pod's last termination state, events, volume and liveness-probe configuration, and crash logs before exposing a root-cause confirmation branch.

Actions:
- `DescribeResource` `{"name": "<failing_pod>", "namespace": "<namespace>", "resource_type": "pod"}`
- `GetRecentLogs` `{"namespace": "<namespace>", "service_name": "<failing_service>"}`

Expected evidence:
- `container_exit_code`
- `container_termination_reason`
- `pod_events`
- `pod_restart_counts`
- `volume_mount_details`
- `liveness_probe_config`
- `container_port`
- `probe_failure_events`
- `crash_log_message`
- `panic_message`
- `stack_trace_present`
- `permission_denied_indicator`
- `probe_port_mismatch_suspected`
- `probe_protocol_mismatch_suspected`
- `repeated_probe_timeout`
- `container_memory_limit_too_low`

Next nodes: `confirm_volume_mount_permission_denied`, `confirm_probe_port_mismatch`, `confirm_probe_protocol_mismatch`, `confirm_liveness_probe_timing`, `classify_missing_parameter_logs`, `locate_relevant_source_files`

### confirm_volume_mount_permission_denied

Goal: Confirm whether container logs contain a permission denied error on the mounted volume that explains the pod crash.

Actions:
- `GetRecentLogs` `{"namespace": "the affected namespace", "service_name": "the backing datastore service"}`

Expected evidence:
- `container_error_log`
- `permission_denied_confirmed`
- `volume_permission_denied_error`
- `security_context_uid`

Next nodes: none

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

### confirm_liveness_probe_timing

Goal: Confirm whether liveness probe timing misconfiguration causes pod restarts by inspecting pod events, container state, and probe timing configuration.

Actions:
- `DescribeResource` `{"name": "<failing_pod>", "namespace": "<namespace>", "resource_type": "pod"}`

Expected evidence:
- `probe_failure_events`
- `effective_probe_configuration`
- `container_termination_reason`
- `pod_events`
- `liveness_probe_timing_configuration`
- `liveness_probe_timing_config_from_yaml`
- `root_cause_confirmation`

Next nodes: none

### classify_missing_parameter_logs

Goal: Inspect runtime and error logs of a suspected service to classify error pattern or exclude the service.

Actions:
- `GetRecentLogs` `{"namespace": "<application_namespace>", "service_name": "<service_name>"}`
- `GetErrorLogs` `{"namespace": "<application_namespace>", "service_name": "<service_name>"}`

Expected evidence:
- `error_log_pattern`
- `error_type`
- `service_logs_clean`
- `failing_service`
- `missing_parameter_hint`

Next nodes: `locate_relevant_source_files`, `confirm_missing_parameter_code`

### locate_relevant_source_files

Goal: List source files for a localized service to identify candidate files for code-level defect inspection.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`

Expected evidence:
- `source_file_list`
- `candidate_source_file_identified`
- `primary_source_file`
- `candidate_handler_files`
- `candidate_source_files`
- `source_file_candidates`
- `relevant_source_files_identified`

Next nodes: `confirm_missing_parameter_code`, `confirm_wrong_argument_order_code`

### confirm_missing_parameter_code

Goal: Confirm code_missing_parameter when required request data is made absent before use: omitted from a request, assigned nil, overwritten with an empty string, or replaced by an empty/default value. This includes code such as currency = "" or productID = "" even when the downstream consequence is an error response. Do not reinterpret that downstream error as code_wrong_return, and require an actual swap of existing meaningful values before considering code_wrong_argument_order.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<candidate_source_file>"}`

Expected evidence:
- `source_code_evidence`
- `required_parameter`
- `parameter_cleared_or_omitted`
- `code_missing_parameter_confirmed`

Next nodes: none

### confirm_wrong_argument_order_code

Goal: Confirm code_wrong_argument_order only when two or more existing meaningful values are supplied to the wrong parameter or request-field positions, such as value A being passed where B belongs and value B where A belongs. A value merely cleared, omitted, nil, empty, or defaulted is code_missing_parameter, not wrong argument order.

Actions:
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<candidate_source_file>"}`

Expected evidence:
- `source_code_evidence`
- `request_or_call_site`
- `swapped_arguments`
- `code_wrong_argument_order_confirmed`

Next nodes: none
