---
skill_id: oom_or_memory_exhaustion
description: A container terminates or restarts with observable out-of-memory evidence
  such as OOMKilled or exit code 137.
source_sha256: 78ab9c39e2da7b209977f3426a1e7064b1da6bf4edcdccff458f339d1f5543ea
derived_sha256: f3c0b4d4c782ba98da9e7e409b7a30ed7a2c36dda1e9d3f87d9d81ef90890c6b
signals:
- pod or container is OOMKilled
- container exit code is 137
- pod restarts after memory exhaustion
- memory exhaustion is visible in pod status, events, or alerts
- service restarts after memory pressure
entry:
- locate_oom_affected_workload
allowed_tools: &id001
- get_resources
- describe_resource
- get_app_yaml
- list_code_files
- read_file_when_source_workspace_is_bound
expected_evidence: &id002
- failing_pod_identified
- failing_pod_name
- affected_workload
- failing_service
- symptom_service
- pod_status_summary
- pod_restart_counts
- container_termination_reason
- container_exit_code
- pod_events
- container_state
- container_memory_limit
- oom_kill_event
- memory_exhaustion_mechanism
- resource_configuration_check_needed
- deployment_resource_limits
- deployment_memory_limit
- memory_limit_value
- configuration_validity
- resource_limit_independently_anomalous
- resource_configuration_not_explanatory
- root_cause_confirmation
- deployment_memory_limit_insufficient
- source_file_content
- memory_allocation_pattern
- source_code_evidence
- repeated_or_unbounded_file_read_path
- memory_growth_from_file_reads
- code_excessive_file_reads_confirmed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_resources
- describe_resource
- get_app_yaml
- list_code_files
- read_file
registry_projection:
registry_projection:
  objective: A container terminates or restarts with observable out-of-memory evidence
    such as OOMKilled or exit code 137.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_oom_affected_workload:
    goal: Locate the pod, workload, and service affected by an observed OOM or memory-exhaustion
      restart without assigning configuration or code ownership.
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
    - pod_status_summary
    - pod_restart_counts
    next:
    - inspect_memory_exhaustion_mechanism
  inspect_memory_exhaustion_mechanism:
    goal: Establish whether OOMKilled or exit code 137 is the container termination
      mechanism and collect the observed memory limit without deciding whether configuration
      or application code caused the exhaustion.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <failing_pod>
        namespace: <namespace>
    evidence:
    - container_termination_reason
    - container_exit_code
    - pod_events
    - container_state
    - container_memory_limit
    - oom_kill_event
    - memory_exhaustion_mechanism
    - resource_configuration_check_needed
    next:
    - evaluate_oom_resource_configuration
  evaluate_oom_resource_configuration:
    goal: Read the affected workload YAML once and decide whether an independently
      anomalous memory limit explains the OOM or whether source-level behavior remains
      the leading branch. A finite limit plus OOMKilled is not sufficient evidence
      of misconfiguration. When configuration is not explanatory, keep generic allocation/retention
      and repeated or unbounded file reads as distinct competing source-code hypotheses.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_service>
    evidence:
    - deployment_resource_limits
    - deployment_memory_limit
    - memory_limit_value
    - configuration_validity
    - resource_limit_independently_anomalous
    - resource_configuration_not_explanatory
    next:
    - confirm_oom_limit_misconfiguration
    - confirm_memory_leak_code
    - confirm_excessive_file_reads_code
  confirm_oom_limit_misconfiguration:
    goal: Confirm container_memory_limit_too_low as a memory-limit configuration root
      cause only when the affected application container's configured limit is extremely
      small, for example only a few MiB and clearly unable to support normal startup
      or basic operation, so that the configuration independently explains the repeated
      OOM termination. An ordinary finite limit such as 128Mi is not sufficient evidence
      by itself. If the limit is not extremely small or the evidence is uncertain,
      reject this branch and inspect the still-open memory-leak and excessive-file-read
      source branches.
    actions: []
    evidence:
    - root_cause_confirmation
    - deployment_memory_limit_insufficient
    next: []
  confirm_memory_leak_code:
    goal: Confirm code_memory_leak only when retrieved source shows non-file allocation
      growth, unbounded collection growth, or retained references that explain the
      OOMKilled event. Repeated or unbounded file reads are a distinct code_excessive_file_reads
      root cause; if file I/O is the concrete growth mechanism, reject this branch
      and load confirm_excessive_file_reads_code.
    actions:
    - tool: ListCodeFiles
      args:
        app_name: <service_name>
    - tool: GetSourceCode
      args:
        app_name: <service_name>
        file_path: <relevant_source_file>
    evidence:
    - source_file_content
    - memory_allocation_pattern
    - root_cause_confirmation
    next: []
  confirm_excessive_file_reads_code:
    goal: 'Confirm code_excessive_file_reads when retrieved source shows a repeated
      or unbounded file-read path whose allocated or retained data explains the OOM
      after resource configuration is non-explanatory. This is distinct from code_memory_leak:
      when file I/O is the concrete growth mechanism, use this branch rather than
      generic memory-leak attribution.'
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
    - repeated_or_unbounded_file_read_path
    - memory_growth_from_file_reads
    - code_excessive_file_reads_confirmed
    next: []
---

# oom_or_memory_exhaustion

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_app_yaml`, `list_code_files`, `read_file`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_oom_affected_workload

Goal: Locate the pod, workload, and service affected by an observed OOM or memory-exhaustion restart without assigning configuration or code ownership.

Actions:
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `failing_pod_identified`
- `failing_pod_name`
- `affected_workload`
- `failing_service`
- `symptom_service`
- `pod_status_summary`
- `pod_restart_counts`

Next nodes: `inspect_memory_exhaustion_mechanism`

### inspect_memory_exhaustion_mechanism

Goal: Establish whether OOMKilled or exit code 137 is the container termination mechanism and collect the observed memory limit without deciding whether configuration or application code caused the exhaustion.

Actions:
- `DescribeResource` `{"name": "<failing_pod>", "namespace": "<namespace>", "resource_type": "pod"}`

Expected evidence:
- `container_termination_reason`
- `container_exit_code`
- `pod_events`
- `container_state`
- `container_memory_limit`
- `oom_kill_event`
- `memory_exhaustion_mechanism`
- `resource_configuration_check_needed`

Next nodes: `evaluate_oom_resource_configuration`

### evaluate_oom_resource_configuration

Goal: Read the affected workload YAML once and decide whether an independently anomalous memory limit explains the OOM or whether source-level behavior remains the leading branch. A finite limit plus OOMKilled is not sufficient evidence of misconfiguration. When configuration is not explanatory, keep generic allocation/retention and repeated or unbounded file reads as distinct competing source-code hypotheses.

Actions:
- `GetAppYAML` `{"app_name": "<failing_service>"}`

Expected evidence:
- `deployment_resource_limits`
- `deployment_memory_limit`
- `memory_limit_value`
- `configuration_validity`
- `resource_limit_independently_anomalous`
- `resource_configuration_not_explanatory`

Next nodes: `confirm_oom_limit_misconfiguration`, `confirm_memory_leak_code`, `confirm_excessive_file_reads_code`

### confirm_oom_limit_misconfiguration

Goal: Confirm container_memory_limit_too_low as a memory-limit configuration root cause only when the affected application container's configured limit is extremely small, for example only a few MiB and clearly unable to support normal startup or basic operation, so that the configuration independently explains the repeated OOM termination. An ordinary finite limit such as 128Mi is not sufficient evidence by itself. If the limit is not extremely small or the evidence is uncertain, reject this branch and inspect the still-open memory-leak and excessive-file-read source branches.

Actions:
- none (semantic classification uses already-admitted observations)

Expected evidence:
- `root_cause_confirmation`
- `deployment_memory_limit_insufficient`

Next nodes: none

### confirm_memory_leak_code

Goal: Confirm code_memory_leak only when retrieved source shows non-file allocation growth, unbounded collection growth, or retained references that explain the OOMKilled event. Repeated or unbounded file reads are a distinct code_excessive_file_reads root cause; if file I/O is the concrete growth mechanism, reject this branch and load confirm_excessive_file_reads_code.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<relevant_source_file>"}`

Expected evidence:
- `source_file_content`
- `memory_allocation_pattern`
- `root_cause_confirmation`

Next nodes: none

### confirm_excessive_file_reads_code

Goal: Confirm code_excessive_file_reads when retrieved source shows a repeated or unbounded file-read path whose allocated or retained data explains the OOM after resource configuration is non-explanatory. This is distinct from code_memory_leak: when file I/O is the concrete growth mechanism, use this branch rather than generic memory-leak attribution.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<relevant_source_file>"}`

Expected evidence:
- `source_code_evidence`
- `repeated_or_unbounded_file_read_path`
- `memory_growth_from_file_reads`
- `code_excessive_file_reads_confirmed`

Next nodes: none
