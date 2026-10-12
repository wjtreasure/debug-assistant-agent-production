---
skill_id: kubernetes_object_creation_failure
description: An expected Kubernetes Pod, workload instance, or Service object is not
  created or has no ready instance.
source_sha256: 3b8ec1d5863329c91c738601c6d9af93cfeebddb0923caaad50ba6097e20f9c8
derived_sha256: 0772990ab2833e4b0cee8250c9aa4322fd2b9aa80c15e18d6be3138df6209465
signals:
- workload has missing pods
- deployment has zero ready replicas
- service has empty endpoints because backend pods are absent
- ReplicaSet or deployment reports FailedCreate
- pod creation failed before a pod became Running
- expected Service object is missing
- connectivity probe localizes a missing or unavailable Service object
entry:
- locate_creation_failure_scope
allowed_tools: &id001
- get_resources
- describe_resource
- get_service_dependencies
- check_service_connectivity
- get_app_yaml
expected_evidence: &id002
- affected_workload
- affected_deployment
- pod_existence_status
- deployment_readiness
- pod_creation_failure_indicator
- service_inventory
- missing_service_name
- missing_service_object_confirmed
- empty_endpoints_due_to_absent_backends
- admission_creation_suspected
- deployment_conditions
- deployment_condition_failure
- deployment_replica_status
- associated_replicaset_name
- replicaset_reference
- replicaset_creation_errors
- admission_failure_reason
- admission_error_message
- service_account_name
- suspected_missing_service_account
- quota_exceeded_event
- quota_type_exceeded
- pod_quota_limit_reached
- creation_failure_reason
- service_account_reference
- existing_service_accounts
- missing_service_account_confirmed
- competing_quota_evidence
- root_cause_confirmation
- namespace_quota_status
- quota_resource_type
- quota_used_value
- quota_hard_limit
- quota_exceeded_evidence
- cpu_quota_exhausted
- memory_request_quota_match
- pod_quota_exhausted_confirmed
- pod_memory_request
- quota_limit_match
- unreachable_service_name
- connectivity_status
- suspicious_admission_signal
- namespace_service_quota_exceeded
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_resources
- describe_resource
- get_service_dependencies
- check_service_connectivity
- get_app_yaml
registry_projection:
registry_projection:
  objective: An expected Kubernetes Pod, workload instance, or Service object is not
    created or has no ready instance.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_creation_failure_scope:
    goal: Establish whether the observed creation symptom is missing workload pods,
      zero-ready deployment, or a missing Service object, and identify the affected
      workload without repeating namespace resource inventory.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    - tool: GetResources
      args:
        resource_type: services
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: deployments
        namespace: <namespace>
    evidence:
    - affected_workload
    - affected_deployment
    - pod_existence_status
    - deployment_readiness
    - pod_creation_failure_indicator
    - service_inventory
    - missing_service_name
    - missing_service_object_confirmed
    - empty_endpoints_due_to_absent_backends
    - admission_creation_suspected
    next:
    - classify_pod_creation_admission_failure
    - locate_unreachable_downstream_service
    - confirm_namespace_service_quota
  classify_pod_creation_admission_failure:
    goal: Classify the reason for pod creation failure by inspecting deployment conditions,
      ReplicaSet events, and pod template/admission evidence.
    actions:
    - tool: DescribeResource
      args:
        resource_type: deployment
        name: <target_workload>
        namespace: <application_namespace>
    evidence:
    - deployment_conditions
    - deployment_condition_failure
    - deployment_replica_status
    - associated_replicaset_name
    - replicaset_reference
    - replicaset_creation_errors
    - admission_failure_reason
    - admission_error_message
    - service_account_name
    - suspected_missing_service_account
    - quota_exceeded_event
    - quota_type_exceeded
    - pod_quota_limit_reached
    - creation_failure_reason
    next:
    - confirm_missing_service_account
    - confirm_namespace_pod_resource_quota
  confirm_missing_service_account:
    goal: Confirm whether the workload references a ServiceAccount that does not exist.
    actions:
    - tool: GetResources
      args:
        resource_type: serviceaccounts
        namespace: <application_namespace>
    evidence:
    - service_account_reference
    - existing_service_accounts
    - missing_service_account_confirmed
    - competing_quota_evidence
    - root_cause_confirmation
    next: []
  confirm_namespace_pod_resource_quota:
    goal: Confirm whether a namespace ResourceQuota for pod count, CPU requests, or
      memory requests is exhausted and blocking pod creation.
    actions:
    - tool: GetResources
      args:
        resource_type: resourcequotas
        namespace: <application_namespace>
    evidence:
    - namespace_quota_status
    - quota_resource_type
    - quota_used_value
    - quota_hard_limit
    - quota_exceeded_evidence
    - cpu_quota_exhausted
    - memory_request_quota_match
    - pod_quota_exhausted_confirmed
    - pod_memory_request
    - quota_limit_match
    - root_cause_confirmation
    next: []
  locate_unreachable_downstream_service:
    goal: Using service dependencies and connectivity probes, locate which downstream
      service is unreachable.
    actions:
    - tool: GetServiceDependencies
      args:
        service_name: <entrypoint_workload>
    - tool: CheckServiceConnectivity
      args:
        service_name: <target_workload>
        port: 7070
        namespace: <application_namespace>
    - tool: CheckServiceConnectivity
      args:
        service_name: <target_workload>
        port: 8080
        namespace: <application_namespace>
    - tool: CheckServiceConnectivity
      args:
        service_name: <target_workload>
        port: 9555
        namespace: <application_namespace>
    evidence:
    - unreachable_service_name
    - connectivity_status
    next:
    - confirm_namespace_service_quota
  confirm_namespace_service_quota:
    goal: Confirm that the missing service is due to a namespace service quota exceeded
      by inspecting admission signals and verifying the quota.
    actions:
    - tool: DescribeResource
      args:
        resource_type: service
        name: <target_workload>
        namespace: <application_namespace>
    - tool: GetAppYAML
      args:
        app_name: <target_workload>
    - tool: GetResources
      args:
        resource_type: services
        namespace: <application_namespace>
        label_selector: app=<target_workload>
    evidence:
    - suspicious_admission_signal
    - namespace_quota_status
    - namespace_service_quota_exceeded
    next: []
---

# kubernetes_object_creation_failure

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_service_dependencies`, `check_service_connectivity`, `get_app_yaml`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_creation_failure_scope

Goal: Establish whether the observed creation symptom is missing workload pods, zero-ready deployment, or a missing Service object, and identify the affected workload without repeating namespace resource inventory.

Actions:
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`
- `GetResources` `{"namespace": "<namespace>", "resource_type": "services"}`
- `GetResources` `{"namespace": "<namespace>", "resource_type": "deployments"}`

Expected evidence:
- `affected_workload`
- `affected_deployment`
- `pod_existence_status`
- `deployment_readiness`
- `pod_creation_failure_indicator`
- `service_inventory`
- `missing_service_name`
- `missing_service_object_confirmed`
- `empty_endpoints_due_to_absent_backends`
- `admission_creation_suspected`

Next nodes: `classify_pod_creation_admission_failure`, `locate_unreachable_downstream_service`, `confirm_namespace_service_quota`

### classify_pod_creation_admission_failure

Goal: Classify the reason for pod creation failure by inspecting deployment conditions, ReplicaSet events, and pod template/admission evidence.

Actions:
- `DescribeResource` `{"name": "<target_workload>", "namespace": "<application_namespace>", "resource_type": "deployment"}`

Expected evidence:
- `deployment_conditions`
- `deployment_condition_failure`
- `deployment_replica_status`
- `associated_replicaset_name`
- `replicaset_reference`
- `replicaset_creation_errors`
- `admission_failure_reason`
- `admission_error_message`
- `service_account_name`
- `suspected_missing_service_account`
- `quota_exceeded_event`
- `quota_type_exceeded`
- `pod_quota_limit_reached`
- `creation_failure_reason`

Next nodes: `confirm_missing_service_account`, `confirm_namespace_pod_resource_quota`

### confirm_missing_service_account

Goal: Confirm whether the workload references a ServiceAccount that does not exist.

Actions:
- `GetResources` `{"namespace": "<application_namespace>", "resource_type": "serviceaccounts"}`

Expected evidence:
- `service_account_reference`
- `existing_service_accounts`
- `missing_service_account_confirmed`
- `competing_quota_evidence`
- `root_cause_confirmation`

Next nodes: none

### confirm_namespace_pod_resource_quota

Goal: Confirm whether a namespace ResourceQuota for pod count, CPU requests, or memory requests is exhausted and blocking pod creation.

Actions:
- `GetResources` `{"namespace": "<application_namespace>", "resource_type": "resourcequotas"}`

Expected evidence:
- `namespace_quota_status`
- `quota_resource_type`
- `quota_used_value`
- `quota_hard_limit`
- `quota_exceeded_evidence`
- `cpu_quota_exhausted`
- `memory_request_quota_match`
- `pod_quota_exhausted_confirmed`
- `pod_memory_request`
- `quota_limit_match`
- `root_cause_confirmation`

Next nodes: none

### locate_unreachable_downstream_service

Goal: Using service dependencies and connectivity probes, locate which downstream service is unreachable.

Actions:
- `GetServiceDependencies` `{"service_name": "<entrypoint_workload>"}`
- `CheckServiceConnectivity` `{"namespace": "<application_namespace>", "port": 7070, "service_name": "<target_workload>"}`
- `CheckServiceConnectivity` `{"namespace": "<application_namespace>", "port": 8080, "service_name": "<target_workload>"}`
- `CheckServiceConnectivity` `{"namespace": "<application_namespace>", "port": 9555, "service_name": "<target_workload>"}`

Expected evidence:
- `unreachable_service_name`
- `connectivity_status`

Next nodes: `confirm_namespace_service_quota`

### confirm_namespace_service_quota

Goal: Confirm that the missing service is due to a namespace service quota exceeded by inspecting admission signals and verifying the quota.

Actions:
- `DescribeResource` `{"name": "<target_workload>", "namespace": "<application_namespace>", "resource_type": "service"}`
- `GetAppYAML` `{"app_name": "<target_workload>"}`
- `GetResources` `{"label_selector": "app=<target_workload>", "namespace": "<application_namespace>", "resource_type": "services"}`

Expected evidence:
- `suspicious_admission_signal`
- `namespace_quota_status`
- `namespace_service_quota_exceeded`

Next nodes: none
