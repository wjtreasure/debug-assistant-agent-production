---
skill_id: image_pull_startup_failure
description: A pod cannot start because its container image cannot be pulled, appearing
  as ErrImagePull or ImagePullBackOff.
source_sha256: 3471cf7b0a157ee51484dc63e7113350c436ae564599c1d8af7b65844a15799f
derived_sha256: df631c5c5180f02f6586c269a3ee9318c06a905930566445b0fc3c3082abb57e
signals:
- pod status is ErrImagePull
- pod status is ImagePullBackOff
- pod cannot pull container image
- failing pod is non-Running because of image pull
entry:
- locate_failing_startup_pod
allowed_tools: &id001
- get_resources
- describe_resource
- get_app_yaml
expected_evidence: &id002
- active_alerts_status
- pod_health_status
- pod_status_summary
- failing_pod_identified
- image_pull_error_message
- pull_failure_reason
- failing_service_image_registry
- healthy_service_image_registry
- registry_isolation_confirmed
- no_image_pull_secret_issue
- deployment_image_tag
- healthy_service_tag
- image_pull_secrets_status
- incorrect_image_reference_confirmed
- deployment_yaml
- image_pull_secrets_presence
- image_reference
- missing_image_pull_secret_indicator
- image_registry_requirement
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
  objective: A pod cannot start because its container image cannot be pulled, appearing
    as ErrImagePull or ImagePullBackOff.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_failing_startup_pod:
    goal: Locate the failing pod causing service disruption by inspecting pod readiness
      and status reasons in the application namespace.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
    evidence:
    - active_alerts_status
    - pod_health_status
    - pod_status_summary
    - failing_pod_identified
    next:
    - classify_image_pull_failure
  classify_image_pull_failure:
    goal: Classify the image pull failure reason from failing pod events by extracting
      pull error messages and distinguishing between DNS failure, incorrect image
      reference, and registry authorization failure.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <failing_pod_name>
        namespace: <application_namespace>
    evidence:
    - image_pull_error_message
    - pull_failure_reason
    next:
    - confirm_registry_dns_failure
    - confirm_incorrect_image_reference
    - confirm_missing_image_pull_secret
  confirm_registry_dns_failure:
    goal: Verify that the DNS failure is specific to the failing image registry by
      comparing registries across services.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_service_name>
    - tool: GetAppYAML
      args:
        app_name: <healthy_service_name>
    evidence:
    - failing_service_image_registry
    - healthy_service_image_registry
    - registry_isolation_confirmed
    - no_image_pull_secret_issue
    next: []
  confirm_incorrect_image_reference:
    goal: Confirm that the failing deployment image tag is incorrect by inspecting
      deployment YAML, comparing with healthy services, and checking image pull secrets.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing-service-name>
    evidence:
    - deployment_image_tag
    - healthy_service_tag
    - image_pull_secrets_status
    - incorrect_image_reference_confirmed
    next: []
  confirm_missing_image_pull_secret:
    goal: Confirm whether the deployment lacks imagePullSecrets configuration given
      registry authorization failure.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing-service>
    evidence:
    - deployment_yaml
    - image_pull_secrets_presence
    - image_reference
    - missing_image_pull_secret_indicator
    - image_registry_requirement
    next: []
---

# image_pull_startup_failure

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_app_yaml`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_failing_startup_pod

Goal: Locate the failing pod causing service disruption by inspecting pod readiness and status reasons in the application namespace.

Actions:
- `GetResources` `{"namespace": "<namespace>", "resource_type": "pods"}`

Expected evidence:
- `active_alerts_status`
- `pod_health_status`
- `pod_status_summary`
- `failing_pod_identified`

Next nodes: `classify_image_pull_failure`

### classify_image_pull_failure

Goal: Classify the image pull failure reason from failing pod events by extracting pull error messages and distinguishing between DNS failure, incorrect image reference, and registry authorization failure.

Actions:
- `DescribeResource` `{"name": "<failing_pod_name>", "namespace": "<application_namespace>", "resource_type": "pod"}`

Expected evidence:
- `image_pull_error_message`
- `pull_failure_reason`

Next nodes: `confirm_registry_dns_failure`, `confirm_incorrect_image_reference`, `confirm_missing_image_pull_secret`

### confirm_registry_dns_failure

Goal: Verify that the DNS failure is specific to the failing image registry by comparing registries across services.

Actions:
- `GetAppYAML` `{"app_name": "<failing_service_name>"}`
- `GetAppYAML` `{"app_name": "<healthy_service_name>"}`

Expected evidence:
- `failing_service_image_registry`
- `healthy_service_image_registry`
- `registry_isolation_confirmed`
- `no_image_pull_secret_issue`

Next nodes: none

### confirm_incorrect_image_reference

Goal: Confirm that the failing deployment image tag is incorrect by inspecting deployment YAML, comparing with healthy services, and checking image pull secrets.

Actions:
- `GetAppYAML` `{"app_name": "<failing-service-name>"}`

Expected evidence:
- `deployment_image_tag`
- `healthy_service_tag`
- `image_pull_secrets_status`
- `incorrect_image_reference_confirmed`

Next nodes: none

### confirm_missing_image_pull_secret

Goal: Confirm whether the deployment lacks imagePullSecrets configuration given registry authorization failure.

Actions:
- `GetAppYAML` `{"app_name": "<failing-service>"}`

Expected evidence:
- `deployment_yaml`
- `image_pull_secrets_presence`
- `image_reference`
- `missing_image_pull_secret_indicator`
- `image_registry_requirement`

Next nodes: none
