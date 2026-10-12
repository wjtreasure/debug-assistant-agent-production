---
skill_id: pending_pod_storage_dependency
description: A Pending pod reports that a PVC or another persistent-storage dependency
  is unresolved.
source_sha256: 9128d829283e325edf0618bf12a2ce778bb19dbd4db11c0a9a1e2e328f4641f4
derived_sha256: 3e5fbeeb4d37802651cfd675c07e30420f73b291cc18b766311e4d38f134dc70
signals:
- pod is Pending with PVC-related event
- PersistentVolumeClaim is unbound
- scheduler reports unbound immediate PersistentVolumeClaims
- storage-backed workload cannot schedule because volume binding is unresolved
- pod is Pending because a PVC or storage dependency is missing
- expected PVC is absent
- pod events point to a storage claim creation dependency
entry:
- locate_pending_storage_workload
allowed_tools: &id001
- get_resources
- describe_resource
- get_app_yaml
expected_evidence: &id002
- pending_pod_identified
- pending_pod_name
- affected_workload
- failing_service
- pod_status_summary
- active_alerts_summary
- node_status_summary
- pod_events
- pod_pending_cause
- referenced_pvc_name
- pvc_object_exists
- pvc_object_missing
- pvc_status
- pending_pvc_identified
- storage_request
- missing_pvc_reason
- storage_quota_verification_candidate
- pvc_binding_failure_reason
- pvc_storage_request
- pvc_access_modes
- pvc_selector
- pvc_storage_class
- candidate_pv_name
- available_pv_count
- pv_binding_status
- pv_capacity
- pv_access_modes
- pv_labels
- pv_storage_class
- occupied_pv_suspected
- access_mode_mismatch_suspected
- capacity_mismatch_suspected
- selector_mismatch_suspected
- storage_class_mismatch_suspected
- pending_pvc_requirements
- compatible_pv_name
- pv_phase_bound
- pv_bound_claim
- pv_binding_occupied_confirmed
- pv_access_mode
- pvc_access_mode
- access_mode_mismatch_confirmed
- capacity_mismatch_confirmed
- dynamic_provisioning_available
- pvc_pv_comparison
- pvc_selector_labels
- selector_mismatch_confirmation
- available_storage_classes
- candidate_pv_storage_classes
- explicit_storage_class_incompatibility
- pvc_storage_class_mismatch_confirmed
- namespace_storage_quota_limit
- pvc_storage_exceeds_quota
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
  objective: A Pending pod reports that a PVC or another persistent-storage dependency
    is unresolved.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_pending_storage_workload:
    goal: Locate the Pending storage-backed pod and affected workload while retaining
      only the node/alert context needed to enter PVC dependency classification.
    actions:
    - tool: GetResources
      args:
        resource_type: pods
        namespace: <namespace>
        output_wide: true
    evidence:
    - pending_pod_identified
    - pending_pod_name
    - affected_workload
    - failing_service
    - pod_status_summary
    - active_alerts_summary
    - node_status_summary
    next:
    - classify_pod_pvc_dependency
  classify_pod_pvc_dependency:
    goal: Inspect the Pending pod once, identify its referenced PVC, and distinguish
      a missing PVC object from an existing Pending/unbound PVC.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pod
        name: <pending_pod>
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: persistentvolumeclaims
        namespace: <namespace>
    evidence:
    - pod_events
    - pod_pending_cause
    - referenced_pvc_name
    - pvc_object_exists
    - pvc_object_missing
    - pvc_status
    - pending_pvc_identified
    - storage_request
    - missing_pvc_reason
    - storage_quota_verification_candidate
    next:
    - classify_pvc_pv_compatibility
    - confirm_namespace_storage_quota
  classify_pvc_pv_compatibility:
    goal: Inspect the unbound PVC and available PVs once to classify occupied binding,
      access-mode, capacity, selector, or storage-class incompatibility. Evaluate
      a compatible PV's existing claimRef before interpreting the generic event "no
      persistent volumes available and no storage class is set"; an empty storageClassName
      is not itself storage-class mismatch.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pvc
        name: <pvc_name>
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: persistentvolumes
    evidence:
    - pvc_binding_failure_reason
    - pvc_status
    - pvc_storage_request
    - pvc_access_modes
    - pvc_selector
    - pvc_storage_class
    - candidate_pv_name
    - available_pv_count
    - pv_binding_status
    - pv_capacity
    - pv_access_modes
    - pv_labels
    - pv_storage_class
    - occupied_pv_suspected
    - access_mode_mismatch_suspected
    - capacity_mismatch_suspected
    - selector_mismatch_suspected
    - storage_class_mismatch_suspected
    next:
    - confirm_pv_binding_occupied
    - confirm_pvc_access_mode_mismatch
    - confirm_pvc_capacity_mismatch
    - confirm_pvc_selector_mismatch
    - confirm_pvc_storage_class_mismatch
  confirm_pv_binding_occupied:
    goal: Confirm pv_binding_occupied when a PV that otherwise satisfies the pending
      PVC's capacity, access mode, selector, and storage-class requirements is already
      Bound to a different PVC. The generic event "no persistent volumes available
      and no storage class is set" does not imply storage-class mismatch; inspect
      existing PV claimRef/binding first.
    actions:
    - tool: GetResources
      args:
        resource_type: persistentvolumes
    - tool: DescribeResource
      args:
        resource_type: pvc
        name: <bound_pvc>
        namespace: <namespace>
    evidence:
    - pending_pvc_requirements
    - compatible_pv_name
    - pv_phase_bound
    - pv_bound_claim
    - pv_binding_occupied_confirmed
    next: []
  confirm_pvc_access_mode_mismatch:
    goal: Compare the PVC's requested access mode and capacity with available PersistentVolumes
      to verify the binding mismatch.
    actions:
    - tool: GetResources
      args:
        resource_type: persistentvolumes
    - tool: GetAppYAML
      args:
        app_name: <failing_service_name>
    - tool: GetResources
      args:
        resource_type: storageclasses
    evidence:
    - pv_access_mode
    - pvc_access_mode
    - access_mode_mismatch_confirmed
    next: []
  confirm_pvc_capacity_mismatch:
    goal: Retrieve PVC specification and compare storage request, access modes, and
      selector with the available PV to confirm the binding failure root cause.
    actions:
    - tool: GetAppYAML
      args:
        app_name: the-application-using-the-unbound-pvc
    evidence:
    - pvc_storage_request
    - pv_capacity
    - capacity_mismatch_confirmed
    - dynamic_provisioning_available
    - pvc_pv_comparison
    next: []
  confirm_pvc_selector_mismatch:
    goal: Compare the PVC label selector requirements with the available PV labels
      to confirm a selector mismatch preventing binding.
    actions:
    - tool: GetAppYAML
      args:
        app_name: <failing_service_name>
    evidence:
    - pvc_selector_labels
    - pv_labels
    - selector_mismatch_confirmation
    next: []
  confirm_pvc_storage_class_mismatch:
    goal: Confirm pvc_storage_class_mismatch only when the PVC explicitly requests
      a nonexistent StorageClass or its non-empty storageClassName is incompatible
      with otherwise suitable PVs. A PVC with no storage class and a generic "no persistent
      volumes available" event is not enough; first evaluate occupied binding, capacity,
      access mode, and selector.
    actions:
    - tool: DescribeResource
      args:
        resource_type: pvc
        name: <pvc_name>
        namespace: <namespace>
    - tool: GetResources
      args:
        resource_type: storageclasses
    - tool: GetResources
      args:
        resource_type: persistentvolumes
    evidence:
    - pvc_storage_class
    - available_storage_classes
    - candidate_pv_storage_classes
    - explicit_storage_class_incompatibility
    - pvc_storage_class_mismatch_confirmed
    next: []
  confirm_namespace_storage_quota:
    goal: Verify whether a namespace ResourceQuota on storage requests is blocking
      PVC creation.
    actions:
    - tool: GetResources
      args:
        resource_type: resourcequotas
        namespace: <application_namespace>
    - tool: DescribeResource
      args:
        resource_type: resourcequota
        name: <resource_quota>
        namespace: <application_namespace>
    evidence:
    - namespace_storage_quota_limit
    - pvc_storage_exceeds_quota
    next: []
---

# pending_pod_storage_dependency

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_resources`, `describe_resource`, `get_app_yaml`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_pending_storage_workload

Goal: Locate the Pending storage-backed pod and affected workload while retaining only the node/alert context needed to enter PVC dependency classification.

Actions:
- `GetResources` `{"namespace": "<namespace>", "output_wide": true, "resource_type": "pods"}`

Expected evidence:
- `pending_pod_identified`
- `pending_pod_name`
- `affected_workload`
- `failing_service`
- `pod_status_summary`
- `active_alerts_summary`
- `node_status_summary`

Next nodes: `classify_pod_pvc_dependency`

### classify_pod_pvc_dependency

Goal: Inspect the Pending pod once, identify its referenced PVC, and distinguish a missing PVC object from an existing Pending/unbound PVC.

Actions:
- `DescribeResource` `{"name": "<pending_pod>", "namespace": "<namespace>", "resource_type": "pod"}`
- `GetResources` `{"namespace": "<namespace>", "resource_type": "persistentvolumeclaims"}`

Expected evidence:
- `pod_events`
- `pod_pending_cause`
- `referenced_pvc_name`
- `pvc_object_exists`
- `pvc_object_missing`
- `pvc_status`
- `pending_pvc_identified`
- `storage_request`
- `missing_pvc_reason`
- `storage_quota_verification_candidate`

Next nodes: `classify_pvc_pv_compatibility`, `confirm_namespace_storage_quota`

### classify_pvc_pv_compatibility

Goal: Inspect the unbound PVC and available PVs once to classify occupied binding, access-mode, capacity, selector, or storage-class incompatibility. Evaluate a compatible PV's existing claimRef before interpreting the generic event "no persistent volumes available and no storage class is set"; an empty storageClassName is not itself storage-class mismatch.

Actions:
- `DescribeResource` `{"name": "<pvc_name>", "namespace": "<namespace>", "resource_type": "pvc"}`
- `GetResources` `{"resource_type": "persistentvolumes"}`

Expected evidence:
- `pvc_binding_failure_reason`
- `pvc_status`
- `pvc_storage_request`
- `pvc_access_modes`
- `pvc_selector`
- `pvc_storage_class`
- `candidate_pv_name`
- `available_pv_count`
- `pv_binding_status`
- `pv_capacity`
- `pv_access_modes`
- `pv_labels`
- `pv_storage_class`
- `occupied_pv_suspected`
- `access_mode_mismatch_suspected`
- `capacity_mismatch_suspected`
- `selector_mismatch_suspected`
- `storage_class_mismatch_suspected`

Next nodes: `confirm_pv_binding_occupied`, `confirm_pvc_access_mode_mismatch`, `confirm_pvc_capacity_mismatch`, `confirm_pvc_selector_mismatch`, `confirm_pvc_storage_class_mismatch`

### confirm_pv_binding_occupied

Goal: Confirm pv_binding_occupied when a PV that otherwise satisfies the pending PVC's capacity, access mode, selector, and storage-class requirements is already Bound to a different PVC. The generic event "no persistent volumes available and no storage class is set" does not imply storage-class mismatch; inspect existing PV claimRef/binding first.

Actions:
- `GetResources` `{"resource_type": "persistentvolumes"}`
- `DescribeResource` `{"name": "<bound_pvc>", "namespace": "<namespace>", "resource_type": "pvc"}`

Expected evidence:
- `pending_pvc_requirements`
- `compatible_pv_name`
- `pv_phase_bound`
- `pv_bound_claim`
- `pv_binding_occupied_confirmed`

Next nodes: none

### confirm_pvc_access_mode_mismatch

Goal: Compare the PVC's requested access mode and capacity with available PersistentVolumes to verify the binding mismatch.

Actions:
- `GetResources` `{"resource_type": "persistentvolumes"}`
- `GetAppYAML` `{"app_name": "<failing_service_name>"}`
- `GetResources` `{"resource_type": "storageclasses"}`

Expected evidence:
- `pv_access_mode`
- `pvc_access_mode`
- `access_mode_mismatch_confirmed`

Next nodes: none

### confirm_pvc_capacity_mismatch

Goal: Retrieve PVC specification and compare storage request, access modes, and selector with the available PV to confirm the binding failure root cause.

Actions:
- `GetAppYAML` `{"app_name": "the-application-using-the-unbound-pvc"}`

Expected evidence:
- `pvc_storage_request`
- `pv_capacity`
- `capacity_mismatch_confirmed`
- `dynamic_provisioning_available`
- `pvc_pv_comparison`

Next nodes: none

### confirm_pvc_selector_mismatch

Goal: Compare the PVC label selector requirements with the available PV labels to confirm a selector mismatch preventing binding.

Actions:
- `GetAppYAML` `{"app_name": "<failing_service_name>"}`

Expected evidence:
- `pvc_selector_labels`
- `pv_labels`
- `selector_mismatch_confirmation`

Next nodes: none

### confirm_pvc_storage_class_mismatch

Goal: Confirm pvc_storage_class_mismatch only when the PVC explicitly requests a nonexistent StorageClass or its non-empty storageClassName is incompatible with otherwise suitable PVs. A PVC with no storage class and a generic "no persistent volumes available" event is not enough; first evaluate occupied binding, capacity, access mode, and selector.

Actions:
- `DescribeResource` `{"name": "<pvc_name>", "namespace": "<namespace>", "resource_type": "pvc"}`
- `GetResources` `{"resource_type": "storageclasses"}`
- `GetResources` `{"resource_type": "persistentvolumes"}`

Expected evidence:
- `pvc_storage_class`
- `available_storage_classes`
- `candidate_pv_storage_classes`
- `explicit_storage_class_incompatibility`
- `pvc_storage_class_mismatch_confirmed`

Next nodes: none

### confirm_namespace_storage_quota

Goal: Verify whether a namespace ResourceQuota on storage requests is blocking PVC creation.

Actions:
- `GetResources` `{"namespace": "<application_namespace>", "resource_type": "resourcequotas"}`
- `DescribeResource` `{"name": "<resource_quota>", "namespace": "<application_namespace>", "resource_type": "resourcequota"}`

Expected evidence:
- `namespace_storage_quota_limit`
- `pvc_storage_exceeds_quota`

Next nodes: none
