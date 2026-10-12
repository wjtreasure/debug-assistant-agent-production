---
skill_id: application_request_or_response_failure
description: An application request reaches a service but returns an HTTP, RPC, or
  business-level error while basic service connectivity remains available.
source_sha256: d4305c5871f2aed3bbfeae57d891d6e3cd7e74523a2549a22ffae6964295fb22
derived_sha256: ecece63a941f2b188ecad21d5adad640e9553dbf7b60a2b47bce98aececdd973
signals:
- HTTP 400, HTTP 500, or HTTP 503 is observed while downstream connectivity succeeds
- RPC request returns an application or business-semantics error
- downstream Service is reachable but the operation still fails
- logs show nil, missing, misplaced, or semantically invalid request values
- application returns an incorrect response despite healthy pods and routing
entry:
- locate_request_failure_owner
allowed_tools: &id001
- get_error_logs
- get_service_dependencies
- list_code_files
- read_file_when_source_workspace_is_bound
expected_evidence: &id002
- symptom_service
- affected_operation
- error_pattern
- request_failure_type
- current_service_role
- upstream_callers
- downstream_callee
- code_owner_candidate
- connectivity_available
- source_file_list
- primary_source_file
- request_construction_site
- missing_parameter_candidate
- wrong_argument_order_candidate
- wrong_return_candidate
- no_supported_request_code_defect
- source_code_evidence
- required_parameter
- parameter_cleared_or_omitted
- code_missing_parameter_confirmed
- request_or_call_site
- swapped_arguments
- code_wrong_argument_order_confirmed
- valid_inputs_reach_return_path
- incorrect_return_statement_or_response
- code_wrong_return_confirmed
completion_rule: &id003
  owner: runtime
  rule: current incident EvidenceSet satisfies the active VerificationObligation
  skill_text_can_close_obligation: false
runtime_allowed_tools:
- get_error_logs
- get_service_dependencies
- list_code_files
- read_file
registry_projection:
registry_projection:
  objective: An application request reaches a service but returns an HTTP, RPC, or
    business-level error while basic service connectivity remains available.
  allowed_tools: *id001
  expected_evidence: *id002
  completion_rule: *id003
nodes:
  locate_request_failure_owner:
    goal: Localize the service, operation, and caller/callee ownership of HTTP, RPC,
      or business request errors after basic service connectivity remains available.
    actions:
    - tool: GetErrorLogs
      args:
        namespace: <namespace>
        service_name: <symptom_service>
    - tool: GetRecentLogs
      args:
        namespace: <namespace>
        service_name: <symptom_service>
    - tool: GetServiceDependencies
      args:
        service_name: <symptom_service>
    evidence:
    - symptom_service
    - affected_operation
    - error_pattern
    - request_failure_type
    - current_service_role
    - upstream_callers
    - downstream_callee
    - code_owner_candidate
    - connectivity_available
    next:
    - triage_request_source
  triage_request_source:
    goal: Locate and inspect the likely code owner's relevant request-handling source
      files once before selecting missing-parameter, wrong-argument-order, or wrong-return
      confirmation.
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
    - primary_source_file
    - request_construction_site
    - missing_parameter_candidate
    - wrong_argument_order_candidate
    - wrong_return_candidate
    - no_supported_request_code_defect
    next:
    - confirm_missing_parameter_code
    - confirm_wrong_argument_order_code
    - confirm_wrong_return_code
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
  confirm_wrong_return_code:
    goal: 'Confirm code_wrong_return only when the defect is at the return statement,
      response construction, or status-code selection itself: valid inputs reach the
      function, but it unconditionally or incorrectly returns an error, wrong status,
      or wrong response object. An error caused by input data being cleared, omitted,
      nil, empty, defaulted, or passed to the wrong position does not by itself confirm
      code_wrong_return.'
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
    - valid_inputs_reach_return_path
    - incorrect_return_statement_or_response
    - code_wrong_return_confirmed
    next: []
---

# application_request_or_response_failure

This project Skill preserves the audited Cloud-OpsBench public domain prior while using the local Markdown Skill interface.

## Runtime boundary

- This file supplies investigation guidance, signals, tool affinity, and expected evidence.
- Runtime owns tool permission, Observation provenance, Evidence admission, Obligation lifecycle, and convergence.
- Skill text and node completion never close an Obligation directly; closure requires the shared EvidenceSet completion contract.

## Local Tool interface

Runtime adapter allowlist: `get_error_logs`, `get_service_dependencies`, `list_code_files`, `read_file`.
Official action names in the node descriptions are domain-prior names; execution uses only these local capabilities.

## Investigation nodes

### locate_request_failure_owner

Goal: Localize the service, operation, and caller/callee ownership of HTTP, RPC, or business request errors after basic service connectivity remains available.

Actions:
- `GetErrorLogs` `{"namespace": "<namespace>", "service_name": "<symptom_service>"}`
- `GetRecentLogs` `{"namespace": "<namespace>", "service_name": "<symptom_service>"}`
- `GetServiceDependencies` `{"service_name": "<symptom_service>"}`

Expected evidence:
- `symptom_service`
- `affected_operation`
- `error_pattern`
- `request_failure_type`
- `current_service_role`
- `upstream_callers`
- `downstream_callee`
- `code_owner_candidate`
- `connectivity_available`

Next nodes: `triage_request_source`

### triage_request_source

Goal: Locate and inspect the likely code owner's relevant request-handling source files once before selecting missing-parameter, wrong-argument-order, or wrong-return confirmation.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<candidate_source_file>"}`

Expected evidence:
- `source_file_list`
- `primary_source_file`
- `request_construction_site`
- `missing_parameter_candidate`
- `wrong_argument_order_candidate`
- `wrong_return_candidate`
- `no_supported_request_code_defect`

Next nodes: `confirm_missing_parameter_code`, `confirm_wrong_argument_order_code`, `confirm_wrong_return_code`

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

### confirm_wrong_return_code

Goal: Confirm code_wrong_return only when the defect is at the return statement, response construction, or status-code selection itself: valid inputs reach the function, but it unconditionally or incorrectly returns an error, wrong status, or wrong response object. An error caused by input data being cleared, omitted, nil, empty, defaulted, or passed to the wrong position does not by itself confirm code_wrong_return.

Actions:
- `ListCodeFiles` `{"app_name": "<service_name>"}`
- `GetSourceCode` `{"app_name": "<service_name>", "file_path": "<candidate_source_file>"}`

Expected evidence:
- `source_code_evidence`
- `valid_inputs_reach_return_path`
- `incorrect_return_statement_or_response`
- `code_wrong_return_confirmed`

Next nodes: none
