from __future__ import annotations
import json
import copy
import time
from pydantic import ValidationError
from debug_assistant.contracts import ReflectionContract, ReflectionDecision, ObligationReview, compact_validation_error, render_contract, render_contract_compact
from debug_assistant.llm.base import LLMDeadlineExceeded, complete_json_compat, extract_json
from debug_assistant.incidents.contracts import (
    ReflectionContradictionReview, ReflectionFeedback, ReflectionObligationReview,
)


class ReflectionContractExhausted(ValueError):
    """Incident Reflection core schema remained invalid after one repair."""

SYSTEM="""You are a critical reviewer for a read-only debugging agent. Detect goal drift, premature certainty, unsupported claims and direct falsifying evidence. Do not invent evidence. Explicitly state the strongest current diagnosis and whether repository evidence is sufficient to support a specific causal mechanism and location. Evidence IDs must come from the context.

Use required_missing_evidence very narrowly: it contains only facts without which the current causal diagnosis cannot reasonably be supported. Put useful but nonessential confirmation in optional_validation instead. Optional validation must not by itself block finalization once the causal diagnosis is supported.

Also emit a compact structured root-cause identity. root_cause_target should be the most specific causal symbol/component you currently believe is responsible. root_cause_location should be a repository file path when known. root_cause_mechanism should be a concise mechanism statement. If target or mechanism is not yet known, return null rather than an object, list, or invented string. A partial diagnosis is valid and should not be discarded merely because one structured field is unknown. Keep known fields stable when the underlying diagnosis has not changed.

contradicting_evidence_ids contains only evidence that directly falsifies the proposed causal explanation. A buggy test expectation, an alternative implementation detail, incomplete information, missing validation, or a failing test consistent with the bug is NOT a contradiction unless it directly disproves the diagnosis. Keep the state concise."""

REPAIR_SYSTEM="""Repair one invalid reflection JSON object. Preserve its meaning and evidence IDs. Change only fields required to satisfy the supplied schema. For unknown scalar root-cause fields use null, never an object/list or invented content. Return exactly one corrected JSON object and nothing else."""


def _clear_partial_requirement_ranges(data):
    """Keep a malformed line hint from invalidating an otherwise usable reflection.

    A single line number is not a trustworthy range. Preserve the requirement's
    semantic target and scope, but represent that uncertain range as unknown.
    Strict contract validation remains unchanged for direct callers.
    """
    if not isinstance(data, dict):
        return data
    result = copy.deepcopy(data)
    for field in ("required_missing_evidence", "optional_validation", "new_requirements"):
        rows = result.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            start, end = row.get("line_start"), row.get("line_end")
            if (start is None) ^ (end is None) or (
                isinstance(start, int) and isinstance(end, int) and end < start
            ):
                row["line_start"] = None
                row["line_end"] = None
    return result


class Reflector:
    def __init__(self,llm,model='',compact_prompt=False):
        self.llm=llm; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}; self.last_repair_attempted=False


    @staticmethod
    def _sanitize_individual_reviews(data):
        """Drop only schema-invalid ObligationReview rows; preserve the rest of Reflection.

        This lets Runtime build a valid-subset candidate transaction without making one
        malformed review erase independent valid reviews from the same model response.
        """
        if not isinstance(data,dict):
            return data,[]
        rows=data.get('obligation_reviews')
        if not isinstance(rows,list):
            return data,[]
        valid=[]; invalid=[]
        for idx,row in enumerate(rows):
            try:
                valid.append(ObligationReview.model_validate(row).model_dump())
            except ValidationError as exc:
                oid=str(row.get('obligation_id') or '') if isinstance(row,dict) else ''
                invalid.append({'obligation_id':oid,'index':idx,'reason':'schema_invalid','errors':compact_validation_error(exc)})
        out=dict(data); out['obligation_reviews']=valid
        return out,invalid

    def review(self,context:str,logical_timeout_seconds:float|None=None,on_attempt_started=None):
        contract=(render_contract_compact(ReflectionContract,"REFLECTION_SCHEMA") if self.compact_prompt else render_contract(ReflectionContract,"REFLECTION_SCHEMA"))
        extra="Choose finish when repository evidence already supports a specific mechanism and location and no required causal gap remains. hypothesis_changed is telemetry only; the Harness independently fingerprints diagnostic state. root_cause_target/root_cause_location are the primary stability identity, while current_diagnosis may remain natural language. Return at most the most important required/optional items allowed by the schema. For every required_missing_evidence and optional_validation item, populate goal_type explicitly whenever possible: location, behavior, causality, caller, test, history, or contradiction. Use history only for actual version/commit/diff evidence needs; the mere word regression does not make a behavioral requirement historical. Every EvidenceRequirement must be atomic: one verifiable source scope (one file+symbol or one exact range), never a free-text A-and-B multi-source requirement. OPEN_CRITICAL_EVIDENCE_OBLIGATIONS may include obligation_id values. When source for an existing obligation is physically present in this Reflection context, emit exactly one obligation_reviews entry for that obligation: resolved if the shown source answers it, still_open if the shown source is insufficient, or refine with a more precise refined_requirement when the real causal question is delegated elsewhere. Do not review an obligation whose source is not shown in this Reflection prompt."
        user=f"{context}\n\n{contract}\n{extra}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'contract_chars':len(contract),'instruction_chars':len(extra),'repair_attempted':False}
        started=time.monotonic()
        def remaining_timeout():
            if logical_timeout_seconds is None:
                return None
            remaining=float(logical_timeout_seconds)-(time.monotonic()-started)
            if remaining <= 0:
                raise LLMDeadlineExceeded("Reflection logical deadline exhausted before schema repair")
            return remaining
        data=complete_json_compat(self.llm,SYSTEM,user,model=self.model or None,logical_timeout_seconds=remaining_timeout(),on_attempt_started=on_attempt_started)
        self.last_repair_attempted=False
        data = _clear_partial_requirement_ranges(data)
        data,invalid_reviews=self._sanitize_individual_reviews(data)
        try:
            out=ReflectionContract.model_validate(data).model_dump(); out['_invalid_obligation_reviews']=invalid_reviews; return out
        except ValidationError as exc:
            # One bounded repair is cheaper and safer than discarding the entire
            # hypothesis transition because a nullable scalar was returned as an object.
            self.last_repair_attempted=True
            self.last_prompt_breakdown['repair_attempted']=True
            repair_user=(f"INVALID_REFLECTION:\n{json.dumps(data,ensure_ascii=False,default=str)}\n\n"
                         f"VALIDATION_ERRORS:\n{json.dumps(compact_validation_error(exc),ensure_ascii=False)}\n\n{contract}")
            repaired=complete_json_compat(self.llm,REPAIR_SYSTEM,repair_user,model=self.model or None,logical_timeout_seconds=remaining_timeout())
            repaired,repair_invalid_reviews=self._sanitize_individual_reviews(repaired)
            try:
                out=ReflectionContract.model_validate(repaired).model_dump(); out['_invalid_obligation_reviews']=invalid_reviews+repair_invalid_reviews; return out
            except ValidationError as exc2:
                raise ValueError(f"reflection schema validation failed after one repair: {compact_validation_error(exc2)}") from exc2


class TypedReflection:
    """Provider-capability-aware semantic reflection; state derivation lives in Reducer."""
    def __init__(self, llm, model=""):
        self.llm, self.model = llm, model

    def review(self, context: str, *, logical_timeout_seconds=None, on_attempt_started=None) -> ReflectionDecision:
        from debug_assistant.llm.base import ProviderCapabilities
        caps = getattr(self.llm, "capabilities", ProviderCapabilities())
        schema_prompt = render_contract(ReflectionDecision, "REFLECTION_DECISION_SCHEMA")
        system = "Interpret repository evidence semantically. Never emit derived status, gaps, or sufficiency."
        user = f"{context}\n\n{schema_prompt}"
        if caps.json_schema and hasattr(self.llm, "complete_structured"):
            response = self.llm.complete_structured(system, user, schema=ReflectionDecision,
                                                    model=self.model or None,
                                                    logical_timeout_seconds=logical_timeout_seconds,
                                                    on_attempt_started=on_attempt_started)
            data = response.structured
            if data is None:
                from debug_assistant.llm.base import extract_json
                data = extract_json(response.content)
        else:
            data = complete_json_compat(self.llm, system, user, model=self.model or None,
                                        logical_timeout_seconds=logical_timeout_seconds,
                                        on_attempt_started=on_attempt_started)
        # Native typed reflection must have the same per-review fault isolation as
        # the legacy reflector. One malformed refine row must not discard otherwise
        # valid diagnosis/evidence input or turn a recoverable model defect into a
        # whole reflection failure.
        data = _clear_partial_requirement_ranges(data)
        data, _invalid_reviews = Reflector._sanitize_individual_reviews(data)
        return ReflectionDecision.model_validate(data)


def _normalize_incident_reflection_metadata(data):
    """Normalize only unambiguous optional Incident Reflection structures.

    ``reason`` and the flat Evidence-ID arrays remain strict.  The two review
    collections are optional explanatory metadata, so malformed siblings can
    be isolated while valid rows survive.  Explicit non-ev references are left
    untouched and therefore fail the core Evidence contract instead of being
    silently removed.
    """
    if not isinstance(data, dict):
        return data, (), (), ()

    result = dict(data)
    actions = []
    drops = []
    warnings = []
    known_fields = set(ReflectionFeedback.model_fields)
    extra_fields = sorted(set(result) - known_fields)
    for field in extra_fields:
        result.pop(field, None)
    if extra_fields:
        actions.append("top_level:ignored_extra_fields:" + ",".join(extra_fields))
        warnings.append("top_level:ignored_unknown_fields:" + ",".join(extra_fields))

    review_models = {
        "obligation_reviews": ReflectionObligationReview,
        "contradiction_reviews": ReflectionContradictionReview,
    }
    for field_name, model_type in review_models.items():
        if field_name not in result:
            continue
        raw = result[field_name]
        if isinstance(raw, tuple):
            raw = list(raw)
            result[field_name] = raw
            actions.append(f"{field_name}:tuple_to_array")
        elif isinstance(raw, dict):
            required = {
                name for name, field in model_type.model_fields.items() if field.is_required()
            }
            if not required.issubset(raw):
                result.pop(field_name, None)
                drops.append(f"{field_name}:block:ambiguous_object_shape")
                warnings.append(f"{field_name}:discarded_ambiguous_object")
                continue
            raw = [raw]
            result[field_name] = raw
            actions.append(f"{field_name}:object_to_singleton_array")
        elif not isinstance(raw, list):
            result.pop(field_name, None)
            drops.append(f"{field_name}:block:non_array")
            warnings.append(
                f"{field_name}:discarded_non_array:{type(raw).__name__}"
            )
            continue

        valid = []
        allowed_fields = set(model_type.model_fields)
        for index, item in enumerate(raw):
            item_path = f"{field_name}[{index}]"
            if not isinstance(item, dict):
                drops.append(f"{item_path}:non_object")
                warnings.append(f"{item_path}:discarded_non_object:{type(item).__name__}")
                continue
            explicit_non_ev = []
            if field_name == "obligation_reviews":
                values = item.get("supporting_evidence_ids")
                values = values if isinstance(values, (list, tuple)) else [values]
                explicit_non_ev.extend(
                    value for value in values
                    if isinstance(value, str) and not value.startswith("ev-")
                )
            else:
                value = item.get("evidence_id")
                if isinstance(value, str) and not value.startswith("ev-"):
                    explicit_non_ev.append(value)
            # Keep the invalid row intact.  Strict Pydantic validation and the
            # existing one-shot repair path must handle this core violation.
            if explicit_non_ev:
                valid.append(item)
                continue
            unknown_fields = sorted(set(item) - allowed_fields)
            if unknown_fields:
                actions.append(
                    f"{item_path}:ignored_extra_fields:{','.join(unknown_fields)}"
                )
                warnings.append(
                    f"{item_path}:ignored_unknown_fields:{','.join(unknown_fields)}"
                )
            sanitized = {key: value for key, value in item.items() if key in allowed_fields}
            try:
                model_type.model_validate(sanitized)
            except ValidationError as exc:
                detail = ",".join(
                    f"{'.'.join(str(part) for part in error.get('loc', ())) or 'item'}:"
                    f"{error.get('type', 'validation_error')}"
                    for error in exc.errors(include_url=False)[:4]
                ) or "validation_error"
                drops.append(f"{item_path}:schema_invalid")
                warnings.append(f"{item_path}:discarded_invalid:{detail}")
                continue
            valid.append(sanitized)
        result[field_name] = valid

    return result, tuple(actions), tuple(drops), tuple(warnings)


class IncidentReflectionAgent:
    """Triggered, tool-less semantic reflection for the Incident runtime.

    The agent receives a compact structured snapshot and returns feedback only.
    DiagnosisHarness validates Evidence IDs and decides how that feedback changes
    control flow; this class never executes a Tool or mutates a Hypothesis.
    """

    _SYSTEM = """You are the Incident Reflection Agent. Review the compact structured
    diagnosis state supplied by the Harness. Do not call tools, access ground truth,
    invent evidence, or replace the current hypothesis. Return only structured feedback:
    identify supported/unsupported claims, remaining gaps, obligation and contradiction
    reviews, and emit a structured hypothesis_delta plus proposed_evidence_gaps,
    proposed_obligations, contradiction_updates, and next_action_constraint when
    appropriate. Do not emit obligation IDs, gap IDs, or hypothesis versions; Runtime
    allocates those only after canonical novelty validation. All cited Evidence IDs must
    be existing ev-* IDs from the input. When
    application source is declared and available, treat an unknown or gap source-mechanism
    status as an unresolved critical gap until bounded read_file CODE Evidence is present;
    do not infer source coverage from a runtime symptom. Do not return raw chain-of-thought
    or the complete action history."""

    def __init__(self, llm, model: str = ""):
        self.llm = llm
        self.model = model
        self.last_usage: dict = {}
        self.last_call_count = 0
        self.last_schema_repaired = False
        self.last_schema_error = ""
        self.last_failure_type = ""
        self.last_normalization_actions: list[str] = []
        self.last_metadata_drops: list[str] = []
        self.last_metadata_warnings: list[str] = []
        self.last_prompt_breakdown: dict = {}
        self.last_prompt_breakdowns: list[dict] = []

    def reflect(self, snapshot: dict, *, logical_timeout_seconds: float | None = None,
                on_attempt_started=None, prompt_budget=None) -> ReflectionFeedback:
        from debug_assistant.contracts import compact_validation_error, render_contract
        self.last_usage = {}
        self.last_call_count = 0
        self.last_schema_repaired = False
        self.last_schema_error = ""
        self.last_failure_type = ""
        self.last_normalization_actions = []
        self.last_metadata_drops = []
        self.last_metadata_warnings = []
        self.last_prompt_breakdowns = []
        contract = render_contract(ReflectionFeedback, "INCIDENT_REFLECTION_SCHEMA")
        user = (
            "STRUCTURED_REFLECTION_INPUT:\n"
            + json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
            + "\n\n"
            + contract
        )
        self.last_prompt_breakdown = {
            "system_chars": len(self._SYSTEM),
            "context_chars": len(user),
            "raw_history_included": False,
        }
        if prompt_budget is not None:
            decision = prompt_budget.check_prompt(
                "reflection", self._SYSTEM, user,
                breakdown={"observation": user},
            )
            self.last_prompt_breakdown.update({
                "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                "input_hard_capacity": decision.input_hard_capacity,
                "budget_state": decision.state.value,
                "token_breakdown": dict(decision.breakdown),
            })
        logical_deadline = (
            None if logical_timeout_seconds is None
            else time.monotonic() + max(0.0, float(logical_timeout_seconds))
        )

        def remaining_timeout() -> float | None:
            if logical_deadline is None:
                return None
            remaining = logical_deadline - time.monotonic()
            if remaining <= 0:
                raise LLMDeadlineExceeded(
                    "Incident Reflection logical timeout exhausted before schema repair"
                )
            return remaining

        def complete_feedback(system: str, prompt: str):
            if prompt_budget is not None:
                decision = prompt_budget.check_prompt(
                    "reflection", system, prompt,
                    breakdown={"observation": prompt},
                )
                self.last_prompt_breakdown.update({
                    "last_estimated_prompt_tokens": decision.estimated_prompt_tokens,
                    "last_budget_state": decision.state.value,
                    "last_token_breakdown": dict(decision.breakdown),
                })
                self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
            timeout = remaining_timeout()
            if hasattr(self.llm, "complete_json"):
                return complete_json_compat(
                    self.llm, system, prompt, model=self.model or None,
                    logical_timeout_seconds=timeout,
                    on_attempt_started=on_attempt_started,
                )
            # A structured-only provider is a valid capability boundary.  Keep
            # the same JSON-shaped contract and extract the provider-neutral
            # response payload without adding another reflection implementation.
            method = getattr(self.llm, "complete_structured", None)
            if method is None:
                raise ValueError("provider has no JSON or structured reflection API")
            import inspect
            try:
                parameters = inspect.signature(method).parameters
                has_varkw = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values()
                )
            except (TypeError, ValueError):
                parameters, has_varkw = {}, False
            kwargs = {"model": self.model or None, "schema": ReflectionFeedback}
            if "logical_timeout_seconds" in parameters or has_varkw:
                kwargs["logical_timeout_seconds"] = timeout
            if "on_attempt_started" in parameters or has_varkw:
                kwargs["on_attempt_started"] = on_attempt_started
            response = method(system, prompt, **kwargs)
            structured = getattr(response, "structured", None)
            if structured is not None:
                return structured
            content = getattr(response, "content", response)
            return extract_json(content) if isinstance(content, str) else content

        self.last_call_count = 1
        try:
            raw = complete_feedback(self._SYSTEM, user)
        except LLMDeadlineExceeded:
            raise
        except Exception as first_exception:
            # Invalid JSON/provider payloads consume the same single repair
            # slot as typed schema failures; they never trigger open-ended
            # reflection retries.
            self._add_usage()
            self.last_schema_repaired = True
            self.last_prompt_breakdown["repair_attempted"] = True
            self.last_schema_error = (
                f"first-pass provider payload error: {type(first_exception).__name__}: "
                f"{first_exception}"
            )
            self.last_call_count += 1
            repair_user = (
                "INVALID_INCIDENT_REFLECTION_PROVIDER_PAYLOAD:\n"
                + json.dumps(str(first_exception), ensure_ascii=False)
                + "\n\n"
                + contract
            )
            try:
                repaired = complete_feedback(
                    "Repair only the JSON shape of Incident Reflection feedback; preserve all semantics and Evidence IDs.",
                    repair_user,
                )
            except LLMDeadlineExceeded:
                raise
            except Exception as repair_exception:
                self.last_failure_type = "reflection_contract_exhausted"
                self.last_schema_error += (
                    f"\nRepair provider payload error: {type(repair_exception).__name__}: "
                    f"{repair_exception}"
                )
                raise ReflectionContractExhausted(
                    "incident reflection provider payload remained invalid after one repair"
                ) from repair_exception
            self._add_usage()
            repaired, actions, drops, warnings = _normalize_incident_reflection_metadata(repaired)
            self.last_normalization_actions.extend(actions)
            self.last_metadata_drops.extend(drops)
            self.last_metadata_warnings.extend(warnings)
            try:
                return ReflectionFeedback.model_validate(repaired)
            except ValidationError as second_error:
                self.last_schema_error += "\nRepair validation error: " + json.dumps(
                    compact_validation_error(second_error), ensure_ascii=False,
                )
                self.last_failure_type = "reflection_contract_exhausted"
                raise ReflectionContractExhausted(
                    "incident reflection schema validation failed after one repair: "
                    + json.dumps(compact_validation_error(second_error), ensure_ascii=False)
                ) from second_error
        self._add_usage()
        raw, actions, drops, warnings = _normalize_incident_reflection_metadata(raw)
        self.last_normalization_actions.extend(actions)
        self.last_metadata_drops.extend(drops)
        self.last_metadata_warnings.extend(warnings)
        try:
            return ReflectionFeedback.model_validate(raw)
        except ValidationError as first_error:
            self.last_schema_error = "validation error: " + json.dumps(
                compact_validation_error(first_error), ensure_ascii=False,
            )
            repair_user = (
                "INVALID_INCIDENT_REFLECTION:\n"
                + json.dumps(raw, ensure_ascii=False, default=str)
                + "\n\nVALIDATION_ERRORS:\n"
                + json.dumps(compact_validation_error(first_error), ensure_ascii=False)
                + "\n\n"
                + contract
            )
            self.last_schema_repaired = True
            self.last_prompt_breakdown["repair_attempted"] = True
            self.last_call_count += 1
            repaired = complete_feedback(
                "Repair only the JSON shape of Incident Reflection feedback; preserve all semantics and Evidence IDs.",
                repair_user,
            )
            self._add_usage()
            repaired, actions, drops, warnings = _normalize_incident_reflection_metadata(repaired)
            self.last_normalization_actions.extend(actions)
            self.last_metadata_drops.extend(drops)
            self.last_metadata_warnings.extend(warnings)
            try:
                return ReflectionFeedback.model_validate(repaired)
            except ValidationError as second_error:
                self.last_schema_error += "\nRepair validation error: " + json.dumps(
                    compact_validation_error(second_error), ensure_ascii=False,
                )
                self.last_failure_type = "reflection_contract_exhausted"
                raise ReflectionContractExhausted(
                    "incident reflection schema validation failed after one repair: "
                    + json.dumps(compact_validation_error(second_error), ensure_ascii=False)
                ) from second_error

    # A descriptive alias makes the role convenient for direct deterministic tests.
    review = reflect

    def _add_usage(self) -> None:
        usage = dict(getattr(self.llm, "last_usage", {}) or {})
        self.last_usage["prompt_tokens"] = int(self.last_usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        self.last_usage["completion_tokens"] = int(self.last_usage.get("completion_tokens", 0) or 0) + int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
