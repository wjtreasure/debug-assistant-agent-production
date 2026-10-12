"""Snapshot-local command schemas and exact compilation into Runtime proposals.

No stores, semantic inference, qualification, lifecycle writes or Tool execution.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def text():
    return {'type': 'string', 'minLength': 1, 'pattern': r'[\s\S]*\S[\s\S]*'}


def enum(values):
    return {'type': 'string', 'enum': list(values)}


def array(items, **limits):
    if 'enum' in items:
        limits = {'maxItems': len(items['enum']), 'uniqueItems': True, **limits}
    return {'type': 'array', 'items': items, **limits}


def validate_reference_selection(items, field, path):
    refs = [item[field] for item in items]
    if len(refs) != len(set(refs)):
        raise DecisionSurfaceError([{'path': path, 'message': 'duplicate_reference_selection',
                                    'keyword': 'canonical_identity'}])


def strict_schema(schema):
    """Close objects; optional generated fields become explicitly nullable.

    Do not copy Pydantic union/discriminator annotations into provider schemas.
    Runtime still executes every Pydantic and semantic validator after compilation.
    """
    if not isinstance(schema, dict):
        return schema
    result = {key: deepcopy(value) for key, value in schema.items()
              if key not in {'title', 'default', 'discriminator'}}
    if 'const' in result:
        result['enum'] = [result.pop('const')]
    if 'oneOf' in result:
        result['anyOf'] = result.pop('oneOf')
    if result.get('type') == 'string' and result.get('minLength', 0) > 0 and 'pattern' not in result:
        result['pattern'] = r'[\s\S]{' + str(result['minLength']) + ',}'
    if result.get('type') == 'object':
        required = result.get('required', [])
        properties = {}
        for key, value in result.get('properties', {}).items():
            value = strict_schema(value)
            properties[key] = value if key in required or any(branch.get('type') == 'null' for branch in value.get('anyOf', ())) else {'anyOf': [value, {'type': 'null'}]}
        result.update(properties=properties, required=list(properties), additionalProperties=False)
    for key in ('$defs',):
        if key in result:
            result[key] = {name: strict_schema(value) for name, value in result[key].items()}
    for key in ('items',):
        if key in result:
            result[key] = strict_schema(result[key])
    for key in ('anyOf', 'allOf'):
        if key in result:
            result[key] = [strict_schema(value) for value in result[key]]
    return result


def intersect_field_shapes(base, predicate):
    """Distribute scalar type/enum profiles into complete generation fields.

    Full canonical predicates remain local. No conjunctive/partial object
    schema is sent to a decoder that can lose sibling required constraints.
    """
    alternatives = []
    for left in base.get('anyOf', [base]):
        for right in predicate.get('anyOf', [predicate]):
            if left.get('type') and right.get('type') and left['type'] != right['type']:
                continue
            merged = {**left, **right}
            if 'enum' in left or 'enum' in right:
                values = right.get('enum', left.get('enum'))
                values = [value for value in values if Draft202012Validator(left).is_valid(value)
                          and Draft202012Validator(right).is_valid(value)]
                if not values:
                    continue
                merged['enum'] = values
            for lower in ('minLength', 'minimum'):
                if lower in left and lower in right:
                    merged[lower] = max(left[lower], right[lower])
            for upper in ('maxLength', 'maximum'):
                if upper in left and upper in right:
                    merged[upper] = min(left[upper], right[upper])
            alternatives.append(merged)
    if not alternatives:
        return False  # An impossible scalar intersection, not an invalid empty union.
    return alternatives[0] if len(alternatives) == 1 else {'anyOf': alternatives}


def without_nulls(value):
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None or key == 'value'}
    if isinstance(value, list):
        return [without_nulls(item) for item in value]
    return value


@dataclass(frozen=True)
class LegalActionSurface:
    """One snapshot-bound compiler for owner-derived legal command choices."""
    name: str
    schema: dict
    projection: dict
    bindings: dict
    active_skill: str = ''

    def model_json_schema(self, **kwargs):
        return deepcopy(self.schema)

    def packing_json_schema(self):
        """Same grammar for dry envelope sizing; validate before real I/O."""
        return self.provider_json_schema(validate=False)

    def provider_json_schema(self, *, validate=True):
        """Typed generation grammar; conjunctive executable policies stay local.

        Provider alternatives are complete schemas, never partial properties
        layered over sibling required fields. compile() still validates the
        full owner-derived schema before any Runtime publication.
        """
        def project(value):
            if isinstance(value, dict):
                result = {key: project(item) for key, item in value.items()
                          if key not in {'allOf', 'uniqueItems'}}
                if 'anyOf' in value:
                    branches = value['anyOf']
                    # Registry cross-field profiles are predicates, not types.
                    if value.get('type') == 'object' and 'properties' in value and any(
                            'properties' in branch and 'type' not in branch for branch in branches):
                        base = {key: item for key, item in result.items() if key != 'anyOf'}
                        profiles = []
                        for predicate in branches:
                            fields = predicate.get('properties', {})
                            properties = {key: intersect_field_shapes(field, project(fields.get(key, {})))
                                          for key, field in base['properties'].items()}
                            if all(properties.get(key) is not False for key in base.get('required', ())):
                                profiles.append({**base, 'properties': properties})
                        result = {'anyOf': profiles} if profiles else False
                    else:
                        branches = [branch for branch in branches
                                    if set(branch.get('properties', {})) != {'state_ref'}]
                        projected = [project(branch) for branch in branches]
                        result['anyOf'] = [branch for branch in projected if branch is not False]
                        if not result['anyOf']:
                            return False
                if result is False:
                    return False
                if result.get('pattern') == r'\S':
                    result['pattern'] = r'[\s\S]*\S[\s\S]*'
                if len(result.get('anyOf', ())) == 1 and len(result) == 1:
                    return result['anyOf'][0]
                return result
            if isinstance(value, list):
                return [project(item) for item in value]
            return value
        result = project(self.schema)
        if result is False:
            raise DecisionSurfaceError([{'path': [], 'message': 'provider_schema_uninhabited'}])
        definitions = result.get('$defs', {})
        used = set()
        def references(node):
            if isinstance(node, dict):
                reference = node.get('$ref', '').removeprefix('#/$defs/')
                if reference and reference not in used:
                    used.add(reference)
                    references(definitions[reference])
                for key, value in node.items():
                    if key != '$defs':
                        references(value)
            elif isinstance(node, list):
                for item in node:
                    references(item)
        references(result)
        if result is False:
            raise DecisionSurfaceError([{'path': [], 'message': 'provider_schema_uninhabited'}])
        if '$defs' in result:
            result['$defs'] = {key: definitions[key] for key in sorted(used)}
        # Validate the complete emitted grammar, including unreachable alternatives.
        if validate:
            Draft202012Validator.check_schema(result)
        return result

    def compile(self, arguments):
        validate_command(self.schema, arguments)
        value = without_nulls(arguments)
        if self.name == 'submit_planning_decision':
            return self._plan(value)
        command = value['decision']
        if command['kind'] == 'CONTEXT_REQUEST':
            selected = command.get('requests', command.get('pending_requests', {}))
            if isinstance(selected, list):
                references = [item['reference'] for item in selected]
                if len(set(references)) != len(references):
                    raise DecisionSurfaceError([{'path': ['decision', 'requests'],
                        'message': 'context_request_duplicate_reference'}])
                selected = {item['reference']: item['selector'] for item in selected}
            if not 1 <= len(selected) <= 8:
                raise DecisionSurfaceError([{'path': ['decision', 'requests'],
                    'message': 'context_request_batch_size_invalid'}])
            from debug_assistant.context.requests import ContextRequestBatch
            from pydantic import ValidationError
            batch = {'requests': [{**self.bindings['context'][reference], 'selector': selector}
                                  for reference, selector in selected.items()]}
            try:
                ContextRequestBatch.model_validate(batch)
            except ValidationError as exc:
                raise DecisionSurfaceError([{'path': ['decision', 'requests'],
                    'message': 'context_selector_invalid', 'keyword': 'command_admission'}]) from exc
            return {'kind': 'CONTEXT_REQUEST', 'batch': batch}
        validate_reference_selection(command.get('acknowledgements', []), 'observation_ref',
                                     ['decision', 'acknowledgements'])
        validate_reference_selection(command.get('contradictions', []), 'evidence_id',
                                     ['decision', 'contradictions'])
        hypothesis = command['hypothesis'].copy()
        kind = hypothesis.pop('kind')
        dispositions = [{'obligation_id': ref, **item} for ref, item in hypothesis.pop('dispositions', {}).items() if item is not None]
        hypothesis['transition'] = {'kind': kind, 'obligation_dispositions': dispositions}
        delta = {'skill_id': self.active_skill or command['skill_id'],
            'evidence_need': command['evidence_need'], 'hypothesis': hypothesis,
            'new_obligation_proposals': [dict(item, claim=item['verification_question'],
                evidence_requirement='; '.join(fact['description'] for fact in item['completion_contract']['required_facts']))
                for item in command.get('questions', [])],
            'operational_acknowledgements': command.get('acknowledgements', []),
            'contradictions': command.get('contradictions', []), 'evidence_proposals': []}
        # Creating a Question never re-declares a committed canonical identity.
        # Contract conflict admission remains the independent Runtime barrier.
        from debug_assistant.incidents.contracts import build_verification_key
        existing = {row.get('verification_key') or build_verification_key(
            target_component=row.get('target_component', ''),
            verification_question=row.get('verification_question', ''),
            evidence_type=row.get('evidence_type', ''))
            for row in self.projection.get('obligations', ())}
        proposed = set()
        for question in delta['new_obligation_proposals']:
            key = build_verification_key(target_component=question.get('target_component', ''),
                verification_question=question['verification_question'], evidence_type=question.get('evidence_type', ''))
            if key in existing or key in proposed:
                detail = {'path': ['decision', 'questions'],
                    'message': 'existing_question_requires_canonical_reference' if key in existing
                               else 'duplicate_question_proposal', 'keyword': 'canonical_identity'}
                if key in existing:
                    matches = [row for row in self.projection.get('obligations', ())
                        if (row.get('verification_key') or build_verification_key(
                            target_component=row.get('target_component', ''),
                            verification_question=row.get('verification_question', ''),
                            evidence_type=row.get('evidence_type', ''))) == key]
                    ids = {row['id'] for row in matches}
                    detail.update(canonical_questions=[{'id': row['id'], 'status': row['status']} for row in matches],
                        canonical_requirements=[{'requirement_ref': ref, **{name: row[name] for name in
                            ('obligation_id', 'slot', 'description', 'role')}}
                            for ref, row in self.bindings['requirements'].items() if row['obligation_id'] in ids])
                raise DecisionSurfaceError([detail])
            proposed.add(key)
        for fact in command.get('evidence', []):
            fact = fact.copy()
            source = self.bindings['evidence'][fact.pop('source_ref')]
            coverage = fact.pop('coverage')
            if coverage['action'] == 'COVER_EXISTING_REQUIREMENT':
                targets = defaultdict(list)
                for reference in coverage['requirement_refs']:
                    targets[self.bindings['requirements'][reference]['obligation_id']].append(reference)
                action = 'COVER_EXISTING_REQUIREMENT'
            else:
                targets = {coverage['obligation_ref']: []}
                action = 'NO_RELEVANT_COVERAGE'
            for target, references in targets.items():
                delta['evidence_proposals'].append(dict(fact, **source, obligation_id=target,
                    requirement_refs=references, coverage_action=action,
                    hypothesis_target='CURRENT' if kind == 'KEEP_CURRENT' else 'PROPOSED'))
        return {'kind': 'SEMANTIC_DELTA', 'delta': delta}

    def _plan(self, value):
        tasks = []
        item = value.get('action')
        while item is not None:
            reference = item['action_ref']
            binding = self.bindings['actions'][reference]
            args = item['arguments'].copy()
            for name, choices in self.bindings.get('argument_choices', {}).get(reference, {}).items():
                args.update(choices[args.pop(name)])
            # State refs are already typed and selected exactly; no fuzzy target.
            for name, value in list(args.items()):
                if isinstance(value, dict) and 'state_ref' in value:
                    args[name] = deepcopy(self.bindings['state_values'][value['state_ref']])
            primary = binding['semantic_intents'][0]
            tasks.append(dict(choice_ref=binding['choice_ref'], **primary,
                semantic_intents=binding['semantic_intents'], args=args,
                task_id=f'task-{len(tasks)+1}', dependencies=[]))
            item = item.get('next')
        return {'plan': {'tasks': tasks}}


class DecisionSurfaceError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('semantic_command_schema_invalid')


def validate_command(schema, arguments):
    error = best_match(Draft202012Validator(schema).iter_errors(arguments))
    if error is None:
        return
    while error.context:
        error = best_match(error.context)
    raise DecisionSurfaceError([{'path': list(error.absolute_path),
                                'message': error.message[:300], 'keyword': error.validator}])


def interpretation_surface(view, projection, *, active_skill='', feasible_skills=(), provenance_categories=None,
                           context_resolution_remaining=None):
    from debug_assistant.agent.planner import acknowledgement_targets, pending_context_targets, PlannerHypothesisProposal, PlannerObligationProposal
    from debug_assistant.incidents.contracts import Contradiction
    from debug_assistant.incidents.semantic_transition import disposition_targets
    from debug_assistant.memory.evidence_projector import EvidenceProjector

    definitions = {}
    def model_schema(model, name):
        schema = strict_schema(model.model_json_schema())
        definitions.update(schema.pop('$defs', {}))
        definitions[name] = schema
        return {'$ref': '#/$defs/' + name}

    hypothesis = strict_schema(PlannerHypothesisProposal.model_json_schema())
    definitions.update(hypothesis.pop('$defs', {}))
    meaning = hypothesis['properties']
    for key in ('transition', 'supporting_evidence_ids', 'contradicting_evidence_ids'):
        meaning.pop(key)
    # These are citations selected by the model, not inferred support.
    evidence_ids = [row['evidence_ref'] for row in projection.get('existing_evidence', ())]
    from debug_assistant.incidents.semantic_transition import contradiction_reference_ids
    contradictory_ids = contradiction_reference_ids(projection)
    for key in ('supporting_evidence_ids', 'contradicting_evidence_ids'):
        citations = contradictory_ids if key == 'contradicting_evidence_ids' else evidence_ids
        if citations:
            meaning[key] = array(enum(citations))
    source_ids = [row['evidence_ref'] for row in projection.get('existing_evidence', ())
                  if row.get('canonical_source') is True]
    if source_ids and 'SourceClaim' in definitions:
        definitions['SourceClaim']['properties']['evidence_ids'] = array(enum(source_ids), minItems=1)
    else:
        meaning.pop('source_claims', None)
    meaning['claim'] = text()
    dispositions = disposition_targets(projection.get('obligations', ()))
    required_dispositions = {row['id'] for row in projection.get('obligations', ())
                             if row.get('critical', True) and row['id'] in dispositions}
    disposition = obj({reference: {'anyOf': [obj({'action': enum(('RETAIN', 'SUPERSEDE')),
        'reason': text()}), {'type': 'null'}]} for reference in dispositions}) if dispositions else None
    if disposition:
        for reference in required_dispositions:
            disposition['properties'][reference] = disposition['properties'][reference]['anyOf'][0]
    current = projection.get('active_hypothesis')
    variants = []
    if current:
        keep = {'kind': enum(('KEEP_CURRENT',))}
        if disposition:
            keep['dispositions'] = disposition
        variants.append(obj(keep))
    for kind in ('REFINE', 'REPLACE') if current else ('INITIAL',):
        fields = {'kind': enum((kind,)), **deepcopy(meaning)}
        if disposition:
            fields['dispositions'] = deepcopy(disposition)
        variants.append(obj(fields))

    question = strict_schema(PlannerObligationProposal.model_json_schema())
    definitions.update(question.pop('$defs', {}))
    question['properties'].pop('claim')
    question['properties'].pop('evidence_requirement')
    question['properties']['verification_question'] = text()
    question['properties']['completion_contract'] = {'$ref': '#/$defs/CompletionContract'}
    question['required'] = list(question['properties'])
    definitions['Question'] = question
    if provenance_categories is not None:
        for category, values in provenance_categories.items():
            field = definitions['EvidenceConstraint']['properties'][category]
            # Put array constraints on the actual nullable array branch. Some
            # strict decoders compile anyOf without its sibling keywords.
            for branch in field.get('anyOf', (field,)):
                if branch.get('type') == 'array':
                    branch['items'] = enum(sorted(values)) if values else {'type': 'string'}
                    # Constraint categories are finite sets, not unbounded
                    # sequences. Keep both limits on the actual array branch;
                    # the Provider projection omits only unsupported uniqueness.
                    branch.update(maxItems=len(values), uniqueItems=True)
    semantic = {'kind': enum(('SEMANTIC_DELTA',)), 'evidence_need': text(),
        'hypothesis': {'anyOf': variants}, 'questions': array({'$ref': '#/$defs/Question'})}
    if not active_skill:
        semantic['skill_id'] = enum(feasible_skills)
    ack_targets = acknowledgement_targets(view)
    if ack_targets:
        semantic['acknowledgements'] = array(obj({'observation_ref': enum(ack_targets), 'reason': text()}),
                                             maxItems=len(ack_targets))
    requirement_rows = projection.get('canonical_requirements', ())
    requirements = {row['requirement_ref']: row for row in requirement_rows}
    sources = {row['observation_id']: {'observation_refs': [row['observation_id']]}
        for row in view.reference_metadata.get('visible_observations', ())
        if row['observation_id'] in view.visible_observation_ids and row.get('ok') is True
        and row.get('metadata', {}).get('information_source') not in EvidenceProjector._NON_INCIDENT_SOURCES
        and ('admissible_observation_refs' not in projection
             or row['observation_id'] in projection['admissible_observation_refs'])
        and row.get('metadata', {}).get('result_kind') != 'OPERATIONAL_ONLY'
        and row.get('metadata', {}).get('semantic_negative') is not True}
    sources.update({ref: {'evidence_ref': ref} for ref in view.visible_evidence_refs})
    if current and requirements and sources:
        coverage = {'anyOf': [None,
            obj({'action': enum(('NO_RELEVANT_COVERAGE',)),
                 'obligation_ref': enum(dict.fromkeys(row['obligation_id'] for row in requirement_rows))})]}
        fields = {'coverage': coverage,
            'semantic_claim': text(), 'reason': text(),
            'hypothesis_relation': enum(('SUPPORTS', 'CONTRADICTS', 'NEUTRAL')),
            'relation': enum(('SUPPORTS', 'CONTRADICTS', 'NEUTRAL'))}
        raw = [reference for reference, source in sources.items() if 'observation_refs' in source]
        reused = [reference for reference, source in sources.items() if 'evidence_ref' in source]
        variants = []
        if raw:
            for reference in raw:
                allowed_slots = projection.get('source_requirement_slots', {}).get(reference, [])
                allowed = [ref for ref, row in requirements.items() if [row['obligation_id'], row['slot']] in allowed_slots]
                source_fields = deepcopy(fields)
                selected_sets = [obj({'action': enum(('COVER_EXISTING_REQUIREMENT',)),
                    'requirement_refs': array(enum(allowed), minItems=1, maxItems=len(allowed), uniqueItems=True)})] if allowed else []
                source_fields['coverage'] = {'anyOf': [*selected_sets, coverage['anyOf'][1]]}
                variants.append(obj(dict(source_fields, source_ref=enum((reference,)), source_spans=array({'type': 'string'}))))
        if reused:
            for reference in reused:
                allowed_slots = projection.get('source_requirement_slots', {}).get(reference, [])
                allowed = [ref for ref, row in requirements.items() if [row['obligation_id'], row['slot']] in allowed_slots]
                source_fields = deepcopy(fields)
                selected_sets = [obj({'action': enum(('COVER_EXISTING_REQUIREMENT',)),
                    'requirement_refs': array(enum(allowed), minItems=1, maxItems=len(allowed), uniqueItems=True)})] if allowed else []
                source_fields['coverage'] = {'anyOf': [*selected_sets, coverage['anyOf'][1]]}
                variants.append(obj(dict(source_fields, source_ref=enum((reference,)))))
        semantic['evidence'] = array(variants[0] if len(variants) == 1 else {'anyOf': variants})
    if contradictory_ids:
        contradiction = model_schema(Contradiction, 'CommandContradiction')
        definitions['CommandContradiction']['properties']['evidence_id'] = enum(contradictory_ids)
        semantic['contradictions'] = array(contradiction, maxItems=len(contradictory_ids))

    from debug_assistant.context.manager import ContextManager
    pending = pending_context_targets(view, projection)
    recoveries = ContextManager.legal_recovery_choices(view)
    context = {reference: {'observation_id': reference} for reference in recoveries
               if not pending or reference in pending}
    choices = [] if pending else [obj(semantic)]
    if pending and (set(pending) - set(context) or len(pending) > 8):
        context = {}
    if context_resolution_remaining == 0:
        context = {}
    if context:
        def selectors_schema(selectors):
            variants = []
            for selector in selectors:
                if selector['kind'] == 'FULL_STORED_BODY':
                    variants.append(obj({'kind': enum(('FULL_STORED_BODY',))}))
                else:
                    bounds = {'type': 'integer', 'minimum': selector['start_line'],
                              'maximum': selector['end_line']}
                    variants.append(obj({'kind': enum(('LINE_RANGE',)),
                        'start_line': deepcopy(bounds), 'end_line': deepcopy(bounds)}))
            return {'anyOf': variants}
        requests = [obj({'reference': enum((reference,)), 'selector': selectors_schema(recoveries[reference])})
                    for reference in context]
        if pending:
            # All pending targets, once each: no subset can pass the schema.
            fields = {reference: {'anyOf': [branch['properties']['selector']]} for reference, branch in zip(context, requests)}
            choices.append(obj({'kind': enum(('CONTEXT_REQUEST',)), 'pending_requests': obj(fields)}))
        else:
            # A positive selection: an empty/null map is not a legal command.
            choices.append(obj({'kind': enum(('CONTEXT_REQUEST',)),
                'requests': array({'anyOf': requests}, minItems=1, maxItems=min(8, len(requests)))}))
    schema = obj({'decision': {'anyOf': choices} if choices else {'type': 'null'}})
    schema['$defs'] = definitions
    # Remove unreachable model definitions (notably the old hypothesis transition).
    used = set()
    def visit(node):
        if isinstance(node, dict):
            reference = node.get('$ref', '').removeprefix('#/$defs/')
            if reference and reference not in used:
                used.add(reference)
                visit(definitions[reference])
            for key, value in node.items():
                if key != '$defs':
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    schema['$defs'] = {key: definitions[key] for key in sorted(used)}
    return LegalActionSurface('submit_interpretation_decision', schema, projection,
        {'context': context, 'evidence': sources, 'requirements': requirements, 'unavailable': not choices}, active_skill)


def executable_argument_schema(model):
    """No optional Tool text has two absence encodings: use null, or nonblank text."""
    schema = model.model_json_schema()
    # Strict commands supply every Tool argument. A Python default is not a
    # nullable type: only the Registry's actual nullable union may admit null.
    schema['required'] = list(schema.get('properties', {}))
    schema = strict_schema(schema)
    def tighten(node):
        if not isinstance(node, dict):
            return
        if any(branch.get('type') == 'null' for branch in node.get('anyOf', ())):
            for branch in node['anyOf']:
                if branch.get('type') == 'string':
                    branch['minLength'] = max(1, branch.get('minLength', 0))
                    branch.setdefault('pattern', r'[\s\S]*\S[\s\S]*')
        for value in node.values():
            if isinstance(value, dict):
                tighten(value)
            elif isinstance(value, list):
                for item in value:
                    tighten(item)
    tighten(schema)
    return schema


def project_argument_choices(arguments, selections):
    """Carry Registry cross-field predicates through exact flag projection."""
    def visit(node):
        if not isinstance(node, dict):
            return
        properties = node.get('properties', {})
        for name, choices in selections.items():
            flags = next(iter(choices.values()))
            present = set(flags).intersection(properties)
            if present:
                legal = [choice for choice, values in choices.items() if all(
                    Draft202012Validator(properties[flag]).is_valid(values[flag]) for flag in present)]
                for flag in present:
                    properties.pop(flag)
                properties[name] = enum(legal)
                if 'required' in node:
                    node['required'] = [key for key in node['required'] if key not in flags]
                    node['required'].append(name)
        for key, value in node.items():
            if key == 'properties':
                for field in value.values():
                    visit(field)
            elif isinstance(value, dict):
                visit(value)
            elif isinstance(value, list):
                for field in value:
                    visit(field)
    visit(arguments)


def state_reference_bindings(projection):
    """Readonly scalar references shared by schema compilation and request rendering."""
    state_paths, state_values = {}, {}
    def walk(value, path):
        if value is not None and not isinstance(value, (dict, list)):
            ref = f'state-{len(state_paths)+1}'
            state_paths[ref], state_values[ref] = list(path), value
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, key))
        elif isinstance(value, list):
            for key, item in enumerate(value):
                walk(item, (*path, key))
    for key in ('incident', 'entities', 'visible_observations', 'capabilities'):
        if key in projection:
            walk(projection[key], (key,))
    return state_paths, state_values


def planning_surface(turn, tools, *, intent_catalog=False):
    """One executable family per operation, with all eligible semantic intents.

    Ordered commands make duplicate physical acquisitions impossible within a
    batch: an operation occurs once, with up to four distinct operations. Known
    AcquisitionKey exclusions belong to committed-state Runtime admission.
    """
    projection = turn.semantic_projection
    choices = {row['choice_ref']: row for row in turn.action_choices}
    specs = {spec.name: spec for spec in tools.specs()}
    definitions, actions, argument_choices = {}, {}, {}
    grouped = {}
    targets = {row['id']: row.get('required_fact_slots', ()) for row in projection.get('obligations', ()) if row.get('status') == 'OPEN'}
    gaps = projection.get('gap_choice_surface', ())
    if intent_catalog:
        grouped = {row['operation']: {'choice_ref': row['choice_ref'], 'semantic_intents': []}
                   for row in choices.values() if row['skill_id'] == turn.active_skill_id and row['operation'] in specs}
    for gap in gaps:
        if gap['target_fact_slot'] not in targets.get(gap['obligation_id'], ()):
            continue
        for ref in gap['available_choice_refs']:
            choice = choices.get(ref)
            if choice is None or choice['operation'] not in specs or choice['skill_id'] != turn.active_skill_id:
                continue
            operation = choice['operation']
            entry = grouped.setdefault(operation, {'choice_ref': ref, 'semantic_intents': []})
            intent = {'target_obligation_id': gap['obligation_id'], 'target_fact_slot': gap['target_fact_slot']}
            if intent not in entry['semantic_intents']:
                entry['semantic_intents'].append(intent)
    state_paths, state_values = state_reference_bindings(projection)
    for operation, binding in grouped.items():
        spec = specs[operation]
        if spec.side_effect != 'none' or tools.get(operation) is None:
            continue
        selections = getattr(spec.args_model, 'command_choices', {})
        arguments = executable_argument_schema(spec.args_model)
        definitions.update(arguments.pop('$defs', {}))
        project_argument_choices(arguments, selections)
        arguments['required'] = list(arguments['properties'])
        def bind_state(node):
            if not isinstance(node, dict):
                return
            for name in ('anyOf', 'allOf'):
                for branch in node.get(name, ()):
                    bind_state(branch)
            for key, field in list(node.get('properties', {}).items()):
                if key in selections or key not in spec.args_model.model_fields:
                    continue
                typed = Draft202012Validator({**field, '$defs': definitions})
                legal = [ref for ref, value in state_values.items() if typed.is_valid(value)]
                if legal:
                    node['properties'][key] = {'anyOf': [field, obj({'state_ref': enum(legal)})]}
        bind_state(arguments)

        reference = f'action-{len(actions)+1}'
        actions[reference] = binding
        argument_choices[reference] = selections
        # Executable args only. Unresolved future Observation bindings are not
        # current-state executable choices; acquire then replan against retained state.
        definitions[operation] = arguments
    if intent_catalog:
        return LegalActionSurface('next_action_catalog', {**obj({}), '$defs': definitions}, projection,
            {'actions': actions, 'state': state_paths, 'state_values': state_values,
             'argument_choices': argument_choices}, turn.active_skill_id)
    refs = list(actions)
    def chain(after, remaining):
        branches = []
        for index in range(after, len(refs)):
            reference = refs[index]
            operation = choices[actions[reference]['choice_ref']]['operation']
            next_schema = {'type': 'null'}
            if remaining > 1 and index+1 < len(refs):
                name = f'after_{index}_{remaining-1}'
                if name not in definitions:
                    definitions[name] = chain(index+1, remaining-1)
                next_schema = {'$ref': '#/$defs/' + name}
            branches.append(obj({'action_ref': enum((reference,)),
                'arguments': {'$ref': '#/$defs/' + operation}, 'next': next_schema}))
        return {'anyOf': [{'type': 'null'}, *branches]}
    schema = obj({'action': chain(0, 4)})
    schema['$defs'] = definitions
    return LegalActionSurface('submit_planning_decision', schema, projection,
        {'actions': actions, 'state': state_paths, 'state_values': state_values,
         'argument_choices': argument_choices}, turn.active_skill_id)
