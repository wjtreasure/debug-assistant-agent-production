"""One reasoning proposal; exact post-commit intent binding, without semantic writes."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json

from .decision_surface import (
    LegalActionSurface, DecisionSurfaceError, obj, enum, array, planning_surface, validate_command,
)


@dataclass(frozen=True)
class NextActionIntent:
    operation: str
    arguments: dict
    obligation_id: str = ''
    slot: str = ''
    question: dict | None = None


@dataclass(frozen=True)
class DecisionTurnSurface(LegalActionSurface):
    semantic_surface: LegalActionSurface | None = None

    def compile(self, arguments):
        validate_command(self.schema, arguments)
        value = deepcopy(arguments)
        command = value['decision']
        if command['kind'] == 'CONTEXT_REQUEST':
            return self.semantic_surface.compile(value)
        mode = command.pop('question_mode')
        raw = command.pop('next_action_intent')
        question = command.pop('question')
        if question is not None:
            from .question_protocol import compile_question
            question = compile_question(question, self.bindings['source_profiles'])
        command['questions'] = [question] if question is not None else []
        intent = None
        if raw is not None:
            skill = self.active_skill or command['skill_id']
            catalog = self.bindings['intent_catalogs'][skill]
            operations = {row['operation'] for row in self.bindings['choices'].values()
                          if row['skill_id'] == skill}
            if raw['operation'] not in operations:
                raise DecisionSurfaceError([{'path': ['decision', 'next_action_intent'],
                    'message': 'operation_unavailable_for_selected_skill', 'keyword': 'command_admission'}])
            for name, argument in list(raw['arguments'].items()):
                if isinstance(argument, dict) and 'state_ref' in argument:
                    raw['arguments'][name] = deepcopy(catalog.bindings['state_values'][argument['state_ref']])
            if mode == 'EXISTING':
                row = self.bindings['requirements'][raw['target']['requirement_ref']]
                intent = NextActionIntent(raw['operation'], raw['arguments'], row['obligation_id'], row['slot'])
            else:
                if question is None:
                    raise DecisionSurfaceError([{'path': ['decision', 'question'],
                        'message': 'proposed_question_required', 'keyword': 'command_admission'}])
                intent = NextActionIntent(raw['operation'], raw['arguments'],
                    slot=question['completion_contract']['required_facts'][0]['slot'], question=question)
        try:
            canonical = self.semantic_surface.compile(value)
        except DecisionSurfaceError as exc:
            details = deepcopy(exc.details)
            for detail in details:
                if detail.get('path') == ['decision', 'questions']:
                    detail['path'] = ['decision', 'question']
                if detail.get('message') == 'existing_question_requires_canonical_reference':
                    detail.update(required_question_mode='EXISTING', question_must_be_null=True,
                        intent_target_field='next_action_intent.target.requirement_ref')
            raise DecisionSurfaceError(details) from exc
        from debug_assistant.incidents.contracts import CompletionContract
        from debug_assistant.incidents.obligation_admission import unrealizable_provenance_requirements
        for proposal in canonical['delta']['new_obligation_proposals']:
            rejected = unrealizable_provenance_requirements(
                CompletionContract.model_validate(proposal['completion_contract']),
                self.bindings['provenance_profiles'])
            if rejected:
                raise DecisionSurfaceError([{'path': ['decision', 'question', 'completion_contract'],
                    'message': 'completion_contract_provenance_unrealizable',
                    'keyword': 'command_admission', 'fact_slots': list(rejected),
                    'allowed_source_profiles': self.bindings['provenance_profiles']}])
        # This transient proposal is never published into RCAState/EvidenceMemory.
        canonical['next_action_intent'] = asdict(intent) if intent else None
        return canonical


def intent_projection(turn, semantic_projection):
    """Use the role-visible metadata for every exposed argument reference."""
    return {**turn.semantic_projection,
            "visible_observations": semantic_projection.get("visible_observations", [])}


def decision_turn_surface(semantic, turn, tools):
    """Share one command schema, selecting Skill and its executable surface together."""
    if not any(branch['properties']['kind']['enum'] == ['SEMANTIC_DELTA']
               for branch in semantic.schema['properties']['decision'].get('anyOf', ())):
        return semantic
    schema = deepcopy(semantic.schema)
    definitions = schema['$defs']
    from debug_assistant.tools.provenance import (
        registry_provenance_profiles, observation_provenance_profiles, provenance_profile_catalog,
    )
    operations_available = {row['operation'] for row in turn.action_choices}
    profiles = registry_provenance_profiles(tools, operations_available) + observation_provenance_profiles(
        semantic.projection.get('visible_observations', ()))
    source_profiles = provenance_profile_catalog(profiles)
    from .question_protocol import install_question_contract
    definitions['NextQuestion'] = install_question_contract(definitions, definitions['Question'], source_profiles)
    # One semantic object, one proposal owner, one typed command per operation.
    # Cross-field command admission stays local instead of multiplying whole
    # decisions for Skill x operation x target x hypothesis dispositions.
    catalogs, choices = {}, {row['choice_ref']: row for row in turn.action_choices}
    operations = {}
    for skill in (turn.active_skill_id,) if turn.active_skill_id else turn.feasible_skill_ids:
        catalog = planning_surface(replace(turn, active_skill_id=skill,
            semantic_projection_json=json.dumps(intent_projection(turn, semantic.projection))),
            tools, intent_catalog=True)
        catalogs[skill] = catalog
        definitions.update(catalog.schema['$defs'])
        for binding in catalog.bindings['actions'].values():
            operation = choices[binding['choice_ref']]['operation']
            references = [reference for reference, row in semantic.bindings['requirements'].items()
                          if {'target_obligation_id': row['obligation_id'], 'target_fact_slot': row['slot']}
                          in binding['semantic_intents']]
            operations.setdefault(operation, set()).update(references)
    existing_intents, new_intents = [], []
    for operation, references in operations.items():
        fields = {'operation': enum((operation,)), 'arguments': {'$ref': '#/$defs/' + operation}}
        new_name = 'new_intent_' + operation
        definitions[new_name] = obj(fields)
        new_intents.append({'$ref': '#/$defs/' + new_name})
        if references:
            from debug_assistant.tools.provenance import investigation_argument_domains
            groups = {}
            for reference in sorted(references):
                requirement = semantic.bindings['requirements'][reference]
                fact = next(fact for obligation in semantic.projection['obligations']
                    if obligation['id'] == requirement['obligation_id']
                    for fact in obligation['completion_contract']['required_facts']
                    if fact['slot'] == requirement['slot'])
                arguments = deepcopy(definitions[operation])
                domains = investigation_argument_domains(tools, operation, fact.get('evidence_constraints', {}))
                if any(not values for values in domains.values()):
                    continue
                for parameter, values in domains.items():
                    arguments['properties'][parameter] = enum(values)
                identity = json.dumps(arguments, sort_keys=True, separators=(',', ':'))
                groups.setdefault(identity, (arguments, []))[1].append(reference)
            alternatives = []
            for index, (arguments, targets) in enumerate(groups.values()):
                name = operation if arguments == definitions[operation] else f'existing_arguments_{operation}_{index}'
                if name != operation:
                    definitions[name] = arguments
                alternatives.append(obj({'operation': enum((operation,)),
                    'arguments': {'$ref': '#/$defs/' + name},
                    'target': obj({'requirement_ref': enum(targets)})}))
            if alternatives:
                name = 'existing_intent_' + operation
                definitions[name] = {'anyOf': alternatives}
                existing_intents.append({'$ref': '#/$defs/' + name})
    branches = []
    for branch in schema['properties']['decision'].get('anyOf', ()):
        if branch['properties']['kind']['enum'] != ['SEMANTIC_DELTA']:
            branches.append(branch)
            continue
        fields = {key: value for key, value in branch['properties'].items() if key != 'questions'}
        # Existing/new Question ownership changes only those fields. Shared
        # semantic decisions have one declaration in the complete wire grammar.
        for key in ('hypothesis', 'evidence', 'acknowledgements', 'contradictions'):
            if key in fields:
                name = 'Semantic_' + key
                definitions[name] = fields[key]
                fields[key] = {'$ref': '#/$defs/' + name}
        branches.extend([
            obj({**fields, 'question_mode': enum(('EXISTING',)), 'question': {'type': 'null'},
                'next_action_intent': {'anyOf': [{'type': 'null'}, *existing_intents]}}),
            obj({**fields, 'question_mode': enum(('NEW',)), 'question': {'$ref': '#/$defs/NextQuestion'},
                'next_action_intent': {'anyOf': [{'type': 'null'}, *new_intents]}}),
        ])
    schema['properties']['decision'] = {'anyOf': branches}
    return DecisionTurnSurface(semantic.name, schema, semantic.projection,
        {**semantic.bindings, 'intent_catalogs': catalogs, 'choices': choices,
         'provenance_profiles': profiles, 'source_profiles': source_profiles, 'registry': tools}, semantic.active_skill, semantic)


def compile_next_action(intent, turn, tools):
    """Resolve only model-selected targets, against actual committed state/owners."""
    from debug_assistant.incidents.contracts import CompletionContract, build_verification_key
    from debug_assistant.incidents.verification_plan import VerificationPlanError, VerificationPlanProposal
    projection = turn.semantic_projection
    if intent is None:
        return VerificationPlanProposal(tasks=())
    target_id = intent.obligation_id
    slot = intent.slot
    if intent.question is not None:
        question = intent.question
        key = build_verification_key(target_component=question.get('target_component', ''),
            verification_question=question['verification_question'], evidence_type=question.get('evidence_type', ''))
        contract = CompletionContract.model_validate(question['completion_contract'])
        matches = [row for row in projection.get('obligations', ())
                   if row.get('verification_key') == key and row.get('completion_contract')
                   and CompletionContract.model_validate(row['completion_contract']).equivalent(contract)]
        if len(matches) != 1:
            raise VerificationPlanError('next_action_target_ambiguous')
        target_id = matches[0]['id']
    from debug_assistant.tools.provenance import operation_can_investigate
    target = next((row for row in projection.get('obligations', ()) if row['id'] == target_id), None)
    if target is None or target.get('status') != 'OPEN' or not target.get('completion_contract'):
        raise VerificationPlanError('next_action_unavailable')
    fact = next((row for row in target['completion_contract']['required_facts'] if row['slot'] == slot), None) if target else None
    if fact is None:
        raise VerificationPlanError('next_action_unavailable')
    if not operation_can_investigate(tools, intent.operation, intent.arguments,
            fact.get('evidence_constraints', {})):
        raise VerificationPlanError('next_action_source_profile_mismatch')
    surface = planning_surface(turn, tools)
    choices = {row['choice_ref']: row for row in turn.action_choices}
    matches = [(ref, binding) for ref, binding in surface.bindings['actions'].items()
               if choices[binding['choice_ref']]['operation'] == intent.operation
               and {'target_obligation_id': target_id, 'target_fact_slot': slot} in binding['semantic_intents']]
    if len(matches) != 1:
        raise VerificationPlanError('next_action_unavailable')
    reference, binding = matches[0]
    try:
        proposal = surface.compile({'action': {'action_ref': reference, 'arguments': intent.arguments, 'next': None}})
    except DecisionSurfaceError as exc:
        raise VerificationPlanError('next_action_arguments_unavailable', detail=str(exc.details)) from exc
    task = proposal['plan']['tasks'][0]
    task.update(target_obligation_id=target_id, target_fact_slot=slot)
    return VerificationPlanProposal.model_validate(proposal['plan'])
