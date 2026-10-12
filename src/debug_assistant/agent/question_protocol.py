"""One wire Question/CompletionContract compiler for normal and recovery proposals."""
from copy import deepcopy

from .decision_surface import obj, enum, array


def install_question_contract(definitions, question, profiles):
    fact = deepcopy(definitions['FactRequirement'])
    fact['properties'].pop('evidence_constraints')
    fact['properties']['source_profile_ref'] = (
        {'anyOf': [enum(profiles), {'type': 'null'}]} if profiles else {'type': 'null'})
    fact['required'] = list(fact['properties'])
    definitions['DecisionFact'] = fact
    question = deepcopy(question)
    for field in ('claim', 'evidence_requirement'):
        question['properties'].pop(field, None)
    question['properties']['completion_contract'] = obj({
        'target_fact': {'$ref': '#/$defs/DecisionFact'},
        'additional_facts': array({'$ref': '#/$defs/DecisionFact'}),
    })
    question['required'] = list(question['properties'])
    return question


def compile_question(question, profiles):
    proposal = deepcopy(question)
    # Historical offline fixtures may omit the additive role. Real strict
    # requests include it; an omitted legacy value remains safely necessary.
    proposal.setdefault('verification_role', 'NECESSARY')
    contract = proposal['completion_contract']
    contract['required_facts'] = [contract.pop('target_fact'), *contract.pop('additional_facts')]
    for fact in contract['required_facts']:
        reference = fact.pop('source_profile_ref')
        fact['evidence_constraints'] = (deepcopy(profiles[reference]) if reference else {
            'context_kinds': [], 'information_sources': [], 'source_tools': []})
    return proposal
