"""Read-only source profiles from Registry declarations and captured observations."""
from debug_assistant.memory.evidence_projector import EvidenceProjector
import hashlib
import json


def profile_matches_constraints(profile, constraints):
    """AND across categories, OR within values; no content or truth inference."""
    return all(not values or set(values).intersection(profile.get(category, ()))
               for category, values in constraints.items())


def provenance_profile_catalog(profiles):
    """Exact producer groups: one kind/source pair and its declared tools."""
    groups = {}
    for row in profiles:
        if not all(row.get(key) for key in ('context_kinds', 'information_sources', 'source_tools')):
            continue
        key = (tuple(row['context_kinds']), tuple(row['information_sources']))
        groups.setdefault(key, set()).update(row['source_tools'])
    catalog = {}
    for (kinds, sources), tools in sorted(groups.items()):
        profile = dict(context_kinds=list(kinds), information_sources=list(sources), source_tools=sorted(tools))
        identity = json.dumps(profile, sort_keys=True, separators=(',', ':'))
        catalog['profile-' + hashlib.sha256(identity.encode()).hexdigest()[:16]] = profile
    return catalog


def registry_provenance_profiles(registry, names=None):
    profiles = []
    for spec in registry.specs():
        if names is not None and spec.name not in names:
            continue
        tool = registry.get(spec.name)
        if (tool is None or spec.side_effect != 'none' or spec.capability == 'knowledge_read'
                or getattr(tool, 'available', True) is False):
            continue
        for declared in getattr(tool, 'provenance_profiles', lambda: ())():
            if declared['information_source'] in EvidenceProjector._NON_INCIDENT_SOURCES:
                continue
            profiles.append({'source_tools': (spec.name,),
                'context_kinds': (declared['context_kind'],),
                'information_sources': (declared['information_source'],)})
    return tuple(profiles)


def observation_provenance_profiles(observations):
    profiles = []
    for row in observations:
        metadata = row.get('metadata', {})
        source = metadata.get('information_source')
        if row.get('ok') is not True or not row.get('tool') or source in EvidenceProjector._NON_INCIDENT_SOURCES:
            continue
        profiles.append({'source_tools': (row['tool'],),
            'context_kinds': (metadata['context_kind'],) if metadata.get('context_kind') else (),
            'information_sources': (source,) if source else ()})
    return tuple(profiles)


def profile_categories(profiles):
    """Vocabulary is a projection of exact producer tuples, never their product."""
    return {'context_kinds': tuple(sorted({row['context_kind'] for row in profiles})),
            'information_sources': tuple(sorted({row['information_source'] for row in profiles}))}


def operation_can_investigate(registry, operation, arguments, constraints):
    """Registry-declared exact output or declared discovery companion, no diagnosis."""
    tool = registry.get(operation)
    if tool is None:
        return False
    exact = getattr(tool, 'provenance_for_arguments', None)
    profiles = tuple({'source_tools': (operation,),
        'context_kinds': (row['context_kind'],), 'information_sources': (row['information_source'],)}
        for row in exact(arguments)) if exact else registry_provenance_profiles(registry, {operation})
    if any(profile_matches_constraints(profile, constraints) for profile in profiles):
        return True
    producers = {name for profile in registry_provenance_profiles(registry)
                 if profile_matches_constraints(profile, constraints) for name in profile['source_tools']}
    return any(operation in spec.companion_tools for spec in registry.specs() if spec.name in producers)


def investigation_argument_domains(registry, operation, constraints):
    """Owner-declared finite argument domains; navigation is not a source witness."""
    tool = registry.get(operation)
    declaration = getattr(tool, 'provenance_argument_domains', None)
    direct = registry_provenance_profiles(registry, {operation})
    if not any(profile_matches_constraints(row, constraints) for row in direct):
        return {}  # Declared companions use their own executable argument contract.
    return declaration(constraints) if declaration and any(constraints.values()) else {}
