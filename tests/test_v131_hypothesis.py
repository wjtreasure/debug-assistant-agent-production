from pathlib import Path
from debug_assistant.memory.hypothesis import HypothesisManager, normalize_location, normalize_target

def test_location_normalization_repo_relative_and_line_safe(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    assert normalize_location(str(repo/'pvlib'/'tools.py')+':350',repo)=='pvlib/tools.py'
    assert normalize_location(r'pvlib\\tools2.py:350-400',repo)=='pvlib/tools2.py'
    assert normalize_location('pvlib/../pvlib/tools2.py#L350-L400',repo)=='pvlib/tools2.py'
    # digits in a file name are not stripped as line numbers
    assert normalize_location('pvlib/tools2.py',repo)=='pvlib/tools2.py'

def test_target_normalization_is_lightweight_and_deterministic():
    assert normalize_target(' Verify zero-bound behavior! ')=='verify zero bound behavior'

def test_evidence_growth_does_not_reset_diagnosis_stability(tmp_path):
    h=HypothesisManager(tmp_path)
    base={'current_diagnosis':'Zero bound causes division by zero','contradicting_evidence_ids':[],
          'required_missing_evidence':[],'optional_validation':[],'evidence_sufficient':True,'confidence':.9}
    a=h.update({**base,'supporting_evidence_ids':['ev-1']},5)
    b=h.update({**base,'supporting_evidence_ids':['ev-1','ev-2']},7)
    assert a.stable_diagnosis_transitions==0
    assert b.stable_diagnosis_transitions==1
    assert a.evidence_fingerprint != b.evidence_fingerprint

def test_required_gap_reason_rewrite_does_not_change_gap_fingerprint(tmp_path):
    h=HypothesisManager(tmp_path)
    common={'current_diagnosis':'candidate','supporting_evidence_ids':['ev-1'],'contradicting_evidence_ids':[],
            'evidence_sufficient':False,'confidence':.5}
    a=h.update({**common,'required_missing_evidence':[{'target':'Verify zero bound!','location':'pvlib/tools.py:350','reason':'first wording'}]},1)
    b=h.update({**common,'required_missing_evidence':[{'target':'verify zero bound','location':'pvlib\\tools.py:999','reason':'rewritten wording'}]},2)
    assert a.required_gap_fingerprint==b.required_gap_fingerprint

def test_bare_and_absolute_location_canonicalize_to_unique_repo_file(tmp_path):
    repo=tmp_path/'repo'; (repo/'pvlib').mkdir(parents=True)
    target=repo/'pvlib'/'tools.py'; target.write_text('x=1')
    assert normalize_location('tools.py',repo)=='pvlib/tools.py'
    assert normalize_location(str(target)+':350',repo)=='pvlib/tools.py'
    # Windows-like absolute output containing the repository directory also resolves.
    win=f"C:/tmp/{repo.name}/pvlib/tools.py:350-400"
    assert normalize_location(win,repo)=='pvlib/tools.py'


def test_structured_root_cause_identity_ignores_diagnosis_prose_rewrite(tmp_path):
    repo=tmp_path/'repo'; (repo/'pvlib').mkdir(parents=True); (repo/'pvlib'/'tools.py').write_text('x=1')
    h=HypothesisManager(repo)
    common={
        'root_cause_target':'_golden_sect_DataFrame',
        'root_cause_location':'tools.py:350',
        'supporting_evidence_ids':['ev-1'],
        'contradicting_evidence_ids':[],
        'required_missing_evidence':[],
        'optional_validation':[],
        'evidence_sufficient':True,
        'confidence':.9,
    }
    a=h.update({**common,'current_diagnosis':'Equal bounds cause -inf/NaN iterlimit.',
                'root_cause_mechanism':'zero-width bounds make iteration limit invalid'},5)
    b=h.update({**common,'current_diagnosis':'When VH equals VL the iteration limit becomes -inf.',
                'root_cause_location':'pvlib/tools.py',
                'root_cause_mechanism':'division by zero produces invalid iteration limit'},7)
    assert a.root_cause_location=='pvlib/tools.py'
    assert b.root_cause_location=='pvlib/tools.py'
    assert a.diagnosis_fingerprint==b.diagnosis_fingerprint
    assert b.stable_diagnosis_transitions==1
