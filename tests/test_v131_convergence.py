from debug_assistant.harness.convergence import ConvergenceController, ConvergenceMode, ProgressKind

def hyp(step, *, desc='root', status='confirmed', support=None, contra=None, gaps=None, stable=0, dfp='d', efp='e', gfp='g'):
    return {'description':desc,'status':status,'supporting_evidence_ids':support or ['ev-1'],
            'contradicting_evidence_ids':contra or [],'required_missing_evidence':gaps or [],
            'stable_diagnosis_transitions':stable,'diagnosis_fingerprint':dfp,'evidence_fingerprint':efp,
            'required_gap_fingerprint':gfp,'updated_step':step,'evidence_sufficient': True}

def test_convergence_requires_stable_diagnosis_and_no_required_gap():
    c=ConvergenceController(no_progress_limit=2)
    a=c.assess_reflection(hyp(5,stable=0),usage_totals={'tokens':10})
    assert a.kind is ProgressKind.PROGRESS and c.state.mode is ConvergenceMode.NORMAL
    b=c.assess_reflection(hyp(7,stable=1),usage_totals={'tokens':20})
    assert b.kind is ProgressKind.NO_PROGRESS
    assert c.state.mode is ConvergenceMode.CONVERGENCE_REQUIRED
    assert c.state.first_stable_diagnosis_step==7

def test_two_true_no_progress_cycles_force_finalization_only_when_safe():
    c=ConvergenceController(no_progress_limit=2)
    c.assess_reflection(hyp(1,stable=0),usage_totals={})
    c.assess_reflection(hyp(2,stable=1),usage_totals={})
    c.assess_reflection(hyp(3,stable=2),usage_totals={})
    assert c.state.mode is ConvergenceMode.FORCE_FINALIZATION
    assert c.state.forced_finalization

def test_unsafe_stagnation_enters_budget_critical_then_exhausts():
    c=ConvergenceController(no_progress_limit=2)
    gap=[{'target':'need caller','location':'a.py','reason':'required'}]
    c.assess_reflection(hyp(1,status='supported',gaps=gap,stable=0,gfp='x'),usage_totals={})
    c.assess_reflection(hyp(2,status='supported',gaps=gap,stable=1,gfp='x'),usage_totals={})
    c.assess_reflection(hyp(3,status='supported',gaps=gap,stable=2,gfp='x'),usage_totals={})
    assert c.state.mode is ConvergenceMode.BUDGET_CRITICAL
    assert c.allow_critical_tool_attempt('resolve caller',hyp(3,status='supported',gaps=gap))
    a=c.assess_reflection(hyp(4,status='supported',gaps=gap,stable=3,gfp='x'),usage_totals={})
    assert a.kind is ProgressKind.NO_PROGRESS
    assert c.critical_failed_after_reflection(a)

def test_redundant_request_does_not_increment_no_progress():
    c=ConvergenceController()
    assert c.note_redundant()==1
    assert c.state.no_progress_streak==0
