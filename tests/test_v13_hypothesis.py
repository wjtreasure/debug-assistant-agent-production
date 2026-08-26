from debug_assistant.memory.hypothesis import HypothesisManager


def test_runtime_fingerprint_drives_stability_not_model_claim():
    h=HypothesisManager()
    review={'current_diagnosis':'Zero irradiance causes division by zero','supporting_evidence_ids':['ev-1'],
            'contradicting_evidence_ids':[],'missing':[],'evidence_sufficient':True,'confidence':.9,'hypothesis_changed':True}
    a=h.update(review,5); assert a.stable_reflections==0
    review['hypothesis_changed']=False
    b=h.update(review,10); assert b.stable_reflections==1
    # Text changes => runtime fingerprint resets even if model claims unchanged.
    review['current_diagnosis']='Different mechanism'; review['hypothesis_changed']=False
    c=h.update(review,15); assert c.stable_reflections==0
