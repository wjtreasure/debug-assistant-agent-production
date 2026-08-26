from debug_assistant.evaluation.localization import evaluate_one

def test_metrics():
    g={'files':['a.py'],'symbols':[{'file':'a.py','symbol':'foo'}]}
    p={'likely_files':['b.py','a.py'],'likely_symbols':['foo'],'recommended_change_points':[]}
    m=evaluate_one(g,p)
    assert m['file_hit1']==0 and m['file_hit3']==1 and m['file_mrr']==0.5 and m['symbol_hit']==1
