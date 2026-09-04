import json
from pathlib import Path
from debug_assistant.evaluation.localization import evaluate_dataset

def test_missing_predictions_count_as_misses(tmp_path):
    gold=tmp_path/'gold'; gold.mkdir()
    for name in ('a','b'):
        d=gold/name; d.mkdir(); (d/'ground_truth.json').write_text(json.dumps({'files':['x.py'],'symbols':[]}))
    pred=tmp_path/'p.jsonl'; pred.write_text(json.dumps({'task_id':'a','report':{'likely_files':['x.py']}})+'\n')
    r=evaluate_dataset(gold,pred)
    assert r['coverage']['expected_tasks']==2 and r['coverage']['missing_predictions']==1
    assert r['aggregate']['n']==2 and r['aggregate']['file_hit@1']==0.5
