import json
from debug_assistant.config import AppConfig
from debug_assistant.evaluation.batch import run_prepared

class Harness:
    def __init__(self): self.config=AppConfig()
    def run(self,task,output):
        return {'state':{'status':'success'},'report':{'likely_files':['x.py']},'trace':{},'failure':None,'report_source':'llm'}

def _tasks(tmp_path):
    root=tmp_path/'tasks'; root.mkdir(); d=root/'t1'; d.mkdir()
    (d/'task.json').write_text(json.dumps({'task_id':'t1','workspace':str(d),'repo':'x','base_commit':'abc'}))
    (d/'issue.md').write_text('issue'); (d/'ground_truth.json').write_text(json.dumps({'files':['x.py']}))
    return root

def test_no_resume_truncates_and_resume_fingerprint_protects(tmp_path):
    root=_tasks(tmp_path); out=tmp_path/'out'; h=Harness()
    run_prepared(root,out,h,resume=False)
    run_prepared(root,out,h,resume=False)
    assert len((out/'predictions.jsonl').read_text().splitlines())==1
    h.config.harness.max_steps=99
    try: run_prepared(root,out,h,resume=True)
    except RuntimeError as e: assert 'fingerprint' in str(e)
    else: raise AssertionError('expected fingerprint mismatch')
