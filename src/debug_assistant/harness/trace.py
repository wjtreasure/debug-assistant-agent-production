from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, is_dataclass
import json, time, uuid

from debug_assistant.security.redaction import redact_sensitive


class TraceRecorder:
    def __init__(self, trace_dir: str, task_id: str):
        d=Path(trace_dir); d.mkdir(parents=True,exist_ok=True)
        self.run_id=f"{task_id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.path=d/f"{self.run_id}.jsonl"; self.seq=0

    def record(self,event_type: str,payload):
        if is_dataclass(payload): payload=asdict(payload)
        payload=redact_sensitive(payload)
        self.seq+=1
        row={"schema_version":"2.6","seq":self.seq,"ts":time.time(),"run_id":self.run_id,"type":event_type,"payload":payload}
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')

    def export_meta(self):
        return {"run_id":self.run_id,"trace_path":str(self.path)}
