from __future__ import annotations
import re

def parse_unified_patch(patch:str):
    files=[]; cur=None
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            parts=line.split(); old=parts[2][2:]; new=parts[3][2:]
            cur={"path":new,"old_path":old,"modified_ranges":[]}; files.append(cur)
        elif line.startswith('@@') and cur is not None:
            m=re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?',line)
            if m:
                a,b,c,d=int(m.group(1)),int(m.group(2) or 1),int(m.group(3)),int(m.group(4) or 1)
                cur['modified_ranges'].append({"old_start":a,"old_end":a+max(b,1)-1,"new_start":c,"new_end":c+max(d,1)-1})
    return {"files":files}
