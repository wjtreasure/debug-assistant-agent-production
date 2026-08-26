from debug_assistant.harness.tool_executor import execute_with_retry
from debug_assistant.models import ToolObservation

class Flaky:
    def __init__(self): self.n=0
    def execute(self,**kwargs):
        self.n+=1
        if self.n==1:return ToolObservation('x',False,'timeout',{},'TimeoutError',0)
        return ToolObservation('x',True,'ok',{},None,0)

def test_transient_tool_retry():
    t=Flaky(); o=execute_with_retry(t,{},attempts=2)
    assert o.ok and t.n==2
