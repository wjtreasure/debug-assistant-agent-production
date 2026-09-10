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


class ExceptionFlaky:
    def __init__(self): self.n=0
    def execute(self,**kwargs):
        self.n+=1
        if self.n==1: raise TimeoutError("temporary backend timeout")
        return ToolObservation('x',True,'ok',{},None,0)


def test_transient_tool_exception_is_normalized_and_retried():
    t=ExceptionFlaky(); o=execute_with_retry(t,{},attempts=2)
    assert o.ok and t.n==2


class PermanentException:
    def __init__(self): self.n=0
    def execute(self,**kwargs):
        self.n+=1
        raise ValueError("invalid backend request")


def test_non_transient_tool_exception_is_not_retried():
    t=PermanentException(); o=execute_with_retry(t,{},attempts=3)
    assert not o.ok and o.error_type == "ValueError" and t.n==1
    assert o.metadata["execution_failure"] is True
    assert o.metadata["retryable"] is False
