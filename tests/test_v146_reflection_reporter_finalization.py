import json
import re

import pytest

from debug_assistant.agent.reporter import Reporter
from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.llm.base import LLMDeadlineExceeded
from debug_assistant.models import Evidence, TaskSpec


def _evidence(eid="ev-1"):
    return Evidence(eid, "read_file", "read_file", "source", file="a.py",
                    raw_observation_id="obs-1", source_start_line=1, source_end_line=3)


def _report(claim_type="source_fact", status="observed", evidence_ids=None):
    return {
        "summary": "partial report", "root_cause": "uncertain",
        "likely_files": ["a.py"], "likely_symbols": [], "impact_scope": [],
        "recommended_change_points": [], "uncertainties": [], "next_checks": [],
        "evidence_ids": list(evidence_ids or ["ev-1"]), "confidence": 0.3,
        "claims": [{"text": "source was inspected", "claim_type": claim_type,
                    "status": status, "evidence_ids": list(evidence_ids or ["ev-1"])}],
    }


class ReporterRepairLLM:
    def __init__(self, repair_valid=True):
        self.calls = []
        self.repair_valid = repair_valid

    def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
        self.calls.append({"prompt_chars": len(system) + len(user)})
        if "REPORTER_FORMAT_REPAIR" in user:
            return _report() if self.repair_valid else _report("fact", "partial")
        return _report("fact", "partial")


def test_reporter_invalid_claims_get_one_bounded_format_repair():
    llm = ReporterRepairLLM(repair_valid=True)
    reporter = Reporter(llm, compact_prompt=True)
    report = reporter.build("t", "ctx", [_evidence()])
    assert report.report_source == "llm"
    assert len(llm.calls) == 2
    assert [x[0] for x in reporter.last_events] == [
        "REPORTER_CONTRACT_INVALID", "REPORTER_REPAIR_STARTED",
        "REPORTER_REPAIR_FINISHED", "REPORTER_CORRECTED",
    ]
    assert report.claims[0]["claim_type"] == "source_fact"


def test_reporter_invalid_repair_falls_back_after_exactly_one_repair():
    llm = ReporterRepairLLM(repair_valid=False)
    reporter = Reporter(llm, compact_prompt=True)
    with pytest.raises(ValueError, match="after one repair"):
        reporter.build("t", "ctx", [_evidence()])
    assert len(llm.calls) == 2
    assert reporter.last_fallback_reason == "contract_invalid_after_repair"


def test_reporter_deadline_denies_repair():
    llm = ReporterRepairLLM(repair_valid=True)
    reporter = Reporter(llm, compact_prompt=True)
    with pytest.raises(ValueError, match="report schema validation failed"):
        reporter.build("t", "ctx", [_evidence()], logical_timeout_seconds=0)
    assert len(llm.calls) == 1
    assert reporter.last_fallback_reason == "stage_deadline_insufficient"


def test_reporter_repair_drops_unknown_claim_evidence_id():
    class UnknownEvidenceLLM(ReporterRepairLLM):
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.calls.append({"prompt_chars": len(system) + len(user)})
            if "REPORTER_FORMAT_REPAIR" in user:
                return _report(evidence_ids=["unknown-id"])
            return _report("fact", "partial")

    report = Reporter(UnknownEvidenceLLM(), compact_prompt=True).build("t", "ctx", [_evidence()])
    assert report.claims == []
    assert "unknown-id" not in report.evidence_ids


class ReflectionReporterFaultLLM:
    """Two real Reflector calls fail only after a READY obligation exists."""

    def __init__(self):
        self.calls = []
        self.planner_calls = 0
        self.reflection_calls = 0
        self.reporter_calls = 0

    def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
        low = user.lower()
        self.calls.append({"model": "fake", "prompt_tokens": 1, "completion_tokens": 1,
                           "total_tokens": 2, "input_tokens": 1, "output_tokens": 1,
                           "prompt_chars": len(system) + len(user), "completion_chars": 1})
        if "reporter_format_repair" in low:
            self.reporter_calls += 1
            return _report("fact", "partial")
        if "final_report_schema" in low:
            self.reporter_calls += 1
            return _report("fact", "partial")
        if "reflection_schema" in low:
            self.reflection_calls += 1
            if "focused_reflection" in low:
                raise LLMDeadlineExceeded("focused reflection timeout")
            ids = list(dict.fromkeys(re.findall(r"ev-[A-Za-z0-9-]+", user)))
            return {
                "decision": "continue", "reason": "source needs a causal review",
                "current_diagnosis": "a.py contains the relevant source boundary",
                "root_cause_target": None, "root_cause_location": None,
                "root_cause_mechanism": None, "evidence_sufficient": False,
                "supporting_evidence_ids": ids[:1], "contradicting_evidence_ids": [],
                "required_missing_evidence": [{
                    "target": "source boundary", "location": "a.py:1-3",
                    "reason": "review the source boundary", "goal_type": "behavior",
                }],
                "optional_validation": [], "recommended_next_goal": "review source",
                "confidence": 0.4, "hypothesis_changed": True,
            }
        self.planner_calls += 1
        if self.planner_calls <= 2:
            return {"kind": "tool", "skill": "repository_exploration",
                    "reason": "read source", "confidence": 0.8,
                    "tool": "read_file", "arguments": {"path": "a.py", "start_line": 1, "end_line": 3},
                    "expected_evidence": "source"}
        return {"kind": "finish", "skill": "report_synthesis", "reason": "finish",
                "confidence": 0.8, "tool": None, "arguments": {}}


def test_reflection_limit_and_reporter_repair_failure_finalize_safely(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("value = 1\nvalue = 2\nvalue = 3\n", encoding="utf-8")
    fake = ReflectionReporterFaultLLM()
    monkeypatch.setattr("debug_assistant.harness.runtime.build_llm", lambda cfg: fake)
    cfg = AppConfig(
        model=ModelConfig(provider="mock", planner_model="mock", critic_model="mock"),
        harness=HarnessConfig(
            build_task_index=False, trace_dir=str(tmp_path / "traces"), max_steps=8,
            reflect_every=2, max_consecutive_reflection_failures=2,
            planner_start_guard_seconds=0, reflection_start_guard_seconds=0,
            obligation_review_min_seconds=0,
        ),
    )
    result = AgentHarness(cfg).run(TaskSpec("joint-failure", "boundary issue", str(repo)))
    events = [json.loads(line) for line in open(result["trace"]["trace_path"], encoding="utf-8")]
    types = [event["type"] for event in events]
    assert fake.reflection_calls == 3  # one successful baseline + exactly two failed focused calls
    assert fake.reporter_calls == 2
    assert len([x for x in events if x["type"] == "REFLECTION_FAILED"]) == 2
    assert result["state"]["max_consecutive_reflection_failures_observed"] == 2
    assert result["state"]["status"] == "partial_success"
    assert result["report_source"] == "fallback"
    assert "RUN_END" in types and "RUN_FAILED" not in types
    degraded = [x["payload"] for x in events if x["type"] == "REFLECTION_RETRY_DEGRADED"]
    assert degraded and degraded[0]["retry_chars"] < degraded[0]["previous_chars"]
    assert degraded[0]["retry_obligation_count"] <= degraded[0]["previous_obligation_count"]
    assert any(x["type"] == "REPORTER_CONTRACT_INVALID" for x in events)
    assert any(x["type"] == "REPORTER_REPAIR_STARTED" for x in events)
    assert any(x["type"] == "REPORTER_FALLBACK_TRIGGERED" and
               x["payload"]["reason"] == "contract_invalid_after_repair" for x in events)
    commits = [x for x in events if x["type"] == "SEMANTIC_STATE_COMMIT"]
    assert len(commits) == 1
    reconciliation = [x for x in events if x["type"] == "FINAL_STATE_RECONCILIATION"][-1]["payload"]
    assert reconciliation["evidence_ready_unreviewed"]
    assert result["report"]["acquired_unreviewed"]
    assert not any(x["type"] in {"EVIDENCE_OBLIGATIONS_SATISFIED", "OBLIGATION_REVIEW_APPLIED"}
                   for x in events)
