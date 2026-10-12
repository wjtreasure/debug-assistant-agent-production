from __future__ import annotations

import json

import pytest

from debug_assistant.config import AppConfig, BaseAppConfig, load_effective_config


def _write_protocol(tmp_path, *, variant: dict | None = None, shared: dict | None = None):
    bundle = {
        "shared": {
            "protocol_revision": "r6.7b-test",
            "model": "protocol-model",
            "temperature": 0.1,
            "max_total_tokens": 12345,
            "public_fault_taxonomy": True,
            **(shared or {}),
        },
        "variants": {"A2": {"execution_mode": "dag", **(variant or {})}},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_base_config_uses_os_environment_and_keeps_role_models_together(monkeypatch):
    monkeypatch.setenv("DEBUG_AGENT_MAX_TOTAL_TOKENS", "180000")
    monkeypatch.setenv("DEBUG_AGENT_MODEL", "os-planner")
    monkeypatch.setenv("DEBUG_AGENT_REFLCETION_MODEL", "os-reflection")
    monkeypatch.setenv("DEBUG_AGENT_CRITIC_MODEL", "os-review")
    config = BaseAppConfig.from_env()

    assert config.model.planner_model == "os-planner"
    assert config.model.reflection_model == "os-reflection"
    assert config.model.critic_model == "os-review"
    assert config.runtime.max_total_tokens == 180000


def test_correctly_spelled_reflection_model_variable_is_supported(monkeypatch):
    monkeypatch.delenv("DEBUG_AGENT_REFLCETION_MODEL", raising=False)
    monkeypatch.setenv("DEBUG_AGENT_REFLECTION_MODEL", "spelled-correctly")
    config = BaseAppConfig.from_env()

    assert config.model.reflection_model == "spelled-correctly"
    assert config.source_map["model.reflection_model"] == "os"


def test_correctly_spelled_reflection_model_takes_precedence(monkeypatch):
    monkeypatch.setenv("DEBUG_AGENT_REFLCETION_MODEL", "legacy-name")
    monkeypatch.setenv("DEBUG_AGENT_REFLECTION_MODEL", "public-name")
    config = BaseAppConfig.from_env()

    assert config.model.reflection_model == "public-name"


def test_legacy_reflection_source_is_reported_when_public_alias_is_empty(monkeypatch):
    monkeypatch.setenv("DEBUG_AGENT_REFLECTION_MODEL", "")
    monkeypatch.setenv("DEBUG_AGENT_REFLCETION_MODEL", "legacy-name")
    config = BaseAppConfig.from_env()

    assert config.model.reflection_model == "legacy-name"
    assert config.source_map["model.reflection_model"] == "os"


def test_protocol_and_arm_are_typed_overlays(tmp_path):
    base = AppConfig.from_env()
    effective = load_effective_config(
        _write_protocol(tmp_path), arm_id="A2", base=base,
    )

    assert effective.model.planner_model == "protocol-model"
    assert effective.model.reflection_model == "protocol-model"
    assert effective.model.critic_model == "protocol-model"
    assert effective.runtime.max_total_tokens == 12345
    assert effective.runtime.execution_mode == "dag"
    assert effective.runtime.features.public_fault_taxonomy is True
    assert effective.source_map["model.planner_model"] == "protocol"
    assert effective.source_map["runtime.features.public_fault_taxonomy"] == "protocol"


def test_arm_cannot_override_shared_model_or_budget(tmp_path):
    with pytest.raises(ValueError, match="unknown ArmDelta fields"):
        load_effective_config(
            _write_protocol(tmp_path, variant={"model": "arm-model"}),
            arm_id="A2", base=AppConfig.from_env(),
        )

    with pytest.raises(ValueError, match="unknown ArmDelta fields"):
        load_effective_config(
            _write_protocol(tmp_path, variant={"max_total_tokens": 1}),
            arm_id="A2", base=AppConfig.from_env(),
        )


def test_protocol_rejects_secret_fields(tmp_path):
    with pytest.raises(ValueError, match="secret field"):
        load_effective_config(
            _write_protocol(tmp_path, shared={"api_key": "must-not-enter-protocol"}),
            base=AppConfig.from_env(),
        )


def test_authorized_protocol_override_is_whitelisted(tmp_path):
    path = _write_protocol(tmp_path)
    effective = load_effective_config(
        path,
        arm_id="A2",
        base=AppConfig.from_env(),
        protocol_overrides={"max_total_tokens": 777},
    )
    assert effective.runtime.max_total_tokens == 777

    with pytest.raises(ValueError, match="unknown protocol override fields"):
        load_effective_config(
            path,
            base=AppConfig.from_env(),
            protocol_overrides={"api_key": "blocked"},
        )


def test_effective_snapshot_is_deterministic_and_redacted(monkeypatch):
    monkeypatch.setenv("DEBUG_AGENT_API_KEY", "secret-value")
    first = AppConfig.from_env().snapshot()
    second = AppConfig.from_env().snapshot()

    assert first == second
    assert first["effective_config_fingerprint"]
    assert "secret-value" not in json.dumps(first, sort_keys=True)
    assert "api_key" not in first["model"]
