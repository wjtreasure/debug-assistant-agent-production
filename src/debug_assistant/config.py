from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os, json
from dotenv import load_dotenv
from debug_assistant.harness.feature_flags import FeatureFlags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

@dataclass(slots=True)
class ModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    planner_model: str = "gpt-5.6"
    critic_model: str = ""
    timeout: float = 60.0
    temperature: float = 0.0

@dataclass(slots=True)
class ContextConfig:
    max_item_chars: int = 12_000
    safety_margin_chars: int = 4_000
    fallback_recent_count: int = 2
    fallback_recent_chars: int = 16_000
    known_index_max_chars: int = 3_500
    target_active_items: int = 8
    hard_active_items: int = 12

@dataclass(slots=True)
class HarnessConfig:
    max_steps: int = 20
    max_tool_calls: int = 45
    max_context_chars: int = 50_000
    max_repeat_action: int = 2
    max_no_progress_steps: int = 4
    reflect_every: int = 5
    max_consecutive_reflection_failures: int = 2
    max_tool_output_chars: int = 14_000
    recent_observation_count: int = 2   # backward-compatible fallback setting
    recent_observation_chars: int = 16_000
    trace_dir: str = ".debug_assistant/traces"
    build_task_index: bool = True
    keep_task_index: bool = False
    context: ContextConfig = field(default_factory=ContextConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        get=os.getenv
        cfg=cls(
            model=ModelConfig(
                provider=get("DEBUG_AGENT_PROVIDER", "openai_compatible"),
                base_url=get("DEBUG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip('/'),
                api_key=get("DEBUG_AGENT_API_KEY", ""),
                planner_model=get("DEBUG_AGENT_MODEL", "gpt-5.6"),
                critic_model=get("DEBUG_AGENT_CRITIC_MODEL", ""),
                timeout=float(get("DEBUG_AGENT_TIMEOUT", "60")),
                temperature=float(get("DEBUG_AGENT_TEMPERATURE", "0")),
            ),
            harness=HarnessConfig(
                max_steps=int(get("DEBUG_AGENT_MAX_STEPS", "20")),
                max_tool_calls=int(get("DEBUG_AGENT_MAX_TOOL_CALLS", "45")),
                max_context_chars=int(get("DEBUG_AGENT_MAX_CONTEXT_CHARS", "50000")),
                max_repeat_action=int(get("DEBUG_AGENT_MAX_REPEAT_ACTION", "2")),
                max_no_progress_steps=int(get("DEBUG_AGENT_MAX_NO_PROGRESS_STEPS", "4")),
                reflect_every=int(get("DEBUG_AGENT_REFLECT_EVERY", "5")),
                max_consecutive_reflection_failures=int(get("DEBUG_AGENT_MAX_CONSECUTIVE_REFLECTION_FAILURES", "2")),
                max_tool_output_chars=int(get("DEBUG_AGENT_MAX_TOOL_OUTPUT_CHARS", "14000")),
                recent_observation_count=int(get("DEBUG_AGENT_RECENT_OBSERVATION_COUNT", "2")),
                recent_observation_chars=int(get("DEBUG_AGENT_RECENT_OBSERVATION_CHARS", "16000")),
                trace_dir=get("DEBUG_AGENT_TRACE_DIR", ".debug_assistant/traces"),
                build_task_index=get("DEBUG_AGENT_BUILD_INDEX", "1").lower() not in ("0","false","no"),
                keep_task_index=get("DEBUG_AGENT_KEEP_INDEX", "0").lower() in ("1","true","yes"),
            )
        )
        cfg.harness.context.fallback_recent_count=cfg.harness.recent_observation_count
        cfg.harness.context.fallback_recent_chars=cfg.harness.recent_observation_chars
        return cfg

    def apply_experiment_file(self, path: str | None) -> "AppConfig":
        if not path: return self
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        c=data.get('context') or {}
        for key in ('max_item_chars','safety_margin_chars','fallback_recent_count','fallback_recent_chars','known_index_max_chars','target_active_items','hard_active_items'):
            if key in c: setattr(self.harness.context,key,int(c[key]))
        f=data.get('features') or {}
        for key,val in f.items():
            if hasattr(self.harness.features,key): setattr(self.harness.features,key,bool(val))
        self.harness.features.validate()
        return self
