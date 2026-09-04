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
class SemanticSearchConfig:
    enabled: bool = False
    provider: str = 'siliconflow'
    base_url: str = 'https://api.siliconflow.cn/v1'
    api_key: str = ''
    model: str = 'BAAI/bge-m3'
    dimension: int = 1024
    timeout: float = 60.0
    batch_size: int = 16
    max_retries: int = 3
    lexical_top_k: int = 20
    semantic_top_k: int = 20
    final_top_k: int = 10
    rrf_k: int = 60
    cache_path: str = '.debug_assistant/embedding_cache.sqlite'
    max_embedding_tokens: int = 6000
    max_isolation_depth: int = 5

@dataclass(slots=True)
class HarnessConfig:
    max_steps: int = 20
    max_tool_calls: int = 45
    max_context_chars: int = 50_000
    max_llm_calls: int = 40
    max_total_tokens: int = 180_000
    max_wall_time_seconds: int = 900
    finalization_reserve_seconds: int = 90
    planner_start_guard_seconds: int = 60
    reflection_start_guard_seconds: int = 60
    planner_llm_timeout_seconds: int = 90
    reflection_llm_timeout_seconds: int = 90
    reporter_llm_timeout_seconds: int = 60
    llm_cleanup_margin_seconds: int = 15
    obligation_review_min_seconds: int = 120
    max_auto_obligation_presentations_per_reflection: int = 3
    focused_reflection_max_obligations: int = 3
    focused_reflection_max_chars: int = 20_000
    focused_reflection_timeout_seconds: float = 60.0
    evidence_bundle_max_chars: int = 16_000
    parallel_max_actions: int = 4
    parallel_group_timeout_seconds: float = 20.0
    tool_retry_attempts: int = 2
    retry_base_delay_seconds: float = 0.2
    retry_max_delay_seconds: float = 2.0
    semantic_no_progress_limit: int = 3
    obligation_presentation_max_chars: int = 8_000
    provider_health_window: int = 5
    provider_failure_threshold: int = 3
    provider_consecutive_failures: int = 2
    provider_recovery_successes: int = 2
    provider_degraded_timeout_seconds: int = 60
    max_consecutive_planner_contract_failures: int = 2
    max_repeat_action: int = 2
    max_no_progress_steps: int = 4
    reflect_every: int = 5
    max_consecutive_reflection_failures: int = 2
    max_tool_output_chars: int = 14_000
    max_reporter_context_tokens: int = 12_000
    max_evidence_per_file: int = 3
    max_snippet_lines: int = 120
    max_trace_summary_items: int = 12
    recent_observation_count: int = 2   # backward-compatible fallback setting
    recent_observation_chars: int = 16_000
    trace_dir: str = ".debug_assistant/traces"
    build_task_index: bool = True
    keep_task_index: bool = False
    context: ContextConfig = field(default_factory=ContextConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)

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
                max_llm_calls=int(get('DEBUG_AGENT_MAX_LLM_CALLS','40')),
                max_total_tokens=int(get('DEBUG_AGENT_MAX_TOTAL_TOKENS','180000')),
                max_wall_time_seconds=int(get('DEBUG_AGENT_MAX_WALL_TIME_SECONDS','900')),
                finalization_reserve_seconds=int(get('DEBUG_AGENT_FINALIZATION_RESERVE_SECONDS','90')),
                planner_start_guard_seconds=int(get('DEBUG_AGENT_PLANNER_START_GUARD_SECONDS','60')),
                reflection_start_guard_seconds=int(get('DEBUG_AGENT_REFLECTION_START_GUARD_SECONDS','60')),
                planner_llm_timeout_seconds=int(get('DEBUG_AGENT_PLANNER_LLM_TIMEOUT_SECONDS','90')),
                reflection_llm_timeout_seconds=int(get('DEBUG_AGENT_REFLECTION_LLM_TIMEOUT_SECONDS','90')),
                reporter_llm_timeout_seconds=int(get('DEBUG_AGENT_REPORTER_LLM_TIMEOUT_SECONDS','60')),
                llm_cleanup_margin_seconds=int(get('DEBUG_AGENT_LLM_CLEANUP_MARGIN_SECONDS','15')),
                obligation_review_min_seconds=int(get('DEBUG_AGENT_OBLIGATION_REVIEW_MIN_SECONDS','120')),
                max_auto_obligation_presentations_per_reflection=int(get('DEBUG_AGENT_MAX_AUTO_OBLIGATION_PRESENTATIONS_PER_REFLECTION','3')),
                focused_reflection_max_obligations=int(get('DEBUG_AGENT_FOCUSED_REFLECTION_MAX_OBLIGATIONS','3')),
                focused_reflection_max_chars=int(get('DEBUG_AGENT_FOCUSED_REFLECTION_MAX_CHARS','20000')),
                focused_reflection_timeout_seconds=float(get('DEBUG_AGENT_FOCUSED_REFLECTION_TIMEOUT_SECONDS','60')),
                evidence_bundle_max_chars=int(get('DEBUG_AGENT_EVIDENCE_BUNDLE_MAX_CHARS','16000')),
                parallel_max_actions=int(get('DEBUG_AGENT_PARALLEL_MAX_ACTIONS','4')),
                parallel_group_timeout_seconds=float(get('DEBUG_AGENT_PARALLEL_GROUP_TIMEOUT_SECONDS','20')),
                tool_retry_attempts=int(get('DEBUG_AGENT_TOOL_RETRY_ATTEMPTS','2')),
                retry_base_delay_seconds=float(get('DEBUG_AGENT_RETRY_BASE_DELAY_SECONDS','0.2')),
                retry_max_delay_seconds=float(get('DEBUG_AGENT_RETRY_MAX_DELAY_SECONDS','2.0')),
                semantic_no_progress_limit=int(get('DEBUG_AGENT_SEMANTIC_NO_PROGRESS_LIMIT','3')),
                obligation_presentation_max_chars=int(get('DEBUG_AGENT_OBLIGATION_PRESENTATION_MAX_CHARS','8000')),
                provider_health_window=int(get('DEBUG_AGENT_PROVIDER_HEALTH_WINDOW','5')),
                provider_failure_threshold=int(get('DEBUG_AGENT_PROVIDER_FAILURE_THRESHOLD','3')),
                provider_consecutive_failures=int(get('DEBUG_AGENT_PROVIDER_CONSECUTIVE_FAILURES','2')),
                provider_recovery_successes=int(get('DEBUG_AGENT_PROVIDER_RECOVERY_SUCCESSES','2')),
                provider_degraded_timeout_seconds=int(get('DEBUG_AGENT_PROVIDER_DEGRADED_TIMEOUT_SECONDS','60')),
                max_consecutive_planner_contract_failures=int(get('DEBUG_AGENT_MAX_CONSECUTIVE_PLANNER_CONTRACT_FAILURES','2')),
                max_repeat_action=int(get("DEBUG_AGENT_MAX_REPEAT_ACTION", "2")),
                max_no_progress_steps=int(get("DEBUG_AGENT_MAX_NO_PROGRESS_STEPS", "4")),
                reflect_every=int(get("DEBUG_AGENT_REFLECT_EVERY", "5")),
                max_consecutive_reflection_failures=int(get("DEBUG_AGENT_MAX_CONSECUTIVE_REFLECTION_FAILURES", "2")),
                max_tool_output_chars=int(get("DEBUG_AGENT_MAX_TOOL_OUTPUT_CHARS", "14000")),
                max_reporter_context_tokens=int(get("DEBUG_AGENT_MAX_REPORTER_CONTEXT_TOKENS", "12000")),
                max_evidence_per_file=int(get("DEBUG_AGENT_MAX_EVIDENCE_PER_FILE", "3")),
                max_snippet_lines=int(get("DEBUG_AGENT_MAX_SNIPPET_LINES", "120")),
                max_trace_summary_items=int(get("DEBUG_AGENT_MAX_TRACE_SUMMARY_ITEMS", "12")),
                recent_observation_count=int(get("DEBUG_AGENT_RECENT_OBSERVATION_COUNT", "2")),
                recent_observation_chars=int(get("DEBUG_AGENT_RECENT_OBSERVATION_CHARS", "16000")),
                trace_dir=get("DEBUG_AGENT_TRACE_DIR", ".debug_assistant/traces"),
                build_task_index=get("DEBUG_AGENT_BUILD_INDEX", "1").lower() not in ("0","false","no"),
                keep_task_index=get("DEBUG_AGENT_KEEP_INDEX", "0").lower() in ("1","true","yes"),
            )
        )
        cfg.harness.semantic_search=SemanticSearchConfig(
            enabled=get('DEBUG_AGENT_SEMANTIC_SEARCH','0').lower() in ('1','true','yes'),
            provider=get('DEBUG_AGENT_EMBEDDING_PROVIDER','siliconflow'),
            base_url=get('DEBUG_AGENT_EMBEDDING_BASE_URL','https://api.siliconflow.cn/v1').rstrip('/'),
            api_key=get('DEBUG_AGENT_EMBEDDING_API_KEY',get('SILICONFLOW_API_KEY','')),
            model=get('DEBUG_AGENT_EMBEDDING_MODEL','BAAI/bge-m3'),
            dimension=int(get('DEBUG_AGENT_EMBEDDING_DIMENSION','1024')),
            timeout=float(get('DEBUG_AGENT_EMBEDDING_TIMEOUT','60')),
            batch_size=int(get('DEBUG_AGENT_EMBEDDING_BATCH_SIZE','16')),
            max_retries=int(get('DEBUG_AGENT_EMBEDDING_MAX_RETRIES','3')),
            lexical_top_k=int(get('DEBUG_AGENT_LEXICAL_TOP_K','20')),
            semantic_top_k=int(get('DEBUG_AGENT_SEMANTIC_TOP_K','20')),
            final_top_k=int(get('DEBUG_AGENT_SEARCH_FINAL_TOP_K','10')),
            rrf_k=int(get('DEBUG_AGENT_RRF_K','60')),
            cache_path=get('DEBUG_AGENT_EMBEDDING_CACHE','.debug_assistant/embedding_cache.sqlite'),
            max_embedding_tokens=int(get('DEBUG_AGENT_EMBEDDING_MAX_TOKENS','6000')),
            max_isolation_depth=int(get('DEBUG_AGENT_EMBEDDING_MAX_ISOLATION_DEPTH','5')),
        )
        cfg.harness.features.native_tool_calling = get("DEBUG_AGENT_NATIVE_TOOL_CALLING", "1").lower() in ("1", "true", "yes")
        cfg.harness.features.structured_reflection = get("DEBUG_AGENT_STRUCTURED_REFLECTION", "1").lower() in ("1", "true", "yes")
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
        sem=data.get('semantic_search') or {}
        for key,val in sem.items():
            if hasattr(self.harness.semantic_search,key): setattr(self.harness.semantic_search,key,val)
        self.harness.features.validate()
        return self
