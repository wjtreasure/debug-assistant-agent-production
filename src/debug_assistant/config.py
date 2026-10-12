from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import hashlib
import json
import os
from typing import Any, Mapping
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
    reflection_model: str = ""
    critic_model: str = ""
    timeout: float = 60.0
    temperature: float = 0.0
    retry_attempts: int = 3
    retry_min_budget_seconds: float = 5.0
    # Optional provider-advertised physical limits. ``None`` selects the
    # deterministic character budget when a provider omits a capability
    # snapshot.
    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizer: str = "char4"
    reserved_output_tokens: int = 0
    protocol_safety_reserve_tokens: int = 0

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


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_steps: int
    max_tool_calls: int
    max_llm_calls: int
    max_total_tokens: int
    terminal_reserve_tokens: int
    terminal_reserve_llm_calls: int
    review_reserve_tokens: int
    review_reserve_calls: int


@dataclass(frozen=True, slots=True)
class RetryConfig:
    provider_attempts: int
    tool_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    planner_seconds: float
    reflection_seconds: float
    review_seconds: float
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class ProgressConfig:
    max_duplicate_actions: int
    max_no_progress: int
    semantic_no_progress_limit: int
    reflect_every: int

@dataclass(slots=True)
class RuntimeConfig:
    # Runtime mode and arm-projected controls. These are configuration values,
    # not semantic state or policy decisions.
    execution_mode: str = "baseline"
    context_projection_mode: str = "task_aware"
    max_steps: int = 20
    max_tool_calls: int = 45
    max_context_chars: int = 50_000
    max_request_input_tokens: int | None = None
    max_llm_calls: int = 40
    max_total_tokens: int = 180_000
    max_cost_per_incident: float | None = None
    terminal_reserve_tokens: int = 0
    terminal_reserve_llm_calls: int = 3
    context_pressure_ratio: float = 0.75
    hard_pressure_ratio: float = 0.95
    max_wall_time_seconds: int = 900
    finalization_reserve_seconds: int = 90
    planner_start_guard_seconds: int = 60
    reflection_start_guard_seconds: int = 60
    planner_llm_timeout_seconds: int = 90
    reflection_llm_timeout_seconds: int = 90
    reporter_llm_timeout_seconds: int = 60
    review_llm_timeout_seconds: float = 60.0
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
    provider_retry_attempts: int = 3
    provider_retry_min_budget_seconds: float = 5.0
    max_consecutive_planner_contract_failures: int = 2
    max_post_reject_tool_calls: int = 3
    max_post_reject_steps: int = 4
    max_post_sufficiency_tool_calls: int = 4
    max_tool_contract_repairs: int = 1
    max_duplicate_actions: int = 2
    max_no_progress: int = 4
    reflect_every: int = 5
    max_consecutive_reflection_failures: int = 2
    max_tool_output_chars: int = 14_000
    max_reporter_context_tokens: int = 12_000
    planner_max_completion_tokens: int = 8_000
    reflection_max_completion_tokens: int = 3_000
    converge_max_reasoning_tokens: int = 2_000
    converge_max_output_tokens: int = 3_000
    converge_max_read_file_calls: int = 3
    converge_disable_reflections: bool = False
    source_verify_max_output_tokens: int = 6_000
    source_verify_max_read_file_calls: int = 6
    review_max_completion_tokens: int = 8_000
    review_repair_max_completion_tokens: int = 4_000
    review_reserve_tokens: int = 24_000
    review_reserve_calls: int = 2
    max_review_rounds: int = 2
    enable_reflection: bool = True
    max_reflection_calls: int = 2
    max_consecutive_reflection_calls: int = 1
    enable_review: bool = True
    max_review_recovery_cycles: int = 1
    tool_circuit_failure_threshold: int = 2
    planner_context_chars: int = 12_000
    tool_exposure_policy_version: str = "unified-visible-tools-v2-candidate"
    domain_asset_manifest_path: str = str(
        PROJECT_ROOT / "docs" / "refactor" / "DOMAIN_ASSET_MANIFEST.json"
    )
    max_evidence_per_file: int = 3
    max_snippet_lines: int = 120
    max_trace_summary_items: int = 12
    recent_observation_count: int = 2
    recent_observation_chars: int = 16_000
    trace_dir: str = ".debug_assistant/traces"
    build_task_index: bool = True
    keep_task_index: bool = False
    context: ContextConfig = field(default_factory=ContextConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)

    @property
    def max_repeat_action(self) -> int:
        """Compatibility name for the canonical duplicate-action limit."""
        return self.max_duplicate_actions

    @property
    def max_no_progress_steps(self) -> int:
        """Compatibility projection for the canonical no-progress limit."""
        return self.max_no_progress

    @property
    def budget(self) -> "BudgetConfig":
        return BudgetConfig(
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            max_total_tokens=self.max_total_tokens,
            terminal_reserve_tokens=self.terminal_reserve_tokens,
            terminal_reserve_llm_calls=self.terminal_reserve_llm_calls,
            review_reserve_tokens=self.review_reserve_tokens,
            review_reserve_calls=self.review_reserve_calls,
        )

    @property
    def retry(self) -> "RetryConfig":
        return RetryConfig(
            provider_attempts=self.provider_retry_attempts,
            tool_attempts=self.tool_retry_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
        )

    @property
    def timeout(self) -> "TimeoutConfig":
        return TimeoutConfig(
            planner_seconds=float(self.planner_llm_timeout_seconds),
            reflection_seconds=float(self.reflection_llm_timeout_seconds),
            review_seconds=float(self.review_llm_timeout_seconds),
            wall_time_seconds=float(self.max_wall_time_seconds),
        )

    @property
    def progress(self) -> "ProgressConfig":
        return ProgressConfig(
            max_duplicate_actions=self.max_duplicate_actions,
            max_no_progress=self.max_no_progress,
            semantic_no_progress_limit=self.semantic_no_progress_limit,
            reflect_every=self.reflect_every,
        )

    @classmethod
    def from_effective_app(
        cls, source: Any, *, trace_dir: str | None = None,
    ) -> "RuntimeConfig":
        """Create the runtime projection from one effective app config.

        This is a complete typed projection, not a second configuration source.
        It deliberately fails when a required runtime field is absent.
        """
        runtime = getattr(source, "runtime", None)
        if runtime is None:
            runtime = getattr(source, "harness", None)
        if not isinstance(runtime, cls):
            raise TypeError("effective app config must expose RuntimeConfig.runtime")
        return replace(runtime, trace_dir=trace_dir or runtime.trace_dir)

    @classmethod
    def from_app_harness(
        cls, source: Any, *, trace_dir: str | None = None,
        planner_llm_timeout_seconds: float | None = None,
        review_llm_timeout_seconds: float | None = None,
    ) -> "RuntimeConfig":
        """Compatibility entry that now returns a complete runtime projection."""
        result = cls.from_effective_app(source, trace_dir=trace_dir)
        updates: dict[str, Any] = {}
        if planner_llm_timeout_seconds is not None:
            updates["planner_llm_timeout_seconds"] = planner_llm_timeout_seconds
        if review_llm_timeout_seconds is not None:
            updates["review_llm_timeout_seconds"] = review_llm_timeout_seconds
        return replace(result, **updates) if updates else result

    def __post_init__(self) -> None:
        if self.execution_mode not in {"baseline", "dag"}:
            raise ValueError("mode must be baseline or dag")
        if self.context_projection_mode not in {"existing", "task_aware"}:
            raise ValueError("context_projection_mode must be existing or task_aware")
        if self.max_steps <= 0 or self.max_tool_calls <= 0:
            raise ValueError("step and tool budgets must be positive")
        if self.max_review_rounds != 2:
            raise ValueError("max_review_rounds is fixed at 2")
        if not 1 <= self.max_post_reject_tool_calls <= 3:
            raise ValueError("post-review tool calls must be between 1 and 3")
        if not 1 <= self.max_post_reject_steps <= 4:
            raise ValueError("post-review steps must be between 1 and 4")
        if not 1 <= self.max_post_sufficiency_tool_calls <= 8:
            raise ValueError("post-sufficiency tool calls must be between 1 and 8")
        if not 1 <= self.max_tool_contract_repairs <= 3:
            raise ValueError("tool contract repairs must be between 1 and 3")
        if self.planner_llm_timeout_seconds <= 0 or self.review_llm_timeout_seconds <= 0:
            raise ValueError("LLM timeouts must be positive")
        if self.max_context_chars <= 0 or self.planner_context_chars <= 0:
            raise ValueError("context limits must be positive")
        if self.max_request_input_tokens is not None and self.max_request_input_tokens <= 0:
            raise ValueError("max_request_input_tokens must be positive")
        if any(value <= 0 for value in (
            self.planner_max_completion_tokens,
            self.reflection_max_completion_tokens,
            self.converge_max_reasoning_tokens,
            self.converge_max_output_tokens,
            self.converge_max_read_file_calls,
            self.source_verify_max_output_tokens,
            self.source_verify_max_read_file_calls,
            self.review_max_completion_tokens,
            self.review_repair_max_completion_tokens,
            self.review_reserve_tokens,
            self.review_reserve_calls,
        )):
            raise ValueError("stage budget controls must be positive")
        if self.max_llm_calls <= 0 or self.max_total_tokens <= 0 or self.max_wall_time_seconds <= 0:
            raise ValueError("LLM, token, and wall-clock budgets must be positive")
        if self.terminal_reserve_tokens < 0 or self.terminal_reserve_llm_calls <= 0:
            raise ValueError("terminal reserves must be non-negative tokens and positive calls")
        if not 0.5 <= self.context_pressure_ratio < 1.0:
            raise ValueError("context pressure ratio must be in [0.5, 1.0)")
        if not self.context_pressure_ratio <= self.hard_pressure_ratio <= 1.0:
            raise ValueError("hard pressure ratio must be >= pressure ratio and <= 1")
        if self.max_cost_per_incident is not None and self.max_cost_per_incident <= 0:
            raise ValueError("max_cost_per_incident must be positive when configured")
        if self.finalization_reserve_seconds < 0 or self.finalization_reserve_seconds >= self.max_wall_time_seconds:
            raise ValueError("finalization reserve must be smaller than the run deadline")
        if self.tool_retry_attempts <= 0 or self.max_duplicate_actions <= 0 or self.max_no_progress <= 0:
            raise ValueError("retry and progress limits must be positive")
        if self.provider_retry_attempts <= 0 or self.provider_retry_min_budget_seconds < 0:
            raise ValueError("provider retry settings must be positive and non-negative")
        if self.max_reflection_calls < 0 or self.max_review_recovery_cycles < 0:
            raise ValueError("recovery limits must be non-negative")
        if self.max_consecutive_reflection_calls <= 0 or self.tool_circuit_failure_threshold <= 0:
            raise ValueError("recovery and circuit limits must be positive")

    @property
    def budget_allocation(self):
        # Lazy import avoids making the configuration boundary depend on the
        # Runtime orchestration module during package import.
        from debug_assistant.harness.dynamic_budget import BudgetAllocation
        return BudgetAllocation.resolve(
            total_tokens=self.max_total_tokens,
            total_llm_calls=self.max_llm_calls,
            review_tokens=self.review_reserve_tokens if self.enable_review else 0,
            review_llm_calls=self.review_reserve_calls if self.enable_review else 0,
            terminal_tokens=self.terminal_reserve_tokens,
            terminal_llm_calls=self.terminal_reserve_llm_calls,
            terminal_seconds=self.finalization_reserve_seconds,
        )


# Historical imports use HarnessConfig. It is an alias, not a second schema.
HarnessConfig = RuntimeConfig


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    revision: str
    model: str | None = None
    temperature: float | None = None
    provider_timeout_seconds: float | None = None
    provider_retry_attempts: int | None = None
    provider_retry_min_budget_seconds: float | None = None
    planner_timeout_seconds: float | None = None
    review_timeout_seconds: float | None = None
    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    max_total_tokens: int | None = None
    planner_max_completion_tokens: int | None = None
    review_reserve_tokens: int | None = None
    review_reserve_calls: int | None = None
    terminal_reserve_tokens: int | None = None
    terminal_reserve_llm_calls: int | None = None
    max_review_rounds: int | None = None
    tool_exposure_policy_version: str | None = None
    transport_profile: str | None = None
    public_fault_taxonomy: bool | None = None


@dataclass(frozen=True, slots=True)
class ArmDelta:
    arm_id: str
    execution_mode: str | None = None
    context_projection_mode: str | None = None
    evidence_qualification: bool | None = None
    context_projection_v2: bool | None = None
    evidence_rehydration: bool | None = None
    semantic_convergence_v2: bool | None = None
    public_fault_taxonomy: bool | None = None


@dataclass(frozen=True, slots=True)
class ExperimentRunSpec:
    protocol_path: str
    bundle_path: str | None
    arm_id: str | None
    case_manifest_path: str | None
    protocol_revision: str


@dataclass(slots=True)
class BaseAppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    source_map: dict[str, str] = field(default_factory=dict)

    @property
    def harness(self) -> RuntimeConfig:
        """Read-only compatibility projection for existing callers."""
        return self.runtime

    @property
    def provider(self) -> dict[str, str]:
        return {
            "provider": self.model.provider,
            "base_url": self.model.base_url,
        }

    @classmethod
    def from_env(cls) -> "BaseAppConfig":
        values = _environment_values()
        model = ModelConfig(
            provider=values.get("DEBUG_AGENT_PROVIDER", "openai_compatible").rstrip("/"),
            base_url=values.get("DEBUG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=values.get("DEBUG_AGENT_API_KEY", ""),
            planner_model=values.get("DEBUG_AGENT_MODEL", "gpt-5.6"),
            # Keep the historical misspelling as a compatibility alias while
            # making the correctly spelled variable the public interface.
            reflection_model=(
                values.get("DEBUG_AGENT_REFLECTION_MODEL")
                or values.get("DEBUG_AGENT_REFLCETION_MODEL", "")
            ),
            critic_model=values.get("DEBUG_AGENT_CRITIC_MODEL", ""),
            timeout=float(values.get("DEBUG_AGENT_TIMEOUT", "60")),
            temperature=float(values.get("DEBUG_AGENT_TEMPERATURE", "0")),
            retry_attempts=int(values.get("DEBUG_AGENT_RETRY_ATTEMPTS", "3")),
            retry_min_budget_seconds=float(values.get("DEBUG_AGENT_RETRY_MIN_BUDGET_SECONDS", "5")),
            context_window=(int(values.get("DEBUG_AGENT_CONTEXT_WINDOW", "0")) or None),
            max_output_tokens=(int(values.get("DEBUG_AGENT_MAX_OUTPUT_TOKENS", "0")) or None),
            tokenizer=values.get("DEBUG_AGENT_TOKENIZER", "char4"),
            reserved_output_tokens=int(values.get("DEBUG_AGENT_RESERVED_OUTPUT_TOKENS", "0")),
            protocol_safety_reserve_tokens=int(values.get("DEBUG_AGENT_PROTOCOL_SAFETY_RESERVE_TOKENS", "0")),
        )
        runtime = RuntimeConfig(
            max_steps=int(values.get("DEBUG_AGENT_MAX_STEPS", "20")),
            max_tool_calls=int(values.get("DEBUG_AGENT_MAX_TOOL_CALLS", "45")),
            max_context_chars=int(values.get("DEBUG_AGENT_MAX_CONTEXT_CHARS", "50000")),
            max_request_input_tokens=(int(values.get("DEBUG_AGENT_MAX_REQUEST_INPUT_TOKENS", "0")) or None),
            max_llm_calls=int(values.get("DEBUG_AGENT_MAX_LLM_CALLS", "40")),
            max_total_tokens=int(values.get("DEBUG_AGENT_MAX_TOTAL_TOKENS", "180000")),
            max_cost_per_incident=(float(values.get("DEBUG_AGENT_MAX_COST_PER_INCIDENT", "0")) or None),
            terminal_reserve_tokens=int(values.get("DEBUG_AGENT_TERMINAL_RESERVE_TOKENS", "0")),
            terminal_reserve_llm_calls=int(values.get("DEBUG_AGENT_TERMINAL_RESERVE_LLM_CALLS", "3")),
            context_pressure_ratio=float(values.get("DEBUG_AGENT_CONTEXT_PRESSURE_RATIO", "0.75")),
            hard_pressure_ratio=float(values.get("DEBUG_AGENT_HARD_PRESSURE_RATIO", "0.95")),
            max_wall_time_seconds=int(values.get("DEBUG_AGENT_MAX_WALL_TIME_SECONDS", "900")),
            finalization_reserve_seconds=int(values.get("DEBUG_AGENT_FINALIZATION_RESERVE_SECONDS", "90")),
            planner_start_guard_seconds=int(values.get("DEBUG_AGENT_PLANNER_START_GUARD_SECONDS", "60")),
            reflection_start_guard_seconds=int(values.get("DEBUG_AGENT_REFLECTION_START_GUARD_SECONDS", "60")),
            planner_llm_timeout_seconds=float(values.get("DEBUG_AGENT_PLANNER_LLM_TIMEOUT_SECONDS", "90")),
            reflection_llm_timeout_seconds=float(values.get("DEBUG_AGENT_REFLECTION_LLM_TIMEOUT_SECONDS", "90")),
            reporter_llm_timeout_seconds=float(values.get("DEBUG_AGENT_REPORTER_LLM_TIMEOUT_SECONDS", "60")),
            review_llm_timeout_seconds=float(values.get(
                "DEBUG_AGENT_REVIEW_LLM_TIMEOUT_SECONDS",
                values.get("DEBUG_AGENT_REPORTER_LLM_TIMEOUT_SECONDS", "60"),
            )),
            planner_max_completion_tokens=int(values.get("DEBUG_AGENT_PLANNER_MAX_COMPLETION_TOKENS", "8000")),
            reflection_max_completion_tokens=int(values.get("DEBUG_AGENT_REFLECTION_MAX_COMPLETION_TOKENS", "3000")),
            converge_max_reasoning_tokens=int(values.get("DEBUG_AGENT_CONVERGE_MAX_REASONING_TOKENS", "2000")),
            converge_max_output_tokens=int(values.get("DEBUG_AGENT_CONVERGE_MAX_OUTPUT_TOKENS", "3000")),
            converge_max_read_file_calls=int(values.get("DEBUG_AGENT_CONVERGE_MAX_READ_FILE_CALLS", "3")),
            converge_disable_reflections=_as_bool(values.get("DEBUG_AGENT_CONVERGE_DISABLE_REFLECTIONS", "0")),
            source_verify_max_output_tokens=int(values.get("DEBUG_AGENT_SOURCE_VERIFY_MAX_OUTPUT_TOKENS", "6000")),
            source_verify_max_read_file_calls=int(values.get("DEBUG_AGENT_SOURCE_VERIFY_MAX_READ_FILE_CALLS", "6")),
            review_max_completion_tokens=int(values.get("DEBUG_AGENT_REVIEW_MAX_COMPLETION_TOKENS", "8000")),
            review_repair_max_completion_tokens=int(values.get("DEBUG_AGENT_REVIEW_REPAIR_MAX_COMPLETION_TOKENS", "4000")),
            review_reserve_tokens=int(values.get("DEBUG_AGENT_REVIEW_RESERVE_TOKENS", "24000")),
            review_reserve_calls=int(values.get("DEBUG_AGENT_REVIEW_RESERVE_CALLS", "2")),
            llm_cleanup_margin_seconds=int(values.get("DEBUG_AGENT_LLM_CLEANUP_MARGIN_SECONDS", "15")),
            obligation_review_min_seconds=int(values.get("DEBUG_AGENT_OBLIGATION_REVIEW_MIN_SECONDS", "120")),
            max_auto_obligation_presentations_per_reflection=int(values.get("DEBUG_AGENT_MAX_AUTO_OBLIGATION_PRESENTATIONS_PER_REFLECTION", "3")),
            focused_reflection_max_obligations=int(values.get("DEBUG_AGENT_FOCUSED_REFLECTION_MAX_OBLIGATIONS", "3")),
            focused_reflection_max_chars=int(values.get("DEBUG_AGENT_FOCUSED_REFLECTION_MAX_CHARS", "20000")),
            focused_reflection_timeout_seconds=float(values.get("DEBUG_AGENT_FOCUSED_REFLECTION_TIMEOUT_SECONDS", "60")),
            evidence_bundle_max_chars=int(values.get("DEBUG_AGENT_EVIDENCE_BUNDLE_MAX_CHARS", "16000")),
            parallel_max_actions=int(values.get("DEBUG_AGENT_PARALLEL_MAX_ACTIONS", "4")),
            parallel_group_timeout_seconds=float(values.get("DEBUG_AGENT_PARALLEL_GROUP_TIMEOUT_SECONDS", "20")),
            tool_retry_attempts=int(values.get("DEBUG_AGENT_TOOL_RETRY_ATTEMPTS", "2")),
            retry_base_delay_seconds=float(values.get("DEBUG_AGENT_RETRY_BASE_DELAY_SECONDS", "0.2")),
            retry_max_delay_seconds=float(values.get("DEBUG_AGENT_RETRY_MAX_DELAY_SECONDS", "2")),
            semantic_no_progress_limit=int(values.get("DEBUG_AGENT_SEMANTIC_NO_PROGRESS_LIMIT", "3")),
            obligation_presentation_max_chars=int(values.get("DEBUG_AGENT_OBLIGATION_PRESENTATION_MAX_CHARS", "8000")),
            provider_health_window=int(values.get("DEBUG_AGENT_PROVIDER_HEALTH_WINDOW", "5")),
            provider_failure_threshold=int(values.get("DEBUG_AGENT_PROVIDER_FAILURE_THRESHOLD", "3")),
            provider_consecutive_failures=int(values.get("DEBUG_AGENT_PROVIDER_CONSECUTIVE_FAILURES", "2")),
            provider_recovery_successes=int(values.get("DEBUG_AGENT_PROVIDER_RECOVERY_SUCCESSES", "2")),
            provider_degraded_timeout_seconds=int(values.get("DEBUG_AGENT_PROVIDER_DEGRADED_TIMEOUT_SECONDS", "60")),
            provider_retry_attempts=int(values.get("DEBUG_AGENT_RETRY_ATTEMPTS", "3")),
            provider_retry_min_budget_seconds=float(values.get("DEBUG_AGENT_RETRY_MIN_BUDGET_SECONDS", "5")),
            max_consecutive_planner_contract_failures=int(values.get("DEBUG_AGENT_MAX_CONSECUTIVE_PLANNER_CONTRACT_FAILURES", "2")),
            max_post_reject_tool_calls=int(values.get("DEBUG_AGENT_MAX_POST_REJECT_TOOL_CALLS", "3")),
            max_post_reject_steps=int(values.get("DEBUG_AGENT_MAX_POST_REJECT_STEPS", "4")),
            max_post_sufficiency_tool_calls=int(values.get("DEBUG_AGENT_MAX_POST_SUFFICIENCY_TOOL_CALLS", "4")),
            max_tool_contract_repairs=int(values.get("DEBUG_AGENT_MAX_TOOL_CONTRACT_REPAIRS", "1")),
            max_duplicate_actions=int(values.get("DEBUG_AGENT_MAX_REPEAT_ACTION", "2")),
            max_no_progress=int(values.get("DEBUG_AGENT_MAX_NO_PROGRESS_STEPS", "4")),
            reflect_every=int(values.get("DEBUG_AGENT_REFLECT_EVERY", "5")),
            max_consecutive_reflection_failures=int(values.get("DEBUG_AGENT_MAX_CONSECUTIVE_REFLECTION_FAILURES", "2")),
            max_tool_output_chars=int(values.get("DEBUG_AGENT_MAX_TOOL_OUTPUT_CHARS", "14000")),
            max_reporter_context_tokens=int(values.get("DEBUG_AGENT_MAX_REPORTER_CONTEXT_TOKENS", "12000")),
            max_evidence_per_file=int(values.get("DEBUG_AGENT_MAX_EVIDENCE_PER_FILE", "3")),
            max_snippet_lines=int(values.get("DEBUG_AGENT_MAX_SNIPPET_LINES", "120")),
            max_trace_summary_items=int(values.get("DEBUG_AGENT_MAX_TRACE_SUMMARY_ITEMS", "12")),
            recent_observation_count=int(values.get("DEBUG_AGENT_RECENT_OBSERVATION_COUNT", "2")),
            recent_observation_chars=int(values.get("DEBUG_AGENT_RECENT_OBSERVATION_CHARS", "16000")),
            trace_dir=values.get("DEBUG_AGENT_TRACE_DIR", ".debug_assistant/traces"),
            build_task_index=_as_bool(values.get("DEBUG_AGENT_BUILD_INDEX", "1")),
            keep_task_index=_as_bool(values.get("DEBUG_AGENT_KEEP_INDEX", "0")),
            enable_reflection=_as_bool(values.get("DEBUG_AGENT_ENABLE_REFLECTION", "1")),
            max_reflection_calls=int(values.get("DEBUG_AGENT_MAX_REFLECTION_CALLS", "2")),
            max_consecutive_reflection_calls=int(values.get("DEBUG_AGENT_MAX_CONSECUTIVE_REFLECTION_CALLS", "1")),
            enable_review=_as_bool(values.get("DEBUG_AGENT_ENABLE_REVIEW", "1")),
            max_review_recovery_cycles=int(values.get("DEBUG_AGENT_MAX_REVIEW_RECOVERY_CYCLES", "1")),
            tool_circuit_failure_threshold=int(values.get("DEBUG_AGENT_TOOL_CIRCUIT_FAILURE_THRESHOLD", "2")),
            planner_context_chars=int(values.get("DEBUG_AGENT_PLANNER_CONTEXT_CHARS", "12000")),
            tool_exposure_policy_version=values.get("DEBUG_AGENT_TOOL_EXPOSURE_POLICY_VERSION", "unified-visible-tools-v2-candidate"),
            domain_asset_manifest_path=values.get(
                "DEBUG_AGENT_DOMAIN_ASSET_MANIFEST",
                str(PROJECT_ROOT / "docs" / "refactor" / "DOMAIN_ASSET_MANIFEST.json"),
            ),
            context=ContextConfig(
                fallback_recent_count=int(values.get("DEBUG_AGENT_RECENT_OBSERVATION_COUNT", "2")),
                fallback_recent_chars=int(values.get("DEBUG_AGENT_RECENT_OBSERVATION_CHARS", "16000")),
            ),
            features=FeatureFlags(
                native_tool_calling=_as_bool(values.get("DEBUG_AGENT_NATIVE_TOOL_CALLING", "1")),
                structured_reflection=_as_bool(values.get("DEBUG_AGENT_STRUCTURED_REFLECTION", "1")),
            ),
            semantic_search=SemanticSearchConfig(
                enabled=_as_bool(values.get("DEBUG_AGENT_SEMANTIC_SEARCH", "0")),
                provider=values.get("DEBUG_AGENT_EMBEDDING_PROVIDER", "siliconflow"),
                base_url=values.get("DEBUG_AGENT_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
                api_key=values.get("DEBUG_AGENT_EMBEDDING_API_KEY", values.get("SILICONFLOW_API_KEY", "")),
                model=values.get("DEBUG_AGENT_EMBEDDING_MODEL", "BAAI/bge-m3"),
                dimension=int(values.get("DEBUG_AGENT_EMBEDDING_DIMENSION", "1024")),
                timeout=float(values.get("DEBUG_AGENT_EMBEDDING_TIMEOUT", "60")),
                batch_size=int(values.get("DEBUG_AGENT_EMBEDDING_BATCH_SIZE", "16")),
                max_retries=int(values.get("DEBUG_AGENT_EMBEDDING_MAX_RETRIES", "3")),
                lexical_top_k=int(values.get("DEBUG_AGENT_LEXICAL_TOP_K", "20")),
                semantic_top_k=int(values.get("DEBUG_AGENT_SEMANTIC_TOP_K", "20")),
                final_top_k=int(values.get("DEBUG_AGENT_SEARCH_FINAL_TOP_K", "10")),
                rrf_k=int(values.get("DEBUG_AGENT_RRF_K", "60")),
                cache_path=values.get("DEBUG_AGENT_EMBEDDING_CACHE", ".debug_assistant/embedding_cache.sqlite"),
                max_embedding_tokens=int(values.get("DEBUG_AGENT_EMBEDDING_MAX_TOKENS", "6000")),
                max_isolation_depth=int(values.get("DEBUG_AGENT_EMBEDDING_MAX_ISOLATION_DEPTH", "5")),
            ),
        )
        source_map = _source_map_for(values)
        return cls(model=model, runtime=runtime, source_map=source_map)


@dataclass(slots=True)
class EffectiveAppConfig(BaseAppConfig):
    protocol: ExperimentProtocol | None = None
    arm: ArmDelta | None = None
    run_spec: ExperimentRunSpec | None = None

    @property
    def transport_profile(self) -> str:
        profile = (self.protocol.transport_profile if self.protocol else None) or "semantic_plan"
        if profile != "semantic_plan":
            raise ValueError("current Planner transport requires semantic_plan; historical transport is offline-only")
        return profile

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "effective-config-snapshot-v4",
            "transport_profile": self.transport_profile,
            "model": {k: v for k, v in asdict(self.model).items() if k != "api_key"},
            "runtime": _redact(asdict(self.runtime)),
            "source_map": dict(sorted(self.source_map.items())),
            "protocol": asdict(self.protocol) if self.protocol else None,
            "arm": asdict(self.arm) if self.arm else None,
            "run_spec": asdict(self.run_spec) if self.run_spec else None,
        }
        payload["model"].pop("api_key", None)
        if isinstance(payload["runtime"].get("semantic_search"), dict):
            payload["runtime"]["semantic_search"].pop("api_key", None)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["effective_config_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload


# Public compatibility name. AppConfig now represents the effective typed
# boundary even when no Protocol/Arm overlay is loaded.
AppConfig = EffectiveAppConfig


_PROTOCOL_KEYS = {
    "model": "model", "temperature": "temperature",
    "provider_timeout_seconds": "provider_timeout_seconds",
    "provider_retry_attempts": "provider_retry_attempts",
    "provider_retry_min_budget_seconds": "provider_retry_min_budget_seconds",
    "planner_timeout_seconds": "planner_timeout_seconds",
    "review_timeout_seconds": "review_timeout_seconds",
    "max_steps": "max_steps", "max_tool_calls": "max_tool_calls",
    "max_llm_calls": "max_llm_calls", "max_total_tokens": "max_total_tokens",
    "planner_max_completion_tokens": "planner_max_completion_tokens",
    "review_reserve_tokens": "review_reserve_tokens",
    "review_reserve_calls": "review_reserve_calls",
    "terminal_reserve_tokens": "terminal_reserve_tokens",
    "terminal_reserve_llm_calls": "terminal_reserve_llm_calls",
    "max_review_rounds": "max_review_rounds",
    "tool_exposure_policy_version": "tool_exposure_policy_version",
    "transport_profile": "transport_profile",
    "public_fault_taxonomy": "public_fault_taxonomy",
}
_LEGACY_SHARED_METADATA = {
    "domain_skill_catalog", "domain_skill_catalog_sha256", "router_llm_calls",
    "tool_registry_and_schema_hash", "gold_isolation", "output_ontology",
    "output_ontology_source", "public_fault_taxonomy_sha256",
    "protocol_revision",
}
_ARM_METADATA = {"control_policy", "control_policy_sha256"}
_DEVELOPMENT_ARMS = {
    "A0": {
        "execution_mode": "baseline", "evidence_qualification": False,
        "context_projection_v2": False, "evidence_rehydration": False,
        "semantic_convergence_v2": False, "public_fault_taxonomy": True,
    },
    "A1": {
        "execution_mode": "dag", "evidence_qualification": True,
        "context_projection_v2": False, "evidence_rehydration": False,
        "semantic_convergence_v2": False, "public_fault_taxonomy": True,
    },
    "A2": {
        "execution_mode": "dag", "evidence_qualification": True,
        "context_projection_v2": True, "evidence_rehydration": True,
        "semantic_convergence_v2": True, "public_fault_taxonomy": True,
    },
}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _environment_values() -> dict[str, str]:
    """Read OS/.env values after python-dotenv has loaded the project file."""
    return {key: str(value).strip() for key, value in os.environ.items()}


def _source_map_for(values: Mapping[str, str]) -> dict[str, str]:
    env_file: dict[str, str] = {}
    try:
        from dotenv import dotenv_values
        env_file = {k: str(v) for k, v in dotenv_values(PROJECT_ROOT / ".env").items() if v is not None}
    except Exception:
        env_file = {}
    result: dict[str, str] = {}
    for env_name, value in values.items():
        if not env_name.startswith("DEBUG_AGENT_") and env_name not in {"SILICONFLOW_API_KEY"}:
            continue
        result[env_name] = ".env" if env_name in env_file and env_file[env_name] == value else "os"
    field_sources = {
        "model.provider": "DEBUG_AGENT_PROVIDER",
        "model.base_url": "DEBUG_AGENT_BASE_URL",
        "model.planner_model": "DEBUG_AGENT_MODEL",
        "model.reflection_model": (
            "DEBUG_AGENT_REFLECTION_MODEL"
            if values.get("DEBUG_AGENT_REFLECTION_MODEL")
            else "DEBUG_AGENT_REFLCETION_MODEL"
        ),
        "model.critic_model": "DEBUG_AGENT_CRITIC_MODEL",
        "model.temperature": "DEBUG_AGENT_TEMPERATURE",
        "model.timeout": "DEBUG_AGENT_TIMEOUT",
        "model.retry_attempts": "DEBUG_AGENT_RETRY_ATTEMPTS",
        "model.retry_min_budget_seconds": "DEBUG_AGENT_RETRY_MIN_BUDGET_SECONDS",
        "runtime.max_steps": "DEBUG_AGENT_MAX_STEPS",
        "runtime.max_tool_calls": "DEBUG_AGENT_MAX_TOOL_CALLS",
        "runtime.max_llm_calls": "DEBUG_AGENT_MAX_LLM_CALLS",
        "runtime.max_total_tokens": "DEBUG_AGENT_MAX_TOTAL_TOKENS",
        "runtime.max_request_input_tokens": "DEBUG_AGENT_MAX_REQUEST_INPUT_TOKENS",
        "model.context_window": "DEBUG_AGENT_CONTEXT_WINDOW",
        "model.protocol_safety_reserve_tokens": "DEBUG_AGENT_PROTOCOL_SAFETY_RESERVE_TOKENS",
        "runtime.review_reserve_tokens": "DEBUG_AGENT_REVIEW_RESERVE_TOKENS",
        "runtime.review_reserve_calls": "DEBUG_AGENT_REVIEW_RESERVE_CALLS",
        "runtime.terminal_reserve_tokens": "DEBUG_AGENT_TERMINAL_RESERVE_TOKENS",
        "runtime.terminal_reserve_llm_calls": "DEBUG_AGENT_TERMINAL_RESERVE_LLM_CALLS",
        "runtime.max_no_progress": "DEBUG_AGENT_MAX_NO_PROGRESS_STEPS",
        "runtime.provider_retry_attempts": "DEBUG_AGENT_RETRY_ATTEMPTS",
        "runtime.provider_retry_min_budget_seconds": "DEBUG_AGENT_RETRY_MIN_BUDGET_SECONDS",
    }
    for field_name, env_name in field_sources.items():
        if env_name in values:
            result[field_name] = (
                ".env" if env_name in env_file and env_file[env_name] == values[env_name] else "os"
            )
        else:
            result[field_name] = "default"
    return result


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<REDACTED>" if key in {"api_key", "credential", "secret", "token"}
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value]
    return value


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _reject_secret_fields(payload: Mapping[str, Any]) -> None:
    forbidden = {"api_key", "secret", "credential", "password", "private_token"}
    for key, value in payload.items():
        normalized = str(key).casefold()
        if normalized in forbidden or normalized.endswith("_secret") or normalized.endswith("_credential"):
            raise ValueError(f"experiment configuration cannot contain secret field: {key}")
        if isinstance(value, Mapping):
            _reject_secret_fields(value)


def _protocol_from_shared(shared: Mapping[str, Any], revision: str) -> ExperimentProtocol:
    _reject_secret_fields(shared)
    unknown = set(shared) - set(_PROTOCOL_KEYS) - _LEGACY_SHARED_METADATA
    if unknown:
        raise ValueError(f"unknown experiment shared fields: {sorted(unknown)}")
    values = {field: shared[key] for key, field in _PROTOCOL_KEYS.items() if key in shared}
    return ExperimentProtocol(revision=revision, **values)


def _arm_from_variant(arm_id: str, variant: Mapping[str, Any]) -> ArmDelta:
    allowed = {
        "execution_mode", "context_projection_mode", "evidence_qualification",
        "context_projection_v2", "evidence_rehydration", "semantic_convergence_v2",
        "public_fault_taxonomy",
    }
    unknown = set(variant) - allowed - _ARM_METADATA
    if unknown:
        raise ValueError(f"unknown ArmDelta fields: {sorted(unknown)}")
    _reject_secret_fields(variant)
    return ArmDelta(arm_id=arm_id, **{key: variant[key] for key in allowed if key in variant})


def _apply_overlay(base: BaseAppConfig, protocol: ExperimentProtocol, arm: ArmDelta | None,
                   run_spec: ExperimentRunSpec) -> EffectiveAppConfig:
    model_updates: dict[str, Any] = {}
    runtime_updates: dict[str, Any] = {}
    if protocol.model is not None:
        model_updates.update({
            "planner_model": protocol.model,
            "reflection_model": protocol.model,
            "critic_model": protocol.model,
        })
    if protocol.temperature is not None:
        model_updates["temperature"] = float(protocol.temperature)
    if protocol.provider_timeout_seconds is not None:
        model_updates["timeout"] = float(protocol.provider_timeout_seconds)
    if protocol.provider_retry_attempts is not None:
        model_updates["retry_attempts"] = int(protocol.provider_retry_attempts)
    if protocol.provider_retry_min_budget_seconds is not None:
        model_updates["retry_min_budget_seconds"] = float(protocol.provider_retry_min_budget_seconds)
    runtime_map = {
        "planner_timeout_seconds": "planner_llm_timeout_seconds",
        "review_timeout_seconds": "review_llm_timeout_seconds",
        "max_steps": "max_steps", "max_tool_calls": "max_tool_calls",
        "max_llm_calls": "max_llm_calls", "max_total_tokens": "max_total_tokens",
        "planner_max_completion_tokens": "planner_max_completion_tokens",
        "review_reserve_tokens": "review_reserve_tokens",
        "review_reserve_calls": "review_reserve_calls",
        "terminal_reserve_tokens": "terminal_reserve_tokens",
        "terminal_reserve_llm_calls": "terminal_reserve_llm_calls",
        "max_review_rounds": "max_review_rounds",
        "tool_exposure_policy_version": "tool_exposure_policy_version",
        "provider_retry_attempts": "provider_retry_attempts",
        "provider_retry_min_budget_seconds": "provider_retry_min_budget_seconds",
    }
    for protocol_field, runtime_field in runtime_map.items():
        value = getattr(protocol, protocol_field)
        if value is not None:
            runtime_updates[runtime_field] = value
    feature_updates: dict[str, Any] = {}
    if protocol.public_fault_taxonomy is not None:
        feature_updates["public_fault_taxonomy"] = bool(protocol.public_fault_taxonomy)
    if arm is not None:
        for field_name in (
            "execution_mode", "context_projection_mode", "max_steps", "max_tool_calls",
        ):
            value = getattr(arm, field_name, None)
            if value is not None and field_name in {"execution_mode", "context_projection_mode"}:
                runtime_updates[field_name] = value
        feature_updates.update({
            name: getattr(arm, name)
            for name in (
                "evidence_qualification", "context_projection_v2",
                "evidence_rehydration", "semantic_convergence_v2", "public_fault_taxonomy",
            ) if getattr(arm, name) is not None
        })
    if feature_updates:
        runtime_updates["features"] = replace(base.runtime.features, **feature_updates)
    model = replace(base.model, **model_updates)
    runtime = replace(base.runtime, **runtime_updates)
    runtime.features.validate()
    source_map = dict(base.source_map)
    for field_name in model_updates:
        source_map[f"model.{field_name}"] = "protocol"
    for field_name in runtime_updates:
        if field_name == "features":
            for feature_name in feature_updates:
                source_map[f"runtime.features.{feature_name}"] = (
                    "arm" if arm and getattr(arm, feature_name, None) is not None else "protocol"
                )
        else:
            source_map[f"runtime.{field_name}"] = "arm" if arm and hasattr(arm, field_name) else "protocol"
    return EffectiveAppConfig(
        model=model, runtime=runtime, source_map=source_map,
        protocol=protocol, arm=arm, run_spec=run_spec,
    )


def load_base_config() -> BaseAppConfig:
    return BaseAppConfig.from_env()


def load_effective_config(
    protocol_path: str | Path | None = None, *, arm_id: str | None = None,
    case_manifest_path: str | Path | None = None,
    base: BaseAppConfig | None = None,
    protocol_overrides: Mapping[str, Any] | None = None,
) -> EffectiveAppConfig:
    base = base or load_base_config()
    if protocol_path is None:
        if arm_id is None:
            return EffectiveAppConfig(
                model=base.model, runtime=base.runtime, source_map=dict(base.source_map),
            )
        if arm_id not in _DEVELOPMENT_ARMS:
            raise ValueError(f"unknown development arm: {arm_id}")
        protocol = ExperimentProtocol(revision="development")
        arm = _arm_from_variant(arm_id, _DEVELOPMENT_ARMS[arm_id])
        run_spec = ExperimentRunSpec(
            protocol_path="", bundle_path=None, arm_id=arm_id,
            case_manifest_path=(
                _relative_to_project(Path(case_manifest_path)) if case_manifest_path else None
            ), protocol_revision="development",
        )
        return _apply_overlay(base, protocol, arm, run_spec)
    path = Path(protocol_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "shared" in payload:
        bundle_payload = payload
        bundle_path = path
        protocol_payload = payload
    else:
        bundle_ref = str(payload.get("arm_bundle", ""))
        if not bundle_ref:
            raise ValueError("ExperimentProtocol must reference an ArmBundle")
        bundle_path = (path.parent / bundle_ref).resolve() if not bundle_ref.startswith("experiments/") else (PROJECT_ROOT / bundle_ref).resolve()
        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        protocol_payload = payload
    revision = str(protocol_payload.get("protocol_revision") or bundle_payload.get("shared", {}).get("protocol_revision") or "unversioned")
    shared = dict(bundle_payload.get("shared", {}))
    if protocol_overrides:
        unknown_overrides = set(protocol_overrides) - set(_PROTOCOL_KEYS)
        if unknown_overrides:
            raise ValueError(f"unknown protocol override fields: {sorted(unknown_overrides)}")
        _reject_secret_fields(protocol_overrides)
        shared.update(protocol_overrides)
    protocol = _protocol_from_shared(shared, revision)
    variants = bundle_payload.get("variants") or {}
    arm = None
    if arm_id is not None:
        if arm_id not in variants:
            raise ValueError(f"unknown experiment arm: {arm_id}")
        arm = _arm_from_variant(arm_id, variants[arm_id])
    run_spec = ExperimentRunSpec(
        protocol_path=_relative_to_project(path),
        bundle_path=_relative_to_project(bundle_path),
        arm_id=arm_id,
        case_manifest_path=(
            _relative_to_project(Path(case_manifest_path)) if case_manifest_path else None
        ),
        protocol_revision=revision,
    )
    return _apply_overlay(base, protocol, arm, run_spec)
