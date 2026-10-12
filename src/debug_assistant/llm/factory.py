from .mock import MockLLMClient
from .openai_compatible import OpenAICompatibleClient
from .base import ModelCapability, ProviderCapabilities

def build_llm(cfg):
    capability = ModelCapability(
        provider=cfg.provider, model=cfg.planner_model,
        context_window=getattr(cfg, "context_window", None),
        max_output_tokens=getattr(cfg, "max_output_tokens", None),
        tokenizer=getattr(cfg, "tokenizer", "char4"),
        reserved_output_tokens=getattr(cfg, "reserved_output_tokens", 0),
        protocol_safety_reserve_tokens=getattr(cfg, "protocol_safety_reserve_tokens", 0),
    )
    if cfg.provider == "mock": return MockLLMClient(model_capability=capability)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleClient(
            cfg.base_url, cfg.api_key, cfg.planner_model, cfg.timeout, cfg.temperature,
            max_attempts=getattr(cfg, "retry_attempts", 3),
            min_retry_budget=getattr(cfg, "retry_min_budget_seconds", 5.0),
            model_capability=capability,
            # The incident interface requires explicit strict response schemas.
            # Unsupported endpoint/model combinations must return a typed failure.
            capabilities=ProviderCapabilities(json_object=True, json_schema=True,
                tool_calling=True, parallel_tool_calls=True,
                provider=cfg.provider, model=cfg.planner_model),
        )
    raise ValueError(f"Unsupported provider: {cfg.provider}. Use openai_compatible or mock.")
