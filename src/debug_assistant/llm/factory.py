from .mock import MockLLMClient
from .openai_compatible import OpenAICompatibleClient

def build_llm(cfg):
    if cfg.provider == "mock": return MockLLMClient()
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleClient(cfg.base_url, cfg.api_key, cfg.planner_model, cfg.timeout)
    raise ValueError(f"Unsupported provider: {cfg.provider}. Use openai_compatible or mock.")
