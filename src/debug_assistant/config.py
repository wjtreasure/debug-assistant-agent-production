from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


# 项目根目录：
# debug-assistant-agent-production/
#
# 当前文件：
# src/debug_assistant/config.py
#
# parents[2] 就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 主动加载项目根目录下的 .env
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(slots=True)
class ModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    planner_model: str = "gpt-5.6"
    critic_model: str = ""
    timeout: float = 60.0
    temperature: float = 0.1


@dataclass(slots=True)
class HarnessConfig:
    max_steps: int = 20
    max_tool_calls: int = 45
    max_context_chars: int = 50_000
    max_repeat_action: int = 2
    max_no_progress_steps: int = 4
    reflect_every: int = 5
    max_tool_output_chars: int = 14_000
    trace_dir: str = ".debug_assistant/traces"
    build_task_index: bool = True
    keep_task_index: bool = False


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        get = os.getenv

        return cls(
            model=ModelConfig(
                provider=get(
                    "DEBUG_AGENT_PROVIDER",
                    "openai_compatible",
                ),
                base_url=get(
                    "DEBUG_AGENT_BASE_URL",
                    "https://api.openai.com/v1",
                ).rstrip("/"),
                api_key=get(
                    "DEBUG_AGENT_API_KEY",
                    "",
                ),
                planner_model=get(
                    "DEBUG_AGENT_MODEL",
                    "gpt-5.6",
                ),
                critic_model=get(
                    "DEBUG_AGENT_CRITIC_MODEL",
                    "",
                ),
                timeout=float(
                    get("DEBUG_AGENT_TIMEOUT", "60")
                ),
                temperature=float(
                    get("DEBUG_AGENT_TEMPERATURE", "0.1")
                ),
            ),
            harness=HarnessConfig(
                max_steps=int(
                    get("DEBUG_AGENT_MAX_STEPS", "20")
                ),
                max_tool_calls=int(
                    get("DEBUG_AGENT_MAX_TOOL_CALLS", "45")
                ),
                max_context_chars=int(
                    get("DEBUG_AGENT_MAX_CONTEXT_CHARS", "50000")
                ),
                max_repeat_action=int(
                    get("DEBUG_AGENT_MAX_REPEAT_ACTION", "2")
                ),
                max_no_progress_steps=int(
                    get("DEBUG_AGENT_MAX_NO_PROGRESS_STEPS", "4")
                ),
                reflect_every=int(
                    get("DEBUG_AGENT_REFLECT_EVERY", "5")
                ),
                max_tool_output_chars=int(
                    get("DEBUG_AGENT_MAX_TOOL_OUTPUT_CHARS", "14000")
                ),
                trace_dir=get(
                    "DEBUG_AGENT_TRACE_DIR",
                    ".debug_assistant/traces",
                ),
                build_task_index=get(
                    "DEBUG_AGENT_BUILD_INDEX",
                    "1",
                ).lower() not in ("0", "false", "no"),
                keep_task_index=get(
                    "DEBUG_AGENT_KEEP_INDEX",
                    "0",
                ).lower() in ("1", "true", "yes"),
            ),
        )