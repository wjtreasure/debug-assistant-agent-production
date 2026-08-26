from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import json, re

class LLMError(RuntimeError): pass

class LLMClient(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str, *, model: str | None = None) -> dict[str, Any]: ...


def extract_json(text: str) -> dict[str, Any]:
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*', '', text)
        text=re.sub(r'\s*```$', '', text)
    try: return json.loads(text)
    except json.JSONDecodeError:
        start=text.find('{'); end=text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise LLMError(f"Model did not return valid JSON: {text[:500]}")
