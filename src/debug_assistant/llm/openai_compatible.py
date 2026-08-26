from __future__ import annotations
from typing import Any
import time
import httpx
from .base import LLMClient, LLMError, extract_json

class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, default_model: str, timeout: float=60.0):
        self.base_url=base_url.rstrip('/')
        self.api_key=api_key
        self.default_model=default_model
        self.timeout=timeout
        self.calls=[]

    def complete_json(self, system: str, user: str, *, model: str | None=None) -> dict[str, Any]:
        if not self.api_key:
            raise LLMError("DEBUG_AGENT_API_KEY is empty. Configure it or use DEBUG_AGENT_PROVIDER=mock for smoke tests.")
        base_payload={
            "model": model or self.default_model,
            "messages":[{"role":"system","content":system},{"role":"user","content":user}],
            "temperature":0.1,
        }
        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}
        last=None
        for attempt in range(3):
            payload=dict(base_payload)
            # First try structured JSON mode. Gateways that do not support it get a compatibility retry.
            if attempt == 0: payload["response_format"]={"type":"json_object"}
            try:
                r=httpx.post(f"{self.base_url}/chat/completions",headers=headers,json=payload,timeout=self.timeout)
                if r.status_code in (429,500,502,503,504):
                    last=LLMError(f"transient LLM HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(0.4*(2**attempt)); continue
                # Some OpenAI-compatible gateways reject response_format with 400/422.
                if attempt == 0 and r.status_code in (400,422):
                    last=LLMError(f"JSON mode unsupported: HTTP {r.status_code}")
                    continue
                r.raise_for_status(); data=r.json(); usage=data.get("usage") or {}
                self.calls.append({"model":payload["model"],"input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0),"total_tokens":usage.get("total_tokens",0)})
                return extract_json(data["choices"][0]["message"]["content"])
            except (httpx.TimeoutException,httpx.NetworkError) as exc:
                last=exc; time.sleep(0.4*(2**attempt))
            except Exception as exc:
                raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError(f"LLM request failed after retries: {last}")
