from __future__ import annotations
from typing import Any
import time
import httpx
from .base import LLMClient, LLMError, extract_json

class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, default_model: str, timeout: float=60.0, temperature: float=0.0):
        self.base_url=base_url.rstrip('/')
        self.api_key=api_key
        self.default_model=default_model
        self.timeout=timeout
        self.temperature=temperature
        self.calls=[]
        self.last_raw_content=None

    def complete_json(self, system: str, user: str, *, model: str | None=None) -> dict[str, Any]:
        if not self.api_key:
            raise LLMError("DEBUG_AGENT_API_KEY is empty. Configure it or use DEBUG_AGENT_PROVIDER=mock for smoke tests.")
        base_payload={
            "model": model or self.default_model,
            "messages":[{"role":"system","content":system},{"role":"user","content":user}],
            "temperature":self.temperature,
        }
        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}
        last=None
        for attempt in range(3):
            payload=dict(base_payload)
            if attempt == 0:
                payload["response_format"]={"type":"json_object"}
            started=time.time()
            try:
                r=httpx.post(f"{self.base_url}/chat/completions",headers=headers,json=payload,timeout=self.timeout)
                if r.status_code in (429,500,502,503,504):
                    last=LLMError(f"transient LLM HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(0.4*(2**attempt)); continue
                if attempt == 0 and r.status_code in (400,422):
                    last=LLMError(f"JSON mode unsupported: HTTP {r.status_code}")
                    continue
                r.raise_for_status(); data=r.json(); usage=data.get("usage") or {}
                content=data["choices"][0]["message"]["content"]
                self.last_raw_content=content
                prompt_details=usage.get("prompt_tokens_details") or {}
                completion_details=usage.get("completion_tokens_details") or {}
                prompt_tokens=usage.get("prompt_tokens",0) or 0
                completion_tokens=usage.get("completion_tokens",0) or 0
                total_tokens=usage.get("total_tokens",0) or (prompt_tokens+completion_tokens)
                self.calls.append({
                    "model":payload["model"],
                    "prompt_tokens":prompt_tokens,
                    "completion_tokens":completion_tokens,
                    "total_tokens":total_tokens,
                    # Backward-compatible aliases.
                    "input_tokens":prompt_tokens,
                    "output_tokens":completion_tokens,
                    "cached_tokens":prompt_details.get("cached_tokens"),
                    "reasoning_tokens":completion_details.get("reasoning_tokens"),
                    "prompt_chars":len(system)+len(user),
                    "completion_chars":len(content or ""),
                    "latency_ms":(time.time()-started)*1000,
                })
                return extract_json(content)
            except (httpx.TimeoutException,httpx.NetworkError) as exc:
                last=exc; time.sleep(0.4*(2**attempt))
            except Exception as exc:
                raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError(f"LLM request failed after retries: {last}")
