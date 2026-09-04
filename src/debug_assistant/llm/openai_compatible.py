from __future__ import annotations

from typing import Any, Callable
import asyncio
import time
import uuid
import random

import httpx

from .base import (
    LLMClient,
    LLMResponse,
    LLMClientUsageError,
    LLMDeadlineExceeded,
    LLMError,
    LLMInvalidJSON,
    LLMOutputError,
    LLMTransportTimeout,
    extract_json,
    parse_tool_calls,
    ProviderCapabilities,
)


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible JSON client with a total logical-call deadline.

    The Harness remains synchronous via :meth:`complete_json`; HTTP transport is async so
    ``asyncio.timeout`` can cancel an in-flight request at the logical deadline. All retry
    attempts share the same deadline rather than receiving independent timeout budgets.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
        *,
        async_transport: httpx.AsyncBaseTransport | None = None,
        max_attempts: int = 3,
        min_retry_budget: float = 5.0,
        capabilities: ProviderCapabilities | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = max(0.01, float(timeout))
        self.temperature = temperature
        self.async_transport = async_transport
        self.max_attempts = max(1, int(max_attempts))
        self.min_retry_budget = max(0.0, float(min_retry_budget))
        self.capabilities = capabilities or ProviderCapabilities(
            json_object=True, json_schema=False, tool_calling=True, parallel_tool_calls=True
        )
        self.calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.last_raw_content = None
        self.last_usage: dict[str, Any] = {}
        self.last_response: LLMResponse | None = None

    def _event(self, event_type: str, **payload) -> None:
        self.events.append({"type": event_type, "payload": payload})

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    async def acomplete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        logical_timeout_seconds: float | None = None,
        on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        return_response: bool = False,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise LLMError("DEBUG_AGENT_API_KEY is empty. Configure it or use DEBUG_AGENT_PROVIDER=mock for smoke tests.")

        logical_budget = float(logical_timeout_seconds) if logical_timeout_seconds is not None else self.timeout * self.max_attempts
        if logical_budget <= 0:
            raise LLMDeadlineExceeded("LLM logical deadline exhausted before request start")

        call_id = uuid.uuid4().hex[:12]
        call_started = time.monotonic()
        deadline = call_started + logical_budget
        base_payload = {
            "model": model or self.default_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        if tools:
            base_payload["tools"] = tools
            base_payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self._event(
            "LLM_LOGICAL_CALL_STARTED",
            logical_call_id=call_id,
            logical_timeout_seconds=logical_budget,
            model=base_payload["model"],
        )

        last: Exception | None = None
        last_provider_failure = False
        attempts_made = 0
        client_timeout = httpx.Timeout(self.timeout)
        async with httpx.AsyncClient(timeout=client_timeout, transport=self.async_transport) as client:
            for attempt in range(1, self.max_attempts + 1):
                attempts_made = attempt
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    elapsed_ms = (time.monotonic() - call_started) * 1000
                    self._event(
                        "LLM_DEADLINE_EXCEEDED",
                        logical_call_id=call_id, attempt_index=attempt,
                        logical_elapsed_ms=elapsed_ms, remaining_seconds=0.0,
                    )
                    self._event(
                        "LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                        provider_attempts=max(0,attempts_made-1), logical_elapsed_ms=elapsed_ms,
                        success=False, error_type="logical_deadline", provider_success=False,
                        provider_failure=True, logical_timeout_seconds=logical_budget,
                    )
                    raise LLMDeadlineExceeded(f"LLM logical deadline exceeded after {elapsed_ms/1000:.2f}s") from last

                payload = dict(base_payload)
                retry_after_seconds=None
                if response_format is not None:
                    payload["response_format"] = response_format
                elif attempt == 1 and not tools and self.capabilities.json_object:
                    payload["response_format"] = {"type": "json_object"}
                attempt_started = time.monotonic()
                provider_response_success = False
                attempt_timeout = max(0.001, min(self.timeout, remaining))
                attempt_meta={
                    "logical_call_id":call_id, "attempt_index":attempt,
                    "attempt_timeout_seconds":attempt_timeout,
                    "logical_remaining_seconds":remaining,
                }
                self._event("LLM_ATTEMPT_STARTED", **attempt_meta)
                if on_attempt_started is not None:
                    on_attempt_started(dict(attempt_meta))
                try:
                    async with asyncio.timeout(remaining):
                        r = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=httpx.Timeout(attempt_timeout),
                        )
                    attempt_elapsed_ms = (time.monotonic() - attempt_started) * 1000

                    if r.status_code in (429, 500, 502, 503, 504):
                        last_provider_failure = True
                        try:
                            retry_after_seconds=max(0.0,float(r.headers.get('Retry-After'))) if r.headers.get('Retry-After') else None
                        except (TypeError,ValueError):
                            retry_after_seconds=None
                        last = LLMError(f"transient LLM HTTP {r.status_code}: {r.text[:300]}")
                        self._event(
                            "LLM_ATTEMPT_FAILED",
                            logical_call_id=call_id, attempt_index=attempt,
                            attempt_elapsed_ms=attempt_elapsed_ms,
                            error_type="transient_http", status_code=r.status_code,
                            retryable=True,
                        )
                    elif attempt == 1 and r.status_code in (400, 422):
                        last_provider_failure = False
                        # JSON response mode is optional for compatible providers. One retry
                        # without response_format remains bounded by the same logical deadline.
                        last = LLMError(f"JSON mode unsupported: HTTP {r.status_code}")
                        if response_format is not None:
                            response_format = {"type": "json_object"} if self.capabilities.json_object else None
                        self._event(
                            "LLM_ATTEMPT_FAILED",
                            logical_call_id=call_id, attempt_index=attempt,
                            attempt_elapsed_ms=attempt_elapsed_ms,
                            error_type="json_mode_unsupported", status_code=r.status_code,
                            retryable=True,
                        )
                    else:
                        r.raise_for_status()
                        provider_response_success = True
                        try:
                            data = r.json()
                            usage = data.get("usage") or {}
                            message = data["choices"][0]["message"]
                        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
                            error = LLMOutputError("Provider returned a malformed chat completion")
                            error.error_type = "provider_contract_mismatch"
                            raise error from exc
                        if not isinstance(message, dict):
                            error = LLMOutputError("Provider returned a malformed chat message")
                            error.error_type = "provider_contract_mismatch"
                            raise error
                        raw_content = message.get("content")
                        if raw_content is not None and not isinstance(raw_content, str):
                            error = LLMOutputError("Provider message content must be a string or null")
                            error.error_type = "provider_contract_mismatch"
                            raise error
                        content = raw_content or ""
                        tool_calls = parse_tool_calls(message)
                        self.last_raw_content = content
                        self.last_usage = dict(usage)
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        completion_details = usage.get("completion_tokens_details") or {}
                        prompt_tokens = usage.get("prompt_tokens", 0) or 0
                        completion_tokens = usage.get("completion_tokens", 0) or 0
                        total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                        logical_elapsed_ms = (time.monotonic() - call_started) * 1000
                        self.calls.append({
                            "model": payload["model"],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "input_tokens": prompt_tokens,
                            "output_tokens": completion_tokens,
                            "cached_tokens": prompt_details.get("cached_tokens"),
                            "reasoning_tokens": completion_details.get("reasoning_tokens"),
                            "prompt_chars": len(system) + len(user),
                            "completion_chars": len(content or ""),
                            "latency_ms": logical_elapsed_ms,
                            "provider_attempts": attempt,
                            "logical_call_id": call_id,
                        })
                        # Native tool responses are not assistant JSON responses. The
                        # provider may include ordinary prose alongside valid calls;
                        # only the legacy complete_json contract decodes content here.
                        parsed = extract_json(content) if content.strip() and tools is None else None
                        self.last_response = LLMResponse(
                            content=content, structured=parsed, tool_calls=tool_calls,
                            usage=dict(usage),
                        )
                        self._event(
                            "LLM_ATTEMPT_FINISHED",
                            logical_call_id=call_id, attempt_index=attempt,
                            attempt_elapsed_ms=attempt_elapsed_ms, success=True,
                        )
                        self._event(
                            "LLM_LOGICAL_CALL_FINISHED",
                            logical_call_id=call_id, provider_attempts=attempt,
                            logical_elapsed_ms=logical_elapsed_ms, success=True,
                            provider_success=True, provider_failure=False,
                            logical_timeout_seconds=logical_budget,
                        )
                        return self.last_response if return_response else (parsed or {})

                except asyncio.TimeoutError as exc:
                    last = LLMDeadlineExceeded("LLM logical deadline exceeded while request was in flight")
                    elapsed_ms = (time.monotonic() - call_started) * 1000
                    self._event(
                        "LLM_ATTEMPT_FAILED",
                        logical_call_id=call_id, attempt_index=attempt,
                        attempt_elapsed_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type="logical_deadline", retryable=False,
                    )
                    self._event(
                        "LLM_DEADLINE_EXCEEDED",
                        logical_call_id=call_id, attempt_index=attempt,
                        logical_elapsed_ms=elapsed_ms, remaining_seconds=0.0,
                    )
                    self._event(
                        "LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                        provider_attempts=attempts_made, logical_elapsed_ms=elapsed_ms,
                        success=False, error_type="logical_deadline", provider_success=False,
                        provider_failure=True, logical_timeout_seconds=logical_budget,
                    )
                    raise LLMDeadlineExceeded(f"LLM logical deadline exceeded after {elapsed_ms/1000:.2f}s") from exc
                except httpx.TimeoutException as exc:
                    last_provider_failure = True
                    last = LLMTransportTimeout(str(exc) or type(exc).__name__)
                    self._event(
                        "LLM_ATTEMPT_FAILED",
                        logical_call_id=call_id, attempt_index=attempt,
                        attempt_elapsed_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type="transport_timeout", retryable=True,
                    )
                except httpx.NetworkError as exc:
                    last_provider_failure = True
                    last = exc
                    self._event(
                        "LLM_ATTEMPT_FAILED",
                        logical_call_id=call_id, attempt_index=attempt,
                        attempt_elapsed_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type="network_error", retryable=True,
                    )
                except httpx.HTTPStatusError as exc:
                    self._event(
                        "LLM_ATTEMPT_FAILED",
                        logical_call_id=call_id, attempt_index=attempt,
                        attempt_elapsed_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type="http_error", status_code=exc.response.status_code,
                        retryable=False,
                    )
                    elapsed_ms=(time.monotonic()-call_started)*1000
                    provider_failure=bool(exc.response.status_code==429 or exc.response.status_code>=500)
                    self._event(
                        "LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                        provider_attempts=attempts_made, logical_elapsed_ms=elapsed_ms,
                        success=False, error_type="http_error", provider_success=not provider_failure,
                        provider_failure=provider_failure, logical_timeout_seconds=logical_budget,
                    )
                    raise LLMError(f"LLM request failed: {exc}") from exc
                except LLMInvalidJSON:
                    self._event("LLM_ATTEMPT_FINISHED", logical_call_id=call_id,
                                attempt_index=attempt,
                                attempt_elapsed_ms=(time.monotonic()-attempt_started)*1000,
                                success=False)
                    self._event("LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                                provider_attempts=attempt, logical_elapsed_ms=(time.monotonic()-call_started)*1000,
                                success=False, error_type="invalid_json", provider_success=True,
                                provider_failure=False, logical_timeout_seconds=logical_budget)
                    raise
                except LLMOutputError:
                    self._event("LLM_ATTEMPT_FINISHED", logical_call_id=call_id,
                                attempt_index=attempt,
                                attempt_elapsed_ms=(time.monotonic()-attempt_started)*1000,
                                success=False)
                    self._event("LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                                provider_attempts=attempt, logical_elapsed_ms=(time.monotonic()-call_started)*1000,
                                success=False, error_type="provider_payload_invalid", provider_success=True,
                                provider_failure=False, logical_timeout_seconds=logical_budget)
                    raise
                except (LLMDeadlineExceeded, LLMError):
                    raise
                except Exception as exc:
                    if provider_response_success:
                        self._event("LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                                    provider_attempts=attempt, logical_elapsed_ms=(time.monotonic()-call_started)*1000,
                                    success=False, error_type="provider_payload_invalid", provider_success=True,
                                    provider_failure=False, logical_timeout_seconds=logical_budget)
                    self._event(
                        "LLM_ATTEMPT_FAILED",
                        logical_call_id=call_id, attempt_index=attempt,
                        attempt_elapsed_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type=type(exc).__name__, retryable=False,
                    )
                    raise LLMError(f"LLM request failed: {exc}") from exc

                remaining = self._remaining(deadline)
                if attempt >= self.max_attempts or remaining <= self.min_retry_budget:
                    if remaining <= 0:
                        elapsed_ms = (time.monotonic() - call_started) * 1000
                        self._event(
                            "LLM_DEADLINE_EXCEEDED",
                            logical_call_id=call_id, attempt_index=attempt,
                            logical_elapsed_ms=elapsed_ms, remaining_seconds=remaining,
                        )
                        self._event(
                            "LLM_LOGICAL_CALL_FINISHED", logical_call_id=call_id,
                            provider_attempts=attempts_made, logical_elapsed_ms=elapsed_ms,
                            success=False, error_type="logical_deadline", provider_success=False,
                            provider_failure=True, logical_timeout_seconds=logical_budget,
                        )
                        raise LLMDeadlineExceeded(f"LLM logical deadline exceeded after {elapsed_ms/1000:.2f}s") from last
                    break

                max_backoff=min(2.0,0.4 * (2 ** (attempt - 1)),max(0.0,remaining-self.min_retry_budget))
                if retry_after_seconds is not None:
                    backoff=min(retry_after_seconds,max(0.0,remaining-self.min_retry_budget))
                elif isinstance(last,LLMError) and str(last).startswith('JSON mode unsupported'):
                    backoff=0.0
                else:
                    backoff=random.uniform(0.0,max_backoff) if max_backoff>0 else 0.0
                if backoff < 0:
                    break
                self._event(
                    "LLM_ATTEMPT_RETRYING",
                    logical_call_id=call_id, attempt_index=attempt,
                    backoff_seconds=backoff, logical_remaining_seconds=remaining,
                )
                await asyncio.sleep(backoff)

        logical_elapsed_ms = (time.monotonic() - call_started) * 1000
        self._event(
            "LLM_LOGICAL_CALL_FINISHED",
            logical_call_id=call_id, provider_attempts=attempts_made,
            logical_elapsed_ms=logical_elapsed_ms, success=False,
            error_type=type(last).__name__ if last else "unknown",
            provider_success=not last_provider_failure, provider_failure=last_provider_failure, logical_timeout_seconds=logical_budget,
        )
        if isinstance(last, LLMTransportTimeout):
            raise LLMTransportTimeout(f"LLM request failed after bounded retries: {last}") from last
        raise LLMError(f"LLM request failed after bounded retries: {last}") from last

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        logical_timeout_seconds: float | None = None,
        on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acomplete_json(
                    system, user, model=model,
                    logical_timeout_seconds=logical_timeout_seconds,
                    on_attempt_started=on_attempt_started,
                )
            )
        raise LLMClientUsageError(
            "Synchronous complete_json() cannot be called from a running event loop; "
            "use 'await acomplete_json(...)' instead."
        )

    def complete_with_tools(
        self, system: str, user: str, *, tools: list[dict[str, Any]], model: str | None = None,
        logical_timeout_seconds: float | None = None,
        on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse:
        if not self.capabilities.tool_calling:
            return super().complete_with_tools(
                system, user, tools=tools, model=model,
                logical_timeout_seconds=logical_timeout_seconds,
                on_attempt_started=on_attempt_started,
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acomplete_json(
                system, user, model=model, logical_timeout_seconds=logical_timeout_seconds,
                on_attempt_started=on_attempt_started, tools=tools, return_response=True,
            ))
        raise LLMClientUsageError(
            "Synchronous complete_with_tools() cannot be called from a running event loop"
        )

    def complete_structured(
        self, system: str, user: str, *, schema=None, model: str | None = None,
        logical_timeout_seconds: float | None = None,
        on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse:
        response_format = None
        if schema is not None and self.capabilities.json_schema:
            schema_data = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            response_format = {"type": "json_schema", "json_schema": {
                "name": getattr(schema, "__name__", "response"), "strict": True, "schema": schema_data,
            }}
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acomplete_json(
                system, user, model=model, logical_timeout_seconds=logical_timeout_seconds,
                on_attempt_started=on_attempt_started, response_format=response_format,
                return_response=True,
            ))
        raise LLMClientUsageError(
            "Synchronous complete_structured() cannot be called from a running event loop"
        )
