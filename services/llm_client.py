"""Thin wrapper around the Anthropic SDK that forces schema-validated JSON.

Every agent calls the LLM through this single service instead of touching
the Anthropic SDK directly, so retry behaviour, schema enforcement, and
logging live in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Protocol, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_TOOL_NAME = "submit_structured_response"
_MAX_ATTEMPTS = 4


class LLMClientError(RuntimeError):
    """Raised when the LLM cannot produce a schema-valid response."""


def _validate_structured(response_model: type[ModelT], data: object) -> ModelT:
    """Validate ``data`` against ``response_model``, tolerating one extra
    layer of wrapping.

    Despite a flat tool schema, Claude occasionally nests the actual
    payload under a single extra top-level key (observed in practice as
    ``{"response": {...}}``, but the wrapper key name isn't guaranteed) —
    likely triggered by wording like "submit a response" in the tool
    description. Rather than failing outright, retry against the inner
    object when the outer payload is exactly one key deep.
    """
    try:
        return response_model.model_validate(data)
    except ValidationError:
        if isinstance(data, dict) and len(data) == 1:
            (inner,) = data.values()
            if isinstance(inner, dict):
                return response_model.model_validate(inner)
        raise


def _build_correction_message(exc: ValidationError) -> str:
    """Feedback sent back to the model after a schema-validation failure.

    Observed failure modes include list fields returned as strings (or as
    XML-ish tag text), the whole payload nested under one extra key, and
    literal tool-calling placeholder tokens leaking in as field names.
    Rather than write a bespoke parser for every variant, name the concrete
    error and ask the model to self-correct in the next turn — the standard
    pattern for tool-call validation retries, and far more reliable than
    blindly resampling with the identical prompt.
    """
    return (
        "Your last tool call did not match the required schema and was "
        f"rejected:\n{exc}\n\n"
        "Call the tool again with a single, flat JSON object matching the "
        "schema exactly: every list-typed field must be a real JSON array "
        "(never a string, and never XML-style tags like <item>), the "
        "object must not be wrapped under an extra top-level key, and it "
        "must not contain any placeholder names/values from the tool-call "
        "format itself as fields."
    )


class LLMClient(Protocol):
    """Interface agents depend on. Lets tests substitute a fake client."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_tokens: int = 2048,
    ) -> ModelT: ...


class AnthropicLLMClient:
    """Production LLM client backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.anthropic_api_key:
            raise LLMClientError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self._client = Anthropic(api_key=self._settings.anthropic_api_key)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_tokens: int = 2048,
    ) -> ModelT:
        schema = response_model.model_json_schema()
        tool = {
            "name": _TOOL_NAME,
            "description": f"Submit a response conforming to the {response_model.__name__} schema.",
            "input_schema": schema,
        }

        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            logger.debug("LLM call attempt %d/%d for %s", attempt, _MAX_ATTEMPTS, response_model.__name__)
            message = self._client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
            tool_use = next(
                (block for block in message.content if block.type == "tool_use"), None
            )
            if tool_use is None:
                last_error = LLMClientError("LLM response did not include a tool_use block.")
                continue
            try:
                return _validate_structured(response_model, tool_use.input)
            except ValidationError as exc:
                logger.warning("Schema validation failed on attempt %d: %s", attempt, exc)
                last_error = exc
                if attempt < _MAX_ATTEMPTS:
                    # Feed the malformed call and a tool_result explaining
                    # exactly what was wrong back to the model, so the next
                    # attempt is a genuine correction rather than a blind
                    # re-roll of the same prompt.
                    messages.append({"role": "assistant", "content": message.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": _build_correction_message(exc),
                                    "is_error": True,
                                }
                            ],
                        }
                    )

        raise LLMClientError(
            f"Failed to get a schema-valid {response_model.__name__} after {_MAX_ATTEMPTS} attempts."
        ) from last_error
