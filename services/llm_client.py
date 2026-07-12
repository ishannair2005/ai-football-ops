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
_MAX_ATTEMPTS = 2


class LLMClientError(RuntimeError):
    """Raised when the LLM cannot produce a schema-valid response."""


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

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            logger.debug("LLM call attempt %d/%d for %s", attempt, _MAX_ATTEMPTS, response_model.__name__)
            message = self._client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
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
                return response_model.model_validate(tool_use.input)
            except ValidationError as exc:
                logger.warning("Schema validation failed on attempt %d: %s", attempt, exc)
                last_error = exc

        raise LLMClientError(
            f"Failed to get a schema-valid {response_model.__name__} after {_MAX_ATTEMPTS} attempts."
        ) from last_error
