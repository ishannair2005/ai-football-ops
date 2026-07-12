"""Application-wide settings loaded from environment variables / .env.

This module holds only cross-cutting runtime configuration (credentials,
model selection, logging). Club-specific configuration lives in
``config.club_config`` and is loaded separately so that switching clubs
never requires touching source code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    active_club: str = "manchester_united"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
