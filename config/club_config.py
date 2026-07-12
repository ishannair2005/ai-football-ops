"""Club configuration model and loader.

Every piece of club-specific knowledge (identity, competition, manager,
tactical baseline, finances) lives in a YAML file under ``data/clubs/``.
Agents and services must read club context through :class:`ClubConfig`
rather than assuming Manchester United — this is what lets the platform
support any club without source changes.
"""

from __future__ import annotations

from functools import lru_cache

import yaml
from pydantic import BaseModel, Field

from config.settings import DATA_DIR


class ClubConfig(BaseModel):
    club_id: str
    name: str
    short_name: str
    league: str
    country: str
    stadium: str
    founded: int
    manager: str
    formation: str
    rival_clubs: list[str] = Field(default_factory=list)
    transfer_budget_gbp: float | None = None
    wage_budget_gbp_per_week: float | None = None
    homegrown_quota: int | None = None
    notes: str | None = None


class ClubConfigError(RuntimeError):
    """Raised when a club configuration file is missing or invalid."""


@lru_cache
def load_club_config(club_id: str) -> ClubConfig:
    """Load a club's configuration from ``data/clubs/<club_id>.yaml``."""
    path = DATA_DIR / "clubs" / f"{club_id}.yaml"
    if not path.exists():
        raise ClubConfigError(
            f"No club configuration found for '{club_id}' at {path}. "
            "Add a YAML file under data/clubs/ to support this club."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return ClubConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise ClubConfigError(f"Invalid club configuration in {path}: {exc}") from exc
