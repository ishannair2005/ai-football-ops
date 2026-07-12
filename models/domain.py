"""Core football domain entities.

Kept separate from ``agent_io`` because domain entities describe the
football world, while ``agent_io`` describes how agents talk to each
other about it.
"""

from __future__ import annotations

from pydantic import BaseModel


class Player(BaseModel):
    name: str
    position: str
    club: str | None = None
    age: int | None = None
    nationality: str | None = None


class PlayerStatsRecord(BaseModel):
    """Normalized player record returned by a :class:`PlayerDataProvider`.

    Fields are deliberately limited to attributes a fixture/CSV source can
    state honestly (identity, profile, season tallies) rather than
    perishable advanced metrics that would go stale silently. Every record
    carries the date it's current as of, per the platform's data-honesty
    rules.
    """

    name: str
    position: str
    club: str | None = None
    age: int | None = None
    nationality: str | None = None
    appearances: int | None = None
    goals: int | None = None
    assists: int | None = None
    as_of_date: str
    source: str
