"""Core football domain entities.

Kept separate from ``agent_io`` because domain entities describe the
football world, while ``agent_io`` describes how agents talk to each
other about it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class InjuryRecord(BaseModel):
    """Normalized injury/availability record returned by an :class:`InjuryProvider`."""

    player: str
    status: str = Field(description="e.g. 'Available', 'Injured', 'Doubtful'.")
    injury_type: str | None = None
    expected_return: str | None = None
    as_of_date: str
    source: str


class NewsItem(BaseModel):
    """A single news item returned by a :class:`NewsProvider`.

    ``subject`` is the player or club the item is about; ``category``
    distinguishes confirmed reporting from speculation (e.g. 'confirmed',
    'rumour', 'injury', 'tactical', 'contract') so agents can weigh it
    appropriately rather than treating every item as equally certain.
    """

    subject: str
    headline: str
    category: str
    summary: str
    as_of_date: str
    source: str
