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
    minutes: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    pass_accuracy: float | None = Field(
        default=None, description="Completed-pass percentage, e.g. 85.9."
    )
    tackles: int | None = None
    interceptions: int | None = None
    progressive_carries: int | None = Field(
        default=None,
        description="Left unset by every current provider — none of them report "
        "this metric, and it is never estimated from adjacent stats.",
    )
    progressive_passes: int | None = Field(default=None, description="See progressive_carries.")
    as_of_date: str
    source: str


class TransferRecord(BaseModel):
    """Normalized transfer/contract record returned by a :class:`TransferProvider`.

    Every money/date field is optional and left ``None`` when a source
    doesn't state it — never estimated or inferred, since a plausible-
    looking fee or expiry date would be indistinguishable from a real one.
    """

    player: str
    transfer_fee: str | None = None
    wages: str | None = None
    release_clause: str | None = None
    contract_expiry: str | None = None
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
