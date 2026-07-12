"""Interface every injury-data adapter must implement.

Mirrors :class:`tools.data_provider.PlayerDataProvider` exactly — same
swap-without-touching-agents rationale, applied to availability data
instead of season stats.
"""

from __future__ import annotations

from typing import Protocol

from models.domain import InjuryRecord


class InjuryProvider(Protocol):
    """A source that can look up a player's injury/availability status."""

    def fetch_injury_record(self, player: str) -> InjuryRecord | None:
        """Return a record for ``player``, or ``None`` if this source has nothing on them."""
