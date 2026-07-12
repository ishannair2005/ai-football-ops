"""Interface every player-data adapter must implement.

Mirrors the :class:`services.llm_client.LLMClient` Protocol pattern: agents
and the gateway depend on this Protocol, never on a concrete adapter, so
swapping or adding data sources never touches agent code.
"""

from __future__ import annotations

from typing import Protocol

from models.domain import PlayerStatsRecord


class PlayerDataProvider(Protocol):
    """A source that can look up a player by name."""

    def fetch_player(self, name: str) -> PlayerStatsRecord | None:
        """Return a record for ``name``, or ``None`` if this source has nothing on them."""
