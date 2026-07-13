"""Interface every player-identity resolver must implement.

Mirrors :class:`tools.data_provider.PlayerDataProvider` — same
swap-without-touching-agents rationale, applied to identity resolution
(raw name -> canonical full name, current club, external IDs) instead of
season stats.
"""

from __future__ import annotations

from typing import Protocol

from models.agent_io import ResolvedIdentity


class PlayerResolver(Protocol):
    """A source that can resolve a raw, user-typed name to a canonical identity."""

    def resolve(self, name: str) -> ResolvedIdentity | None:
        """Return the resolved identity for ``name``, or ``None`` if unknown."""
