"""Interface every transfer/contract adapter must implement.

Mirrors :class:`tools.injury_provider.InjuryProvider` exactly — same
swap-without-touching-agents rationale, applied to transfer fee, wages,
release clause, and contract expiry instead of availability data.
"""

from __future__ import annotations

from typing import Protocol

from models.domain import TransferRecord


class TransferProvider(Protocol):
    """A source that can look up a player's transfer/contract record."""

    def fetch_transfer_record(self, player: str) -> TransferRecord | None:
        """Return a record for ``player``, or ``None`` if this source has nothing on them."""
