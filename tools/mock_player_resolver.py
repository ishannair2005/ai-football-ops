"""Deterministic in-memory player identity resolver, for tests and safe fallback."""

from __future__ import annotations

from models.agent_io import ResolvedIdentity

_DEFAULT_RECORDS: dict[str, ResolvedIdentity] = {
    "demo player": ResolvedIdentity(
        full_name="Demo Player",
        club="Demo FC",
        as_of_date="2025-06-01",
        source="mock_player_resolver",
    )
}


class MockPlayerResolver:
    """Returns canned identities from a fixed, in-memory lookup table."""

    def __init__(self, records: dict[str, ResolvedIdentity] | None = None) -> None:
        self._records = records if records is not None else _DEFAULT_RECORDS

    def resolve(self, name: str) -> ResolvedIdentity | None:
        return self._records.get(name.strip().lower())
