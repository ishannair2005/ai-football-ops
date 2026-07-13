"""Deterministic in-memory transfer/contract provider, for tests and safe fallback."""

from __future__ import annotations

from models.domain import TransferRecord

_DEFAULT_RECORDS: dict[str, TransferRecord] = {
    "demo player": TransferRecord(
        player="Demo Player",
        transfer_fee=None,
        wages=None,
        release_clause=None,
        contract_expiry=None,
        as_of_date="2025-06-01",
        source="mock_transfer_provider",
    )
}


class MockTransferProvider:
    """Returns canned transfer records from a fixed, in-memory lookup table."""

    def __init__(self, records: dict[str, TransferRecord] | None = None) -> None:
        self._records = records if records is not None else _DEFAULT_RECORDS

    def fetch_transfer_record(self, player: str) -> TransferRecord | None:
        return self._records.get(player.strip().lower())
