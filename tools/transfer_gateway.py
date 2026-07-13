"""Aggregates one or more :class:`TransferProvider` sources into evidence.

Simpler than :class:`tools.data_gateway.PlayerDataGateway` by design, same
scope cut as :class:`tools.injury_gateway.InjuryGateway`: no cross-source
disagreement detection. Providers are queried in priority order and the
first hit wins.
"""

from __future__ import annotations

from models.agent_io import Evidence, EvidenceDomain, EvidenceSource
from models.domain import TransferRecord
from tools.transfer_provider import TransferProvider


class TransferGateway:
    def __init__(self, providers: list[TransferProvider]) -> None:
        self._providers = providers
        #: See PlayerIdentityGateway.last_error — set when a live provider
        #: errored on the most recent lookup rather than simply having no
        #: record for the player.
        self.last_error: str | None = None

    def fetch_transfer_evidence(self, player: str) -> list[Evidence]:
        self.last_error = None
        for provider in self._providers:
            record = provider.fetch_transfer_record(player)
            if record is not None:
                return [self._record_to_evidence(record)]
            error = getattr(provider, "last_error", None)
            if error:
                self.last_error = error
        return []

    @staticmethod
    def _record_to_evidence(record: TransferRecord) -> Evidence:
        fields = (
            ("fee", record.transfer_fee),
            ("wages", record.wages),
            ("release clause", record.release_clause),
            ("contract until", record.contract_expiry),
        )
        known = ", ".join(f"{label} {value}" for label, value in fields if value is not None)
        detail = known or "no fee, wage, release clause, or contract expiry on file"
        return Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description=f"{record.player}: {detail}",
            value=f"source={record.source}",
            as_of_date=record.as_of_date,
            confidence=0.9,
            domain=EvidenceDomain.TRANSFER_MARKET,
        )
