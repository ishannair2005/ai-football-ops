"""Live transfer/contract provider backed by sportsapipro.com.

`/api/players/{id}/transfer-history` gives real transfer fees, dates, and
clubs for the player's most recent move. `/api/players/{id}` separately
exposes `contractUntilTimestamp`, so contract expiry is read from there.
Wages and release clauses aren't exposed by this API at all and are left
`None` — never estimated from the transfer fee or market value.
"""

from __future__ import annotations

from datetime import datetime, timezone

from models.domain import TransferRecord
from tools.sportsapipro_client import SportsAPIProClient, search_player_id


class SportsAPIProTransferProvider:
    """Implements :class:`tools.transfer_provider.TransferProvider`."""

    def __init__(self, api_key: str) -> None:
        self._client = SportsAPIProClient(api_key)
        self.last_error: str | None = None

    def fetch_transfer_record(self, player: str) -> TransferRecord | None:
        self.last_error = None
        player_id = search_player_id(self._client, player)
        self.last_error = self._client.last_error
        if player_id is None:
            return None
        return self._fetch_record(player_id, player)

    def _fetch_record(self, player_id: int, queried_name: str) -> TransferRecord | None:
        profile_data = self._client.get(f"/api/players/{player_id}")
        self.last_error = self.last_error or self._client.last_error
        player = (profile_data or {}).get("data", {}).get("player")
        if not player:
            return None

        contract_until_ts = player.get("contractUntilTimestamp")
        contract_expiry = (
            datetime.fromtimestamp(contract_until_ts, tz=timezone.utc).date().isoformat()
            if contract_until_ts
            else None
        )

        history_data = self._client.get(f"/api/players/{player_id}/transfer-history")
        self.last_error = self.last_error or self._client.last_error
        history = (history_data or {}).get("data", {}).get("transferHistory", [])

        transfer_fee = None
        if history:
            most_recent = max(history, key=lambda t: t.get("transferDateTimestamp") or 0)
            description = most_recent.get("transferFeeDescription")
            if description:
                from_club = most_recent.get("fromTeamName")
                to_club = most_recent.get("toTeamName")
                transfer_fee = f"{description} ({from_club} to {to_club})"

        if transfer_fee is None and contract_expiry is None:
            return None

        return TransferRecord(
            player=player.get("name", queried_name),
            transfer_fee=transfer_fee,
            wages=None,
            release_clause=None,
            contract_expiry=contract_expiry,
            as_of_date=datetime.now(timezone.utc).date().isoformat(),
            source="sportsapipro.com",
        )
