"""Live player-identity resolver backed by sportsapipro.com.

The free-search API has no direct "look up by exact name" endpoint —
identity resolution is two calls: `/api/search?q=name` to find the best
matching player entity, then `/api/players/{id}` for the full profile.
Results are cached per queried name so a repeat lookup in the same
process doesn't re-hit the API. Any network error, timeout, or "no
player found" result in ``None`` — never a raised exception — so the
:class:`~tools.player_identity_gateway.PlayerIdentityGateway` falls
through to the next provider exactly as it does today.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from models.agent_io import ResolvedIdentity
from tools.sportsapipro_client import SportsAPIProClient, search_player_id

logger = logging.getLogger(__name__)

_POSITION_LABELS = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}


def _age_from_date_of_birth(date_of_birth: str | None) -> int | None:
    if not date_of_birth:
        return None
    try:
        dob = datetime.fromisoformat(date_of_birth.replace("Z", "+00:00")).date()
    except ValueError:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


class SportsAPIProIdentityProvider:
    """Implements :class:`tools.player_resolver.PlayerResolver`."""

    def __init__(self, api_key: str) -> None:
        self._client = SportsAPIProClient(api_key)
        self._cache: dict[str, ResolvedIdentity | None] = {}
        self.last_error: str | None = None

    def resolve(self, name: str) -> ResolvedIdentity | None:
        key = name.strip().lower()
        if key in self._cache:
            return self._cache[key]

        self.last_error = None
        player_id = search_player_id(self._client, name)
        self.last_error = self._client.last_error
        if player_id is None:
            self._cache[key] = None
            return None

        identity = self._fetch_profile(player_id)
        self._cache[key] = identity
        return identity

    def _fetch_profile(self, player_id: int) -> ResolvedIdentity | None:
        data = self._client.get(f"/api/players/{player_id}")
        self.last_error = self.last_error or self._client.last_error
        if data is None:
            return None

        player = data.get("data", {}).get("player")
        if not player:
            return None

        team = player.get("team") or {}
        tournament = (team.get("uniqueTournament") or team.get("tournament") or {}).get("name")
        position_code = player.get("position")

        return ResolvedIdentity(
            full_name=player.get("name", ""),
            club=team.get("name"),
            position=_POSITION_LABELS.get(position_code, position_code),
            nationality=(player.get("country") or {}).get("name"),
            age=_age_from_date_of_birth(player.get("dateOfBirth")),
            competition=tournament,
            player_ids={"sportsapipro": str(player_id)},
            as_of_date=datetime.now(timezone.utc).date().isoformat(),
            source="sportsapipro.com",
        )
