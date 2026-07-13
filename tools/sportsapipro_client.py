"""Thin internal HTTP helper shared by every SportsAPIPro-backed provider.

Not part of the public provider architecture (no `Protocol` of its own) —
just avoids duplicating auth-header/timeout/error-handling boilerplate
across the identity, statistics, and transfer adapters. Every domain
adapter still implements its own existing Protocol
(:class:`tools.player_resolver.PlayerResolver`,
:class:`tools.data_provider.PlayerDataProvider`,
:class:`tools.transfer_provider.TransferProvider`) and is independently
swappable; this only removes repeated plumbing underneath them.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.sportsapipro.com/v2/football"
_TIMEOUT_SECONDS = 8


class SportsAPIProClient:
    """Wraps one authenticated GET call. Never raises: a network error,
    timeout, or non-2xx response is logged and surfaced as ``None`` plus
    ``last_error``, so a caller can fall back to the next provider in a
    gateway's priority order exactly as if this source had no record."""

    def __init__(self, api_key: str, base_url: str = _BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.last_error: str | None = None

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.last_error = None
        url = f"{self._base_url}{path}"
        try:
            response = requests.get(
                url,
                headers={"x-api-key": self._api_key},
                params=params,
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self.last_error = f"SportsAPIPro request to {path} failed: {exc}"
            logger.warning(self.last_error)
            return None


#: Identity, statistics, and transfer resolution each need the player's
#: numeric id and would otherwise each run their own `/api/search` for the
#: same name within a single query — tripling calls against a rate-limited
#: API for information that never changes mid-process. Shared, process-wide,
#: and deliberately simple: a dict, not a new abstraction.
_player_id_cache: dict[str, int | None] = {}


def search_player_id(client: SportsAPIProClient, name: str) -> int | None:
    """Resolve ``name`` to a SportsAPIPro player id via `/api/search`,
    caching the result (including "not found") across every caller so a
    given process only ever searches for the same name once."""
    key = name.strip().lower()
    if key in _player_id_cache:
        client.last_error = None
        return _player_id_cache[key]

    data = client.get("/api/search", params={"q": name})
    if data is None:
        # A real error (vs. "not found") shouldn't be cached — a later
        # call might succeed once the transient issue clears.
        return None

    results = data.get("data", {}).get("results", [])
    players = [r for r in results if r.get("type") == "player"]
    player_id = max(players, key=lambda r: r.get("score", 0))["entity"].get("id") if players else None
    _player_id_cache[key] = player_id
    return player_id
