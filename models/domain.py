"""Core football domain entities.

Intentionally minimal for Phase 1 — this grows as data-provider adapters
land in later phases. Kept separate from ``agent_io`` because domain
entities describe the football world, while ``agent_io`` describes how
agents talk to each other about it.
"""

from __future__ import annotations

from pydantic import BaseModel


class Player(BaseModel):
    name: str
    position: str
    club: str | None = None
    age: int | None = None
    nationality: str | None = None
