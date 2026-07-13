"""Prompt scaffolding for the General Manager Agent's synthesis step."""

from __future__ import annotations

import json

from config.club_config import ClubConfig
from models.agent_io import AgentRequest, AgentResponse, ManagerSynthesis
from prompts.base_prompts import DATA_HONESTY_RULES

MANAGER_ROLE_DESCRIPTION = """
You are the General Manager of the club's football operations department.
You do not perform specialist analysis yourself — you have already
received structured findings from your specialist staff. Your job is to
weigh their evidence, resolve any disagreements between them using
logic (not by averaging their confidence scores), and produce one final
recommendation the club can act on.
""".strip()


def build_manager_system_prompt(club: ClubConfig) -> str:
    return f"""
You are the General Manager of {club.name} ({club.short_name}), playing
in {club.league} under manager {club.manager}.

{MANAGER_ROLE_DESCRIPTION}

{DATA_HONESTY_RULES}

If specialists disagree or one flags high uncertainty, address that
explicitly in your reasoning rather than silently picking a side. If the
evidence is too thin to make a confident call, say so and lower your
confidence score accordingly rather than projecting false certainty.

Every recommendation must include a categorical `verdict`: Buy, Monitor,
or Do Not Sign. Choose the one that's actually consistent with your
`recommendation` text and `confidence` — don't default to Monitor just to
hedge when the evidence clearly points to Buy or Do Not Sign.

You may also be asked to reconcile a Devil's Advocate's challenge against
your own draft recommendation. When that happens, state explicitly in the
`challenge_resolution` field whether you accept, partially accept, or
reject the challenge, and why — reason it through from the evidence.
Never resolve a challenge by splitting the difference or restating the
draft unchanged without engaging with it.

Respond only via the structured tool call provided to you.
""".strip()


_MAX_EXTRACTED_PLAYERS = 5

PLAYER_EXTRACTION_SYSTEM_PROMPT = f"""
You are a lightweight query-parsing step for a football operations
platform, running before any specialist analysis. Your only job is to
decide which player(s), if any, a question names, and extract each name
exactly as written in the query, in the order mentioned.

Do not evaluate, judge, analyze, or say anything about any player. Do not
invent a name if none is given. If the query is about a position group,
formation, squad, or club in general with no individual named, return an
empty list. If the query compares more than {_MAX_EXTRACTED_PLAYERS}
players, extract only the first {_MAX_EXTRACTED_PLAYERS} mentioned.
""".strip()


def build_player_extraction_user_prompt(query: str) -> str:
    return f"""
Query: {query}

Extract every specific player named in this query, if any, via the
structured response tool.
""".strip()


def _specialist_findings(specialist_responses: list[AgentResponse]) -> list[dict]:
    return [
        {
            "agent_name": r.agent_name,
            "summary": r.summary,
            "confidence": r.confidence,
            "supporting_evidence": [e.model_dump(mode="json") for e in r.supporting_evidence],
            "evidence_gaps": r.evidence_gaps,
            "recommended_next_steps": r.recommended_next_steps,
        }
        for r in specialist_responses
    ]


def _specialist_findings_json(specialist_responses: list[AgentResponse]) -> str:
    return json.dumps(_specialist_findings(specialist_responses), indent=2)


def build_manager_user_prompt(
    request: AgentRequest, specialist_responses: list[AgentResponse]
) -> str:
    return f"""
Original question: {request.query}

Specialist findings (JSON):
{_specialist_findings_json(specialist_responses)}

Synthesize these findings into a single draft recommendation via the
structured response tool. Leave `challenge_resolution` unset — there is
no challenge to resolve yet.
""".strip()


COMPARISON_ROLE_DESCRIPTION = """
You are the General Manager comparing multiple candidate signings. Your
job is NOT to rank which player is the better footballer in the
abstract — decide which candidate (if any) is the better SIGNING for
this specific club, right now: fit with the current formation and
manager's known usage, whether they address a genuine squad need or
duplicate a strength the squad already has, and financial fit. A more
talented player who fits worse or costs more can be the wrong choice.

If several candidates are genuinely comparable, say so (`MULTIPLE_VIABLE`)
rather than forcing a false preference. If none of them should be signed,
say that (`NONE_RECOMMENDED`) rather than picking the least-bad option.
""".strip()


def build_comparison_system_prompt(club: ClubConfig) -> str:
    return f"""
You are the General Manager of {club.name} ({club.short_name}), playing
in {club.league} under manager {club.manager}, typically set up in a
{club.formation} formation.

{COMPARISON_ROLE_DESCRIPTION}

{DATA_HONESTY_RULES}

Respond only via the structured tool call provided to you.
""".strip()


def build_comparison_user_prompt(
    request: AgentRequest,
    player_names: list[str],
    responses_by_player: dict[str, list[AgentResponse]],
) -> str:
    candidates = [
        {
            "index": i,
            "player_name": name,
            "specialist_findings": _specialist_findings(responses_by_player[name]),
        }
        for i, name in enumerate(player_names)
    ]
    return f"""
Original question: {request.query}

Candidates under comparison (JSON, indexed):
{json.dumps(candidates, indent=2)}

Decide which candidate (if any) is the better signing for this club via
the structured response tool. If you pick one, set `preferred_player_index`
to its 0-based index above — never restate its name. `verdict_rationale`
must be grounded in fit for this club/system/budget, not raw ability.
""".strip()


def build_resolution_user_prompt(
    request: AgentRequest,
    specialist_responses: list[AgentResponse],
    draft: ManagerSynthesis,
    challenge: AgentResponse,
) -> str:
    return f"""
Original question: {request.query}

Specialist findings (JSON):
{_specialist_findings_json(specialist_responses)}

Your draft recommendation (JSON):
{json.dumps(draft.model_dump(mode="json"), indent=2)}

The Devil's Advocate's challenge to that draft (JSON):
{json.dumps(challenge.model_dump(mode="json"), indent=2)}

Produce your FINAL recommendation via the structured response tool. You
must fill in `challenge_resolution`, explicitly stating whether you
accept, partially accept, or reject the challenge, and why. Revise your
recommendation, confidence, or key risks if the challenge warrants it —
do not average the draft and the challenge together, and do not restate
the draft unchanged without engaging with the challenge's argument.
""".strip()
