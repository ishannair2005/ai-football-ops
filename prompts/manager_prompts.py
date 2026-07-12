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

You may also be asked to reconcile a Devil's Advocate's challenge against
your own draft recommendation. When that happens, state explicitly in the
`challenge_resolution` field whether you accept, partially accept, or
reject the challenge, and why — reason it through from the evidence.
Never resolve a challenge by splitting the difference or restating the
draft unchanged without engaging with it.

Respond only via the structured tool call provided to you.
""".strip()


def _specialist_findings_json(specialist_responses: list[AgentResponse]) -> str:
    findings = [
        {
            "agent_name": r.agent_name,
            "summary": r.summary,
            "confidence": r.confidence,
            "supporting_evidence": [e.model_dump(mode="json") for e in r.supporting_evidence],
            "assumptions": r.assumptions,
            "uncertainties": r.uncertainties,
            "recommended_next_steps": r.recommended_next_steps,
        }
        for r in specialist_responses
    ]
    return json.dumps(findings, indent=2)


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
