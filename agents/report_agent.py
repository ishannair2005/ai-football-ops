"""Report Agent: writes up the Manager's final recommendation as a report.

Unlike every other agent, its job isn't to judge anything — the verdict,
confidence, and evidence are already decided by the time this agent runs.
It writes the prose; the facts underneath are computed programmatically
and overwritten onto the LLM's output after generation, the same
"attach, don't trust restatement" principle already used for
``FinalRecommendation.agent_responses``.
"""

from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from models.agent_io import FinalRecommendation, ReportRequest, ScoutingReport

ROLE_DESCRIPTION = """
You are the club's report writer. You are given the General Manager's
fully resolved final recommendation — every specialist's findings, the
Devil's Advocate's challenge, and how the Manager resolved it. Combine all
of it into one polished, well-written executive report.

Your `narrative` must read as flowing prose (not a bare restatement of
JSON) and must cover, by name, each of: Scout Report, Performance
Analytics, Tactical Fit, Transfer Cost Analysis, Devil's Advocate
Challenge, and the Manager's Final Decision — pulling each specialist's
findings from the data provided by matching on its agent name.

You do not get to change the verdict, confidence, or the underlying facts
— your job is to present them clearly, not to re-judge them.
""".strip()


class ReportAgent(BaseAgent[ReportRequest, ScoutingReport]):
    # A full narrative covering every specialist plus the Devil's Advocate
    # and the Manager's decision routinely exceeds the 2048-token default,
    # truncating mid-field and producing a schema-invalid response.
    max_output_tokens = 8192

    @property
    def name(self) -> str:
        return "Report Agent"

    @property
    def role_description(self) -> str:
        return ROLE_DESCRIPTION

    @property
    def response_model(self) -> type[ScoutingReport]:
        return ScoutingReport

    def build_user_prompt(self, request: ReportRequest) -> str:
        return f"""
Original question: {request.original_query}

Final recommendation (JSON):
{json.dumps(request.recommendation.model_dump(mode="json"), indent=2)}

Write the executive report via the structured response tool: a fresh,
polished `executive_summary`, and a `narrative` that covers Scout Report,
Performance Analytics, Tactical Fit, Transfer Cost Analysis, Devil's
Advocate Challenge, and the Manager's Final Decision by name, using the
data above. Distinguish verified facts (evidence sourced from a data
provider, with its as-of date) from reasoned analysis and from evidence
gaps — never blur the three together as if a gap were a fact. You may
leave `verdict`, `confidence`, `sources_used`, and `data_freshness` as
placeholders — they will be filled in from the recommendation directly,
not from your response.
""".strip()

    def analyze(self, request: ReportRequest) -> ScoutingReport:
        report = super().analyze(request)
        recommendation = request.recommendation
        report.verdict = recommendation.verdict
        report.confidence = recommendation.confidence
        report.sources_used = self._collect_sources(recommendation)
        report.data_freshness = self._collect_data_freshness(recommendation)
        return report

    @staticmethod
    def _collect_sources(recommendation: FinalRecommendation) -> list[str]:
        sources: set[str] = set()
        responses = list(recommendation.agent_responses)
        if recommendation.devils_advocate_challenge:
            responses.append(recommendation.devils_advocate_challenge)
        for response in responses:
            for evidence in response.supporting_evidence:
                label = f"{evidence.source}"
                if evidence.as_of_date:
                    label += f" (as of {evidence.as_of_date})"
                sources.add(label)
        return sorted(sources)

    @staticmethod
    def _collect_data_freshness(recommendation: FinalRecommendation) -> dict[str, str]:
        """Earliest as-of date per :class:`EvidenceDomain`, so freshness is
        reported per data-type rather than collapsed into one blended date.
        Evidence with no domain tag is skipped rather than guessed into a
        bucket."""
        responses = list(recommendation.agent_responses)
        if recommendation.devils_advocate_challenge:
            responses.append(recommendation.devils_advocate_challenge)

        dates_by_domain: dict[str, list[str]] = {}
        for response in responses:
            for evidence in response.supporting_evidence:
                if evidence.domain is None or evidence.as_of_date is None:
                    continue
                dates_by_domain.setdefault(evidence.domain.value, []).append(evidence.as_of_date)

        return {domain: min(dates) for domain, dates in dates_by_domain.items()}
