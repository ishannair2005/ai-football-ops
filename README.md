# AI Football Operations Platform

A multi-agent AI system that simulates a professional football club's
operations department. Manchester United is the default configured club;
any club can be added under `data/clubs/` without touching source code.

## Phase 1

- `config/` — runtime settings (`.env`-driven) and club configuration
  loader (`ClubConfig`, `data/clubs/*.yaml`).
- `models/` — shared Pydantic contracts (`AgentRequest`, `AgentResponse`,
  `Evidence`, `FinalRecommendation`) that all agents communicate through.
- `prompts/` — shared prompt scaffolding (club context injection,
  data-honesty rules) reused by every agent.
- `services/` — `LLMClient` (Anthropic-backed, schema-enforced structured
  output), logging config, and the platform factory that wires agents
  together.
- `agents/` — `BaseAgent` (shared LLM-call/validation flow), `ScoutAgent`
  (first specialist), `GeneralManagerAgent` (orchestrator; delegates,
  never analyzes itself).
- `streamlit/app.py` — minimal UI wired to the real agent pipeline.
- `tests/` — unit tests using a fake LLM client (no network calls).

## Phase 2

Adds a real, swappable data layer so agents can ground answers in actual
records instead of LLM recall alone:

- `tools/data_provider.py` — `PlayerDataProvider` Protocol every adapter
  implements. Agents and the gateway depend only on this interface.
- `tools/csv_provider.py` — `CSVPlayerDataProvider`, reads structured
  local data. Deliberately **not** a scraper: sites like FBref,
  Transfermarkt, Sofascore, and FotMob often prohibit scraping in their
  ToS and change HTML frequently, which makes scraper adapters brittle
  and risky. A licensed/official feed export would plug in the same way.
- `tools/mock_provider.py` — `MockPlayerDataProvider`, an in-memory
  fallback used by tests and as a safe default.
- `tools/data_gateway.py` — `PlayerDataGateway` queries all configured
  providers, converts records into `Evidence`, and flags disagreements
  between sources instead of silently picking one.
- `data/player_stats/sample_players.csv` — **illustrative fixture data**
  for exercising the pipeline end to end. Not a live feed; replace with a
  real export before relying on it for actual scouting decisions.
- `ScoutAgent` now accepts an optional `PlayerDataGateway`. When a
  request's `context["player"]` is set, it fetches and cites that
  player's data-provider evidence (with an `as_of_date`); otherwise it
  behaves exactly as in Phase 1, with an explicit "no data" disclosure.
- Streamlit gained an optional "Player name" field — try `Sample Striker`
  against the bundled demo CSV.

## Phase 3

Adds two more specialists, so the General Manager now genuinely
synthesizes across multiple viewpoints instead of relaying a single one:

- `agents/tactical_agent.py` — `TacticalAgent`, evaluates system fit,
  role, chemistry, pressing fit, positional flexibility, and manager
  compatibility. Pure reasoning grounded in the club's formation/manager
  (already injected into every agent's system prompt) — no data provider
  needed.
- `agents/transfer_market_agent.py` — `TransferMarketAgent`, evaluates
  fee, wages, resale value, age curve, contract situation, and financial
  efficiency. Reuses the Phase 2 `PlayerDataGateway` for player
  profile/age grounding, and surfaces the club's `transfer_budget_gbp` /
  `wage_budget_gbp_per_week` from `ClubConfig` — explicitly flagging them
  as unset rather than inventing a figure (both are `null` in the bundled
  club YAMLs).
- `prompts/data_prompts.py` — `build_player_data_section`, the
  fetch-and-cite prompt logic extracted out of `ScoutAgent` once
  `TransferMarketAgent` needed the identical behavior. Both agents call
  this shared function instead of duplicating it.
- `services/platform_factory.py` now assembles all three specialists
  (Scout, Tactical, Transfer Market) behind the General Manager.

## Phase 4

Adds a Devil's Advocate agent and a challenge/resolve loop, so the
General Manager's recommendation goes through one round of adversarial
scrutiny before it's final — the core "reasoning" requirement from the
spec (challenge every recommendation, question assumptions, revise
rather than average):

- `agents/base_agent.py` — `BaseAgent` is now generic over its request
  type (`BaseAgent[RequestT]`). Most agents still take a plain
  `AgentRequest`, but the Devil's Advocate needs the Manager's draft
  recommendation and the specialist findings behind it — genericity lets
  it reuse `analyze`/`system_prompt` unchanged instead of duplicating
  that plumbing in a separate class shape.
- `models/agent_io.py` — new `ChallengeRequest` model (what the Devil's
  Advocate receives); `ManagerSynthesis` gains `challenge_resolution`
  (populated only on the post-challenge synthesis); `FinalRecommendation`
  gains `challenge_resolution` and `devils_advocate_challenge`.
- `agents/devils_advocate_agent.py` — `DevilsAdvocateAgent`, argues the
  strongest case against the draft recommendation: hidden risks,
  questioned assumptions, glossed-over disagreement between specialists.
  Reuses the existing `AgentResponse` contract — no new response schema.
- `agents/manager_agent.py` — `GeneralManagerAgent.handle_query` now:
  consult specialists → draft synthesis → (if a Devil's Advocate is
  configured) challenge → **final** resolution synthesis that must
  explicitly accept, partially accept, or reject the challenge and say
  why, never split the difference. The `devils_advocate` param defaults
  to `None`, so callers that don't configure one get Phase 1–3 behavior
  unchanged.
- `prompts/manager_prompts.py` — new `build_resolution_user_prompt`
  (shares a `_specialist_findings_json` helper with the existing draft
  prompt); `build_manager_system_prompt` gained one paragraph on
  resolving challenges by reasoning, not averaging.
- `services/platform_factory.py` now always wires a `DevilsAdvocateAgent`
  into the General Manager.
- Streamlit shows the Devil's Advocate's challenge and a "How This Was
  Resolved" section — the platform's concrete risk-analysis view.

## Phase 5 (current)

The first complete, production-quality vertical slice: a fourth
specialist, a real injury/news data layer, a categorical verdict, a
Report Agent, and a Streamlit UI that renders every required section end
to end for a query like "Should Manchester United sign Sample Striker?"

- `models/agent_io.py` — new `RecommendationVerdict` enum (`Buy` /
  `Monitor` / `Do Not Sign`), **required** on both `ManagerSynthesis` and
  `FinalRecommendation` so the schema forces the Manager to decide it
  every time. New `ReportRequest`, `ScoutingReport`, and `PlatformResult`
  models for the Report Agent and the platform facade.
- `models/domain.py` — new `InjuryRecord` and `NewsItem`.
- `tools/injury_provider.py` / `csv_injury_provider.py` /
  `mock_injury_provider.py` / `injury_gateway.py` and
  `tools/news_provider.py` / `csv_news_provider.py` /
  `mock_news_provider.py` / `news_gateway.py` — mirror the Phase 2
  `PlayerDataProvider` pattern exactly for two new domains. Deliberate
  scope cut: neither gateway does cross-source disagreement detection
  (`PlayerDataGateway` already proves that pattern once). `InjuryGateway`
  wires into `ScoutAgent` (physical traits are already its remit);
  `NewsGateway` wires into `DevilsAdvocateAgent` (recent rumours/manager
  comments are exactly its kind of ammunition) via a new `player` field on
  `ChallengeRequest`, falling back to the club for player-less queries.
  `data/injuries/sample_injuries.csv` and `data/news/sample_news.csv` are
  illustrative fixtures, same convention as the Phase 2 CSV.
- `agents/performance_analytics_agent.py` — `PerformanceAnalyticsAgent`,
  the fourth specialist. Covers xG/xA/NPxG/SCA/GCA/progressive
  actions/pressures/expected threat; grounds what it can (season
  appearances/goals/assists) via the existing `PlayerDataGateway`, and is
  explicit that advanced tracking metrics aren't available from a live
  provider — the same honest-degradation pattern as Transfer Market
  Agent's budget handling.
- `agents/base_agent.py` — `BaseAgent` is now generic over **both** its
  request and response type (`BaseAgent[RequestT, ResponseT]`), via a new
  abstract `response_model` property. This is what lets the Report Agent
  (which returns a `ScoutingReport`, not an `AgentResponse`) reuse
  `analyze`/`system_prompt` unchanged rather than duplicating that
  plumbing a second time.
- `agents/report_agent.py` — `ReportAgent`, writes the polished narrative
  report. It doesn't get to redecide anything: `analyze()` calls the LLM
  for prose (`executive_summary`, `narrative`) and then **overwrites**
  `verdict`, `confidence`, `sources_used`, and `data_as_of` programmatically
  from the actual `FinalRecommendation` evidence — the same
  attach-don't-trust-restatement principle already used for
  `FinalRecommendation.agent_responses`. `sources_used` is deduped across
  every specialist's and the Devil's Advocate's evidence; `data_as_of` is
  the earliest (most conservative) date among it.
- `services/platform_factory.py` — new `FootballOperationsPlatform`
  facade and `build_platform(...)`, running the Manager then the Report
  Agent and returning both (`PlatformResult`). `build_general_manager` is
  unchanged for existing callers/tests.
- Streamlit renders, in order: Executive Summary, Overall Recommendation
  (Buy/Monitor/Do Not Sign badge), Confidence, four labeled specialist
  reports (Scout Report / Performance Analytics / Tactical Fit / Transfer
  Cost Analysis), the Devil's Advocate Challenge, the Manager's Final
  Decision, Sources Used, Data Freshness, and the full narrative.

Not yet built (later phases): remaining specialist agents (Medical,
Opponent Analysis, Squad Planning, News & Context as a dedicated agent),
scraping/licensed-API data adapters beyond the bundled CSV/mock
providers, a real news API integration, Streamlit charts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Run tests

```bash
pytest
```

## Run the app

```bash
streamlit run streamlit/app.py
```

## Switching clubs

Set `ACTIVE_CLUB` in `.env` to any file name under `data/clubs/` (e.g.
`arsenal`), or add a new YAML file there following the same schema as
`data/clubs/manchester_united.yaml`.
