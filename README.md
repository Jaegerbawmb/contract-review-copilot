# Risk Desk — Contract Review Copilot

A multi-agent LangGraph pipeline that reviews vendor contracts against your company's own risk playbook — flagging liability, auto-renewal, data-handling, and termination risk before you sign.

This isn't a single prompt asking "is this contract risky?". It's a graph: clauses are extracted in parallel, checked against your playbook, re-analyzed if a reading is ambiguous, and optionally audited by an independent model before the report reaches you.



## Architecture

```
        START
          |
   dispatch_extraction  --Send--> extractor (parallel, one per clause category)
          |                              |
          '------------------------------'
                       |
                  risk_flagger
                    /      \
        (sufficient)        (ambiguous category flagged, rounds left)
                |                      |
           report_writer      dispatch_refinement --Send--> extractor_deeper
                |                                                |
          [auditor, optional]                              risk_flagger (loop back)
                |
               END
```

- **extractor** — runs once per clause category (`liability`, `auto_renewal`, `data_handling`, `termination`), fanned out in parallel via LangGraph's `Send` API. Extracts a plain-English summary, key terms, and a self-reported confidence level.
- **risk_flagger** — compares all extractions against the company's playbook and assigns each a risk level (`compliant` / `non_standard` / `high_risk` / `not_addressed`). If a category's extraction was too thin or ambiguous, it's flagged for a deeper pass instead of guessing.
- **refinement loop** — flagged categories get a second, sharper extraction pass targeting exactly what was ambiguous, then the risk flagger re-runs. Bounded by `max_refinement_rounds` so the graph always terminates.
- **report_writer** — writes the final report: a plain-English summary, risk flags, an overall risk level, and concrete recommended actions.
- **auditor** *(optional, off by default)* — an independent LLM (separate provider from the main pipeline) reviews the draft report for internal consistency — does each risk level actually match its own rationale? — and can rewrite the report if it finds a real mismatch. This is an LLM-as-judge pattern, deliberately decoupled from the extraction pipeline's provider so it doesn't share the same blind spots.

## Features

- 🔀 True parallel fan-out per clause category via LangGraph's `Send` API
- 🔁 Conditional refinement cycle — the graph re-checks its own low-confidence reads, not a straight-line chain
- ⚖️ Optional LLM-as-judge auditor on an independent model provider, toggled via `ENABLE_AUDITOR`
- 💾 Disk-based response cache — re-running the same contract costs zero API calls after the first pass
- 🖥️ FastAPI backend + a self-contained HTML/CSS/JS frontend
- ✅ Fully mocked structural test proving the fan-out/loop/termination logic, at zero API cost

## Tech stack

`LangGraph` · `LangChain` · `Gemini` (extraction pipeline) · `Groq` (independent judge) · `FastAPI` · `Pydantic`

## Project structure

```
schemas.py             # Pydantic models + LangGraph state (TypedDict + reducers)
config.py               # LLM client factories (main pipeline + independent judge)
graph.py                  # StateGraph wiring: nodes, Send fan-out, conditional edges
clause_extractor.py         # per-category clause extraction (+ deeper-pass variant)
risk_flagger.py               # cross-category risk assessment against the playbook
report_writer.py                # final report writer
judge.py                          # LLM-as-judge auditor (independent model)
llm_cache.py                        # disk-based response cache
main.py                               # FastAPI app (/review)
index.html                              # frontend
test_graph_flow.py                        # mocked end-to-end test (no API calls)
sample_run.py                               # real run against a synthetic sample contract
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY and (optionally) GROQ_API_KEY
```

## Run the mocked structural test (no API key needed)

```bash
python test_graph_flow.py
```

Proves the graph's control flow — parallel fan-out, risk-flagger-triggered refinement loop, targeted re-extraction, and clean termination — independent of LLM output quality.

## Run against a real contract

```bash
python sample_run.py
```

Uses a synthetic sample contract with a deliberately mixed risk profile (one clearly risky clause, one borderline clause, two standard ones) so the output should show meaningfully different risk levels across categories.

## Run the full app

```bash
uvicorn main:app --reload
```

Then open `index.html` directly in a browser.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. Powers the main extraction/risk/report pipeline. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model for the main pipeline. |
| `GROQ_API_KEY` | — | Required only if `ENABLE_AUDITOR=true`. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model for the independent judge. |
| `ENABLE_AUDITOR` | `false` | Turns on the LLM-as-judge auditor node. |
| `LLM_CACHE_ENABLED` | `true` | Caches identical LLM calls to disk. |
| `LLM_CACHE_DIR` | `.llm_cache` | Where cached responses are stored. |

## Design decisions worth knowing

- **Why `Send` fan-out instead of a for-loop?** Clause categories are independent until risk assessment — parallelizing cuts latency roughly linearly with category count. LangGraph's reducer (`Annotated[list, operator.add]`) lets parallel branches safely append to shared state.
- **Why a targeted refinement loop, not a blanket re-run?** Only the specific categories the risk flagger marks ambiguous get a deeper pass — not every category, every time.
- **Why cap `max_refinement_rounds`?** LLM-driven conditional loops can spin forever if the model keeps flagging ambiguity. The round cap forces termination without a hard timeout hack.
- **Why put the judge on a different model provider than the extraction pipeline?** An independent model is less likely to share the same systematic blind spots as the model being audited, and it decouples the judge from the main pipeline's rate limits entirely.
- **Why cache LLM responses to disk?** Iterating on prompts or the frontend shouldn't mean re-paying for identical API calls every test run.

## License

MIT
