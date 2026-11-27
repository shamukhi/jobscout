# Jobscout

A job-search agent: hybrid semantic + keyword retrieval over a constantly changing jobs index, personalised ranking, a tool-using Claude agent with persistent memory, and an eval harness for both retrieval and agent behaviour.

```
                 ┌────────────── FastAPI ──────────────┐
 browser/API ──► │ /chat   /search   /ingest   /jobs   │
                 └───────┬───────────────┬─────────────┘
                         │               │
                   JobAgent (Claude)   Searcher
                   tool-use loop        │  dense: Chroma (cosine)  ─┐
                   memory + guards      │  sparse: BM25            ─┼─► RRF ─► filters ─► personalised re-rank
                         │              │                           ┘
                         └────► SQLite (jobs, profiles, chat memory, saved/dismissed)
                                 ▲
                      JobIndex.sync(): hash-diff incremental updates + pruning
```

## What maps to the Elip job description

| They ask for | Where it lives |
|---|---|
| LLM agents that use tools and remember context | `app/agent.py` (7 tools, profile + chat memory in SQLite) |
| Structured outputs | `present_recommendations` tool: Pydantic-validated, ids checked against the DB |
| Retrieval, ranking, personalisation | `app/retrieval.py` (dense + BM25 → RRF → skill/preference/recency re-rank, with per-hit reasons) |
| Semantic search over a changing index | `app/index.py` `sync()` (adds, re-embeds only changed text, metadata-only updates, prunes expired) |
| Evaluate LLM output quality | `evals/` (retrieval metrics per stage; agent routing, injection and hallucination checks) |
| Python backend, SQL, APIs | FastAPI + SQLite, admin-protected ingest, validated request models |
| Docker / cloud / live users | `Dockerfile`, `static/index.html` chat page |

## Run it

```bash
pip install -r requirements-dev.txt
python scripts/make_sample_data.py          # 120 synthetic jobs + eval set (already committed)
pytest -q                                   # 20 tests, no network or API key needed

export GEMINI_API_KEY=...                   # or GROQ_API_KEY, ANTHROPIC_API_KEY, ... (see "Choosing an LLM")
uvicorn app.main:app --reload               # open http://localhost:8000  (API docs at /docs)
```

Search, indexing and the retrieval eval need no key at all. Only `/chat` and the agent eval call an LLM.

First boot loads `data/jobs.json` and downloads the embedding model (BAAI/bge-small-en-v1.5 via fastembed, ~130 MB). `EMBED_BACKEND=hash` skips the download but is lexical only; use it for tests, not for demos.

## Choosing an LLM (free options included)

Set one provider key and the app picks it up (`LLM_PROVIDER=auto`, checked in the order below), or force one with `LLM_PROVIDER`.

| Provider | Env var | Default model | Notes |
|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` | Free tier from Google AI Studio, no card. Exact per-model limits are shown in AI Studio and change. |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Free tier, very fast. Tight daily caps on larger models. |
| `cerebras` | `CEREBRAS_API_KEY` | set `LLM_MODEL` | Free tier, fast. |
| `openrouter` | `OPENROUTER_API_KEY` | set `LLM_MODEL` (a `:free` model) | Many free models, small daily cap, free model list churns. |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` | Free experiment tier. |
| `ollama` | none | `llama3.1` | Runs locally, no limits, no data leaves your machine. Needs a tool-calling model and decent RAM. |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | Paid, strongest tool use in my testing plan; best for the final demo. |
| `custom` | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | none | Any OpenAI-compatible endpoint. |

All non-Anthropic providers go through the OpenAI-compatible adapter in `app/llm.py`; the agent code is identical. Model ids and free limits change, so override with `LLM_MODEL` if a default is retired.

Not every model is equally good at tool calling. Run `python -m evals.run_agent_eval` on the provider you pick; that is exactly what it is for. Also read each provider's data policy before sending real user messages through a free tier.

## Use real data (do this before you put numbers on a resume)

The bundled data is synthetic, and its eval labels come from the generator, so the eval proves the pipeline works, not that it is good.

1. Download a public postings dataset (e.g. a Kaggle LinkedIn job postings dump) and ingest it:
   `python scripts/ingest_csv.py jobs.csv --title-col ... --desc-col ... --post http://localhost:8000 --admin-token $ADMIN_TOKEN`
2. Hand-label ~30 queries in `evals/queries.jsonl` (`{"query", "filters", "relevant": [job ids], "profile"?}`). Include ambiguous ones with a profile so personalisation is actually tested.
3. `python -m evals.run_retrieval_eval` → `evals/results.md` (Recall@k, MRR, nDCG, p50/p95 latency for bm25 / dense / hybrid / personalized).
4. `python -m evals.run_agent_eval` → tool-routing, filter-extraction, prompt-injection and hallucinated-id checks (real LLM calls).

## Design decisions worth being able to explain

- **Hybrid + RRF.** Dense handles paraphrase ("agents that call tools" ≈ "AI Agents Engineer"); BM25 handles exact skill terms and acronyms. RRF fuses ranks, so no score calibration is needed.
- **Explainable personalisation.** Final score = 0.60 relevance + 0.20 skill overlap + 0.12 preference match + 0.08 recency. Every hit returns its breakdown and human-readable reasons, which the agent quotes.
- **Incremental sync.** Two hashes per job (text vs all fields) so a salary edit does not trigger re-embedding. One Chroma collection per embedding model, so swapping models never mixes vector spaces. Chroma is rebuildable from SQLite.
- **Agent guards.** Recommended ids must have been returned by a tool in the same turn (no hallucinated jobs); job text is flagged as untrusted in the system prompt; tool errors go back to the model as `is_error` results; hard step cap.
- **Memory.** Long-term: profile and saved/dismissed jobs (dismissed are excluded from search). Short-term: last 12 turns.

## Deploy

Any container host works (Render, Fly, Railway). Mount a volume at `/data`, set your provider key (e.g. `GEMINI_API_KEY`) and `ADMIN_TOKEN`, expose the port. The image pre-downloads the embedding model so cold starts are fast. For a feed that changes, run `scripts/ingest_csv.py --prune` (or POST `/ingest` with `"prune": true`) on a schedule.

## Known limits / next steps

- `/chat` has no auth or rate limiting. Add both before sharing publicly, since each call spends LLM tokens.
- BM25 is rebuilt in memory when the jobs table changes; fine to tens of thousands of jobs, then move to SQLite FTS5 or Elasticsearch.
- Add a cross-encoder or LLM re-ranker on the top 20 and measure it in the eval table.
- Add an LLM-as-judge check for recommendation reasons (are they supported by the job text?).
