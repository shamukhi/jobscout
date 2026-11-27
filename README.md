# Jobscout

**An AI job-search agent for students and early-career people: hybrid semantic search over a changing jobs index, personalised ranking, a tool-using LLM agent with persistent memory, and an eval harness that measures both retrieval and agent behaviour.**

**Live demo:** [TODO: paste your Render URL] (free tier: the first load can take about a minute to wake)

[TODO: add a screenshot or a short GIF of the chat finding a role, saved as docs/demo.png]

## What it does

- You tell it your skills and what you want ("I know Python and FastAPI, find remote internships"). It **remembers your profile** between visits.
- It searches a jobs index with **hybrid retrieval** (embeddings plus keyword search), **filters** by location, remote and job type, and **re-ranks for you** using your skills and preferences.
- Every recommendation is **structured, validated and explained**: which of your skills matched, whether it's remote, how recent it is.
- Jobs can be added, updated and expired at any time without rebuilding the index.

## Architecture

```mermaid
flowchart LR
    U[User] --> API[FastAPI]
    API --> AG[Agent loop: LLM + 7 tools]
    API --> SE[Search endpoint]
    AG -->|search_jobs| RE[Retriever]
    SE --> RE
    RE --> DE[(Chroma: dense vectors)]
    RE --> BM[BM25: keywords]
    DE --> RR[RRF fusion]
    BM --> RR
    RR --> PR[Personalised re-rank]
    AG <--> DB[(SQLite: jobs, profiles, chat memory, saved jobs)]
    IN[Ingest: add / update / expire jobs] --> DE
    IN --> DB
```

## What to look at

| If you care about | Look at |
|---|---|
| Retrieval, ranking, personalisation | `app/retrieval.py`: dense + BM25 → Reciprocal Rank Fusion → re-rank with a per-result score breakdown |
| A search index that keeps changing | `app/index.py`: hash-diff sync that re-embeds only changed text, does cheap metadata updates, and prunes expired jobs |
| LLM agents, tool use, memory | `app/agent.py`: tool-use loop, long-term profile memory, short-term chat memory |
| Structured outputs and guardrails | `present_recommendations` is schema-validated and every job id must have come from a tool result that turn, so it cannot recommend jobs that do not exist |
| Evaluating LLM and retrieval quality | `evals/`: Recall@k, MRR, nDCG, latency per pipeline stage, plus behavioural agent tests |
| Provider portability | `app/llm.py`: runs on Gemini, Groq, Cerebras, OpenRouter, Mistral, Ollama or Claude with one env var |

## Results

[TODO: replace every XX with the numbers from evals/results.md after running `python -m evals.run_retrieval_eval`, and fill in the dataset line.]

Dataset: [TODO: N real job postings from <source>], [TODO: M hand-labelled queries]. Embeddings: `BAAI/bge-small-en-v1.5`.

| Pipeline stage | Recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|
| BM25 only | XX | XX | XX | XX ms |
| Dense only | XX | XX | XX | XX ms |
| Hybrid (RRF) | XX | XX | XX | XX ms |
| Hybrid + personalised | XX | XX | XX | XX ms |

Agent behaviour eval (`python -m evals.run_agent_eval`): [TODO: X/5] cases pass, covering correct tool routing, filter extraction, resistance to instructions injected into job text, and no hallucinated job ids.

[TODO: one sentence on the most interesting finding, e.g. what hybrid fixed that dense-only missed, or a failure the agent eval caught and how you fixed it.]

## Engineering decisions

- **Hybrid retrieval with RRF.** Embeddings handle paraphrase ("agents that call tools" ≈ "AI Agents Engineer"); BM25 handles exact skill names and acronyms. Reciprocal Rank Fusion merges ranks, so no score calibration is needed.
- **Explainable personalisation.** The final score is a fixed blend: 0.60 relevance, 0.20 skill overlap, 0.12 preference match, 0.08 recency. Every result returns its breakdown, so ranking is debuggable rather than a black box.
- **Incremental indexing.** Each job carries two hashes (text vs all fields), so editing a salary doesn't pay for a new embedding. One vector collection per embedding model means switching models never mixes vector spaces. Chroma can always be rebuilt from SQLite.
- **Guardrails on the agent.** Job text is treated as untrusted data (prompt-injection defence), tool errors go back to the model instead of crashing, recommended ids are verified against tool results, and the loop has a hard step cap.
- **Provider-agnostic LLM layer.** The agent is written once against a single message format; an adapter converts to any OpenAI-compatible endpoint, so the project runs on free tiers.

## Tech stack

Python · FastAPI · Pydantic · ChromaDB · BM25 (rank-bm25) · fastembed (ONNX embeddings) · SQLite · Gemini / Claude / OpenAI-compatible LLMs · Docker · pytest

## Run it locally

```bash
git clone https://github.com/<you>/jobscout && cd jobscout
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q                                                # 22 tests, no network or API key needed
export GEMINI_API_KEY=...                                # free key from Google AI Studio
uvicorn app.main:app                                     # http://localhost:8000
```

Search, indexing and the retrieval eval work without any API key; only `/chat` needs one. Full setup, provider options, real-data ingestion and deployment are in [docs/SETUP.md](docs/SETUP.md).

## Project layout

```
app/        agent.py  retrieval.py  index.py  llm.py  store.py  models.py  main.py
evals/      retrieval eval, agent eval, labelled queries
scripts/    sample-data generator, CSV ingest for real postings
tests/      22 tests: sync, filters, ranking, agent loop, LLM adapter, API
static/     minimal chat UI
```

## Limitations and next steps

- Free-tier hosting resets user profiles and chat memory when the instance restarts; the job index reloads automatically. A persistent disk or Postgres would fix this.
- `/chat` has no authentication or rate limiting yet.
- BM25 is rebuilt in memory when jobs change, which is fine for tens of thousands of jobs. Beyond that, move to SQLite FTS5 or Elasticsearch.
- Next: a cross-encoder or LLM re-ranker on the top 20, measured in the same eval table, and an LLM-as-judge check that each recommendation reason is supported by the job text.
