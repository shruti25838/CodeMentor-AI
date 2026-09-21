# CodeMentor AI

**An agentic code-intelligence assistant that answers natural-language questions about a codebase — combining semantic retrieval with structural (AST) analysis, so it doesn't just sound right, it's actually right.**

🔗 **Live demo:** [codemento-ai.vercel.app](https://codemento-ai.vercel.app)

*(Formerly named CodeAtlas — the Python package and internal env vars still use that name, e.g. `codeatlas/`, `CODEATLAS_*`.)*

---

## Why it's built this way

The first version of CodeMentor AI relied on pure semantic search: embed the codebase, retrieve the closest chunks by vector similarity, hand them to an LLM. It demoed beautifully — clean answers, fast, impressive in front of anyone watching.

It was also quietly wrong.

Ask "where is this function called?" and a pure embedding retriever will confidently return code that's *semantically* related to the question, not code that's *structurally* connected to the answer. Embeddings capture meaning, not call graphs, import chains, or scope. The demo never revealed this, because demos are usually a handful of questions the builder already expects to work.

I caught the problem by doing the thing that doesn't demo well: building an evaluation set of real questions with known-correct answers, including structural ones like "who calls this?" and "what does this function depend on?" Running that set against the semantic-only pipeline surfaced a consistent, silent failure mode on exactly the questions that matter most for understanding unfamiliar code.

The fix was to rearchitect: add a structural/AST analysis path alongside semantic retrieval, and route each question to the path (or combination of paths) that can actually answer it. The failing cases from the original eval set became permanent regression coverage, so this specific failure mode can never come back unnoticed.

The principle driving the project going forward: **the impressive demo is the easy part. The real work — and the real signal of whether a system works — is the evals that catch what the demo doesn't show you.**

---

## Architecture

CodeMentor AI is an agentic pipeline, not a single retrieval call. A **LangGraph planner** looks at each incoming question and routes it down one of two paths (or both), depending on whether the question is about *meaning* or *structure*:

- **Answer path (semantic/RAG):** the question is embedded, relevant chunks are retrieved via **FAISS** vector search, a reranking step reorders candidates by relevance, and the LLM generates a response grounded in the retrieved chunks — with citations back to the source files.
- **Explain path (structural):** the question is resolved against an **AST/dependency-parsed** representation of the codebase (via Tree-sitter), which can answer questions embeddings can't — call sites, import relationships, definition locations — deterministically, from the actual code structure rather than from vector proximity.

Supporting the two reasoning paths:

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/), serving the planner/agent pipeline over HTTP.
- **Frontend:** a Next.js app (`codementor-ui/`) that talks to the FastAPI backend.
- **Persistence:** local indexes and pipeline state are persisted to disk (mounted as a volume in Docker: `./.codeatlas` → `/app/.codeatlas`), so a repo doesn't need to be re-ingested and re-embedded on every run.
- **Observability:** the service exposes metrics for **Prometheus**, visualized in **Grafana** (`ops/`), so retrieval latency, planner routing behavior, and error rates are visible in production, not just in a terminal.
- **CI:** GitHub Actions (`.github/workflows/`) runs the test suite on every change.

**Tech stack**

| Layer | Tools |
|---|---|
| Orchestration | LangGraph, LangChain |
| Retrieval | FAISS, sentence-transformer / OpenAI embeddings, reranking |
| Structural analysis | Tree-sitter (AST parsing), dependency graph analysis (NetworkX) |
| LLM | OpenAI (configurable provider) |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Next.js (`codementor-ui/`) |
| Observability | Prometheus, Grafana |
| CI/CD | GitHub Actions |
| Deployment | Render (backend), Vercel (frontend) |

---

## Evaluation

Evaluation isn't an afterthought here — it's the mechanism that found the project's core architectural flaw, and it's what keeps that flaw from coming back.

- A dedicated `tests/` suite includes an evaluation set of natural-language questions against known-correct answers, spanning both semantic questions ("what does this module do?") and structural ones ("where is this function called?").
- The cases that exposed the original semantic-only pipeline's blind spot on structural questions are kept as **permanent regression coverage** — any future change that reintroduces that failure mode fails the suite immediately.
- Evals gate changes: the GitHub Actions CI pipeline runs the suite on every push, so a change that looks fine in a quick manual check but breaks retrieval quality or structural accuracy doesn't get to ship silently.

The underlying belief: a codebase-Q&A tool that hasn't been measured against real questions with known answers is a demo, not a system. CodeMentor AI is built to be the latter.

---

## Engineering quality

Beyond the eval story, a few things are worth noting as signals this was built to be maintained, not just demoed:

- **Structured, MVC-style layout** — routing/API layer, agent/orchestration logic, and retrieval/structural services are separated, not tangled into one script.
- **SOLID principles / dependency inversion** — components like the LLM provider, the embedding provider, and the retrieval backend are swappable via configuration (`CODEATLAS_LLM_PROVIDER`, `CODEATLAS_EMBEDDING_PROVIDER`) rather than hard-coded.
- **CI on every change** — via GitHub Actions, so regressions (including the structural-reasoning ones described above) are caught before merge, not after deploy.
- **Containerized and reproducible** — a single Dockerfile and docker-compose setup bring up the full service (and, optionally, the full monitoring stack) with no manual setup steps.

---

## Quick start

### Prerequisites
- Python 3.11
- An OpenAI API key (or Groq key, depending on provider config)

### Local install

```bash
git clone https://github.com/shruti25838/CodeMentor-AI.git
cd CodeMentor-AI

pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and/or GROQ_API_KEY)

uvicorn codeatlas.app.main:app --reload --port 8000
```

The API is now running at `http://localhost:8000`.

### Docker

```bash
docker compose up --build
```

This builds the image and starts the service on port `8000`, with local indexes/state persisted to `./.codeatlas` on the host.

### With monitoring (Prometheus + Grafana)

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

This starts the service, plus:
- **Prometheus** on `http://localhost:9090` (config: `ops/prometheus.yml`)
- **Grafana** on `http://localhost:3000`

### Environment variables

| Variable | Purpose |
|---|---|
| `CODEATLAS_LLM_PROVIDER` | LLM backend to use (e.g. `openai`) |
| `CODEATLAS_LLM_MODEL` | Model name (e.g. `gpt-4o-mini`) |
| `CODEATLAS_LLM_TEMPERATURE` | Generation temperature |
| `OPENAI_API_KEY` | OpenAI credentials (kept out of version control) |
| `GROQ_API_KEY` | Groq credentials, if used as an alternate provider |
| `CODEATLAS_ALLOWED_ORIGINS` | Comma-separated CORS origins for the frontend |
| `CODEATLAS_INDEX_DIR` | Where vector indexes are persisted (`.codeatlas/indexes` by default) |
| `CODEATLAS_STATE_DIR` | Where pipeline/session state is persisted (`.codeatlas/state` by default) |

---

## Repository layout

```
CodeMentor-AI/
├── codeatlas/            # FastAPI backend — agents, retrieval, structural analysis
├── codementor-ui/        # Next.js frontend
├── ops/                  # Prometheus config and monitoring assets
├── tests/                # Evaluation set + regression tests, gating CI
├── .github/workflows/    # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── docker-compose.monitoring.yml
└── requirements.txt
```

---


