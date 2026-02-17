# CodeMentor AI

CodeMentor AI is a Cursor-like, agentic codebase assistant. It indexes a repository, answers questions with citations, previews files, and visualizes architecture connections.

This repo contains:
- A FastAPI backend (`codeatlas/`)
- A Next.js frontend (`codementor-ui/`)

## Features
- Multi-agent Q&A (planner, retrieval, mentor, memory, validation)
- RAG with citations
- Streaming chat responses (SSE)
- File explorer with file preview
- Architecture flow visualization (layered tech stack view)
- Query history and retrieved-context panel
- Evaluation dashboard (`/eval`) with session metrics

## Local Development

### Backend (FastAPI)
```bash
pip install -r requirements.txt
uvicorn codeatlas.app.main:app --port 8000
```

### Frontend (Next.js)
```bash
cd codementor-ui
npm install
npm run dev
```

Open the UI at `http://localhost:3000` (or the port shown by Next.js).

## Environment Variables

Create a `.env` file in the repo root (do not commit it). Example:

```env
CODEATLAS_LLM_PROVIDER=groq
CODEATLAS_LLM_MODEL=llama-3.1-8b-instant
CODEATLAS_LLM_TEMPERATURE=0.0
GROQ_API_KEY=your_key_here

# Allow the frontend to call the backend (comma-separated)
CODEATLAS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

Frontend API base URL can be set with:
- `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`)

## Deployment (Recommended)
- Frontend: Vercel
- Backend: Render

At a minimum you will set:
- Render: `GROQ_API_KEY`, `CODEATLAS_LLM_PROVIDER`, `CODEATLAS_LLM_MODEL`, `CODEATLAS_ALLOWED_ORIGINS`
- Vercel: `NEXT_PUBLIC_API_URL` pointing to your Render backend URL
