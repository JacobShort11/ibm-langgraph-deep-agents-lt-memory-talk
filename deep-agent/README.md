# Deep Research Agent

A LangGraph Deep Agents project for markets research with dedicated sub-agents for analysis, web research, and credibility checking.

## Features
- Three specialized sub-agents: Analysis (Daytona Python), Web Research (Tavily search), Credibility (fact-checking).
- Cloud-hosted memory and checkpointer via the LangGraph server (same store as LangSmith Studio) — no local PostgreSQL required.
- Automatic context compaction (~170k tokens) built into the framework.
- Organized scratchpad for datasets, notes, and reports.

## Quick Start
1. **Install**
   ```bash
   cd deep-agent
   pip install -r ../requirements.txt
   ```
2. **Configure `.env`** (no `DATABASE_URL` needed)
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `DAYTONA_API_KEY`
   - `LANGSMITH_API_KEY`
   - `LANGSMITH_PROJECT`
   - `LANGSMITH_TRACING` (true/false)
   - Plot uploads (Cloudinary): `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` (or `CLOUDINARY_UPLOAD_PRESET` for unsigned), optional `CLOUDINARY_PUBLIC_ID_PREFIX` (default plots/)
3. **Run the LangGraph server** (from `deep-agent/`)
   ```bash
   langgraph dev
   ```
   Wait for `http://localhost:2024` to be ready.
4. **Use the notebooks** in `experiments/`
   - `analysis_agent.ipynb`
   - `web_research_agent.ipynb`
   - `credibility_agent.ipynb`
   - `memory_management.ipynb`
   - `cloudinary_upload_smoke_test.ipynb`

   Each notebook connects to the LangGraph server, auto-discovers the `assistant_id`, and talks to the same LangSmith deployment used by Studio.

### Running a single agent (analysis-only)
- `langgraph.json` now exports the analysis agent as its own graph (`analysis-agent`) alongside the full orchestrator.
- Run `langgraph dev` from `deep-agent/` and the server will register both assistants in LangSmith using the same cloud store/checkpointer.
- In `experiments/analysis_agent.ipynb`, set `ANALYSIS_ASSISTANT_ID` (or leave the default `ANALYSIS_ASSISTANT_NAME=analysis`) and it will bind directly to the analysis agent without the orchestrator.
- If you want only the analysis agent running, you can start just that graph: `langgraph dev --graph agents/analysis_agent.py:analysis_agent_graph --assistant-id analysis-agent --name analysis-agent`.

## Memory Model
- Persistent store is provided by LangGraph (LangSmith cloud), namespaced per `assistant_id`.
- Files under `/memories/` include:
  - `website_quality.txt` — Source ratings
  - `research_lessons.txt` — What works
  - `source_notes.txt` — Source notes
  - `coding.txt` — Code mistakes and lessons
- No local `DATABASE_URL` or Postgres container is used.

## Layout
```
agents/           # Main agent and sub-agents
middleware/       # Memory backend configuration
experiments/      # Jupyter notebooks for testing agents and memory
scratchpad/       # Data, notes, and reports written by agents
langgraph.json    # LangGraph graph definition
```

## Troubleshooting
- Connection errors: ensure `langgraph dev` is running from `deep-agent/` and reachable at `http://localhost:2024`.
- Empty assistant list: start `langgraph dev` once to register the graph with LangSmith.
