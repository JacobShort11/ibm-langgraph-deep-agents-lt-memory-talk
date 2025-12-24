# Deep Research Agent

A LangGraph Deep Agents project for markets research with dedicated sub-agents for analysis, web research, and credibility checking.

## Features
- Three specialized sub-agents: Analysis (Daytona Python), Web Research (Tavily search), Credibility (fact-checking).
- Cloud-hosted memory and checkpointer via the LangGraph server (same store as LangSmith Studio) — no local PostgreSQL required.
- Automatic context compaction (~170k tokens) built into the framework.
- Organized scratchpad for notes and reports.

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
   - Plot uploads (Cloudinary): `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` (or `CLOUDINARY_UPLOAD_PRESET` for unsigned), optional `CLOUDINARY_PUBLIC_ID_PREFIX` (default plots/). Alternatively, set `CLOUDINARY_URL` (e.g., `cloudinary://<api_key>:<api_secret>@<cloud_name>`).
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
   - `md_to_pdf.ipynb`

   Each notebook connects to the LangGraph server, auto-discovers the `assistant_id`, and talks to the same LangSmith deployment used by Studio.

### Running individual agents
- `langgraph.json` exports all agents as standalone graphs:
  - `main-agent` — Full orchestrator
  - `analysis-agent` — Data analysis and visualization
  - `web-research-agent` — Web search and information gathering
  - `credibility-agent` — Fact-checking and source verification
- Run `langgraph dev` from `deep-agent/` and the server will register all assistants in LangSmith using the same cloud store/checkpointer.
- Each notebook in `experiments/` can bind directly to its corresponding agent without the orchestrator.
- To start only a specific agent: `langgraph dev --graph agents/<agent_file>.py:<graph_name> --assistant-id <agent-name> --name <agent-name>`.

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
tools/            # Code execution (Daytona) and web search tools
middleware/       # Memory backend configuration
experiments/      # Jupyter notebooks for testing agents and memory
scratchpad/       # Notes and reports written by agents
langgraph.json    # LangGraph graph definition
```

## Troubleshooting
- Connection errors: ensure `langgraph dev` is running from `deep-agent/` and reachable at `http://localhost:2024`.
- Empty assistant list: start `langgraph dev` once to register the graph with LangSmith.
