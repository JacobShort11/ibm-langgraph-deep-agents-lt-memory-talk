# Deep Research Agent

A research agent built with LangGraph's Deep Agents framework. Orchestrates specialized sub-agents for thorough, credible research.

## Features

- **3 Specialized Sub-Agents**:
  - **Analysis Agent** - Daytona sandboxed Python for data analysis and visualizations
  - **Web Research Agent** - Web searching via Tavily
  - **Credibility Agent** - Fact-checking and source verification

- **Long-Term Memory** - PostgreSQL-backed persistent memory
- **Automatic Context Compaction** - Built into framework (~170k tokens)
- **File System** - Organized workspace for research outputs

## Quick Start
- Create a Python 3.12 virtual environment

### 1. Install

```bash
cd deep-agent
pip install -r ../requirements.txt
```

### 2. Configure

Setup your .env file containing:
- `ANTHROPIC_API_KEY` - Claude models
- `TAVILY_API_KEY` - Web search
- `DAYTONA_API_KEY` - Code execution sandbox
- `DATABASE_URL` - PostgreSQL connection

### 3. Start PostgreSQL

Docker runs PostgreSQL in an isolated container (no system installation needed). Data persists between sessions in Docker volumes.

Ensure Docker is running first: `colima start` or open Docker Desktop.

**First time setup** (run from any directory):
```bash
docker run -d \
  --name deep-agent-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=deep_agent \
  -p 5432:5432 \
  postgres:15
```

**Subsequent sessions** (if container is stopped):
```bash
docker start deep-agent-db
```

**Stop container** (optional - uses minimal resources, safe to leave running):
```bash
docker stop deep-agent-db
```

Check if running: `docker ps`

### 4. Launch

```bash
langgraph dev
```

Opens LangGraph Studio in your browser. We can run our agent from here.

## Project Structure

```
deep_research_agent/
├── .env.example             # Example environment template
├── agent.py                 # Main agent definition
├── langgraph.json           # LangGraph Studio configuration
├── memory_management.ipynb  # Notebook for artificial memory management
├── README.md                # This file
```

## Architecture

TO DO (INSERT IMAGE)

### Built-in Tools (from LangGraph Deep Agents framework)

- `write_todos` - Planning and task management.
- `ls`, `glob`, `grep` - File system navigation
- `read_file`, `write_file`, `edit_file` - File operations
- `task` - Spawn sub-agents
Note: there is no `read_todos` tool. Instead LangGraph Deep Agents' assumes that we will not overflow context losing this (this is a slight oversight in the framework). Instead `write_todos` is there just to encourage the agent to follow the planning behaviour. If needed, we could build a `read_todos` tool to retrieve from state incase of loss in future.

### Custom Tools

- `execute_python_code` - Sandboxed code execution (Daytona)
- `web_search` - Search web pages


### Memory

- **Ephemeral**: `/research/`, `/data/`, `/outputs/` (per-session)
- **Persistent**: `/memories/` (PostgreSQL, cross-session)

Long-Term Memory files:
- `/memories/website_quality.txt` - Source ratings
- `/memories/research_lessons.txt` - What works
- `/memories/source_notes.txt` - Source notes
- `/memories/coding.txt` - Code mistakes and lessons (analysis agent only)
In this project, memories are being searched for by `glob`, `grep` instead of retrieved semantically although we could easily create a `semantic_search` tool / middleware if needed.

### Context Compaction

Framework auto-compacts at ~170k tokens:
- Keeps recent messages
- Summarizes older ones
- Large outputs saved to files


## Troubleshooting

### Database connection errors
- Ensure PostgreSQL is running
- Check `DATABASE_URL` format in `.env`

### Sub-agents not being called
- Check the sub-agent descriptions are clear about when to use them
- Add reminders in the main agent's system prompt
