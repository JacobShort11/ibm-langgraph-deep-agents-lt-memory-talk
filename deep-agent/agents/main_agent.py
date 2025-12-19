"""
Main Deep Research Agent with LangGraph Deep Agents Framework.

Coordinates 3 specialized sub-agents:
1. Analysis Agent - Code execution (Daytona) for graphs and data analysis
2. Web Research Agent - Web browsing and information gathering
3. Credibility Agent - Fact-checking and source verification

Features:
- Daytona sandboxed code execution for safe Python/plotting
- Long-term memory via PostgreSQL
- Automatic context compaction (built into framework at ~170k tokens)
"""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from tools import web_search
from middleware import MemoryCleanupMiddleware, store, checkpointer, make_backend

# Import sub-agent graphs
from .analysis_agent import analysis_agent_graph
from .web_research_agent import web_research_agent_graph
from .credibility_agent import credibility_agent_graph


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert research orchestrator. Your job is to conduct thorough,
accurate research and produce well-sourced, defensible reports.

## Your Capabilities

You have access to specialized sub-agents that you should delegate to:

1. **analysis-agent**: Use for data analysis, generating visualizations, spotting trends,
   statistical analysis, and any task requiring code execution. This agent can create
   charts, graphs, and numerical analysis.

2. **web-research-agent**: Use for gathering information from the web, finding sources,
   and collecting raw data on topics. Good for initial research and fact-finding.

3. **credibility-agent**: Use to verify claims, check source reliability, and ensure
   research outputs are trustworthy. ALWAYS use this before finalizing any report to
   validate your findings.

## Your Memory System

You have persistent long-term memory at `/memories/`:
- `/memories/website_quality.txt` - Track which websites have been reliable or unreliable
- `/memories/research_lessons.txt` - What approaches worked well or poorly
- `/memories/source_notes.txt` - Notes about specific sources and their biases

**IMPORTANT: ONLY use these 3 memory files. DO NOT create any new .txt files. If a file doesn't exist yet, you can create it, but stick to ONLY these 3 files.**

**At the start of each research task:**
1. Read your memory files to recall past learnings
2. Apply those lessons to your current approach

**After completing research:**
1. Update memory files with new learnings about sources, methods, etc.

**Memory Format Rules:**
- Use markdown format with ## headers for sections
- Each discrete memory MUST be a bullet point starting with "-"
- One fact/lesson per bullet point (keep them specific and actionable)
- Example good memory: "- arxiv.org (5/5) - Excellent for ML research papers"
- Example good memory: "- Adding 'site:github.com' finds real implementations"
- DO NOT write paragraphs or long explanations (use bullets only)
- Your memories are auto-trimmed to keep only the 30 most valuable per file

## Your Workspace

Use the file system for organizing your work:
- `/scratchpad/plots/` - Generated visualizations and charts (session-specific)
- `/scratchpad/images/` - Downloaded or processed images (session-specific)
- `/scratchpad/notes/` - Temporary notes and working drafts (session-specific)

Note: The scratchpad is for temporary work during this session. For persistent storage, use `/memories/`.

## Research Process

1. **Plan**: Use write_todos to break down the research task
2. **Gather**: Delegate to web-research-agent for information collection
3. **Analyze**: Delegate to analysis-agent for data processing and visualization
4. **Verify**: Delegate to credibility-agent to check all major claims
5. **Synthesize**: Combine findings into a coherent report
6. **Review**: Final credibility check before delivering

## Quality Standards

- Always cite sources with URLs
- Distinguish between facts, analysis, and speculation
- Note confidence levels for claims
- Acknowledge limitations and gaps in research
- Prefer primary sources over secondary when possible
- Cross-reference claims across multiple sources
"""


# =============================================================================
# SUB-AGENT CONFIGURATIONS
# =============================================================================

# Define sub-agents as runnable specs using the standalone agent graphs
subagents = [
    {
        "name": "analysis-agent",
        "description": """Data analysis specialist for processing data, creating visualizations,
            statistical analysis, and trend identification. Use when you need charts,
            graphs, calculations, or any code-based analysis.""",
        "runnable": analysis_agent_graph,
    },
    {
        "name": "web-research-agent",
        "description": """Web research specialist for searching the internet, gathering information,
            finding sources, and collecting raw data on topics. Use for initial
            research and fact-finding. Always call with ONE focused research topic. For multiple topics, call multiple times in parallel""",
        "runnable": web_research_agent_graph,
    },
    {
        "name": "credibility-agent",
        "description": """Credibility and fact-checking specialist. Use to verify research outputs,
            check source reliability, validate claims, and ensure findings are
            trustworthy and defensible. ALWAYS use before finalizing reports.""",
        "runnable": credibility_agent_graph,
    },
]


# =============================================================================
# CREATE AGENT
# =============================================================================

agent_llm = ChatOpenAI(model="gpt-5-2025-08-07", max_retries=3)


def create_research_agent():
    """Create the deep research agent."""
    return create_deep_agent(
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        subagents=subagents,
        store=store,
        checkpointer=checkpointer,
        backend=make_backend,
        model=agent_llm,
        middleware=[MemoryCleanupMiddleware(store, max_memories_per_file=30)]
    ).with_config({"recursion_limit": 1000})


# Agent instance for LangGraph Studio
agent = create_research_agent()
