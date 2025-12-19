"""
Deep Research Agent with LangGraph Deep Agents Framework

A research agent with 3 specialized sub-agents:
1. Analysis Agent - Code execution (Daytona) for graphs and data analysis
2. Web Research Agent - Web browsing and information gathering  
3. Credibility Agent - Fact-checking and source verification

Features:
- Daytona sandboxed code execution for safe Python/plotting
- Long-term memory via PostgreSQL
- Automatic context compaction (built into framework at ~170k tokens)
"""

import os
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from daytona_sdk import Daytona
from deepagents import create_deep_agent, SubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware, AgentMiddleware
from langchain_openai import ChatOpenAI

from prompts.main_agent import PROMPT as MAIN_AGENT_SYSTEM_PROMPT
from prompts.analysis_agent import PROMPT as ANALYSIS_AGENT_SYSTEM_PROMPT
from prompts.web_research_agent import PROMPT as WEB_RESEARCH_AGENT_SYSTEM_PROMPT
from prompts.credibility_agent import PROMPT as CREDIBILITY_AGENT_SYSTEM_PROMPT
from prompts.trim_middleware import PROMPT as TRIM_SYSTEM_PROMPT

load_dotenv()


# =============================================================================
# DATABASE (Required)
# =============================================================================

DATABASE_URL = os.environ["DATABASE_URL"]
store = PostgresStore.from_conn_string(DATABASE_URL)
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)


# =============================================================================
# TOOLS
# =============================================================================

# Web Search
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> dict:
    """
    Search the web for information.
    
    Args:
        query: Search query string
        max_results: Max results to return (default 5)
        topic: 'general', 'news', or 'finance'
    """
    return tavily_client.search(query, max_results=max_results, topic=topic)


# Code Execution (Daytona)
daytona = Daytona()


def execute_python_code(code: str) -> str:
    """
    Execute Python code in a Daytona sandbox for data analysis and visualization.
    
    Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn
    
    To save plots, use: plt.savefig('/home/daytona/outputs/chart.png')
    Then read the file to get the plot data.
    
    Args:
        code: Python code to execute
        
    Returns:
        Execution output and any generated file paths
    """
    # Create a sandbox
    sandbox = daytona.create()
    
    try:
        # Setup code with common imports
        setup = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('/home/daytona/outputs', exist_ok=True)
"""
        # Run setup + user code
        response = sandbox.process.code_run(setup + "\n" + code)
        
        output_parts = []
        
        # If the code does print("hello") → that goes into response.result
        if response.result:
            output_parts.append(f"Output:\n{response.result}")
        
        # Check for generated files
        # If the code does plt.savefig("/home/daytona/outputs/chart.png") → that creates a file that gets detected by the list_files check
        try:
            files = sandbox.fs.list_files("/home/daytona/outputs")
            if files:
                output_parts.append(f"Generated files: {', '.join(files)}")
        except Exception:
            pass
        
        return "\n\n".join(output_parts) if output_parts else "Code executed successfully"
        
    finally:
        daytona.remove(sandbox)



# =============================================================================
# BACKEND (Ephemeral + Persistent Memory)
# =============================================================================

def make_backend(runtime):
    """Persistent for /memories/, ephemeral for everything else."""
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )



# =============================================================================
# SUB-AGENT CONFIGURATIONS
# =============================================================================

sub_agent_llm = ChatOpenAI(model="gpt-5.1-2025-11-13", max_retries=3)

# Shared middleware for all sub-agents (in addition to auto-added middleware from create_deep_agent)
sub_agent_middleware = [
    ToolCallLimitMiddleware(run_limit=15),
]

# Define sub-agents using SubAgent objects
# create_deep_agent will automatically add: TodoListMiddleware, FilesystemMiddleware, SummarizationMiddleware
subagents = [
    SubAgent(
        name="analysis-agent",
        description="""Data analysis specialist for processing data, creating visualizations,
statistical analysis, and trend identification. Use when you need charts,
graphs, calculations, or any code-based analysis.""",
        system_prompt=ANALYSIS_AGENT_SYSTEM_PROMPT,
        tools=[execute_python_code],
        model=sub_agent_llm,
        middleware=sub_agent_middleware,
    ),
    SubAgent(
        name="web-research-agent",
        description="""Web research specialist for searching the internet, gathering information,
finding sources, and collecting raw data on topics. Use for initial
research and fact-finding. Always call with ONE focused research topic. For multiple topics, call multiple times in parallel""",
        system_prompt=WEB_RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=[web_search],
        model=sub_agent_llm,
        middleware=sub_agent_middleware,
    ),
    SubAgent(
        name="credibility-agent",
        description="""Credibility and fact-checking specialist. Use to verify research outputs,
check source reliability, validate claims, and ensure findings are
trustworthy and defensible. ALWAYS use before finalizing reports.""",
        system_prompt=CREDIBILITY_AGENT_SYSTEM_PROMPT,
        tools=[web_search],
        model=sub_agent_llm,
        middleware=sub_agent_middleware,
    ),
]



# =============================================================================
# MEMORY MANAGEMENT MIDDLEWARE
# =============================================================================

class MemoryCleanupMiddleware(AgentMiddleware):
    """LLM-based memory trimmer that keeps only the best N memories per .txt file."""

    def __init__(self, store_instance, max_memories_per_file: int = 30, cleanup_model: str = "gpt-4o-mini"):
        self.max_memories = max_memories_per_file
        self.store = store_instance
        self.llm = ChatOpenAI(model=cleanup_model, temperature=0)

    def after_agent(self, state, run, tool_calls):
        """Trim all .txt memory files after each agent run."""
        try:
            # Find all .txt files in /memories/
            all_items = list(self.store.search(("filesystem",)))
            txt_files = [item for item in all_items if item.key.startswith("/memories/") and item.key.endswith(".txt")]

            for txt_file in txt_files:
                self._trim_file(txt_file)
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")

        return state, run, tool_calls

    def _trim_file(self, file_item):
        """Trim a single .txt file using LLM."""
        try:
            # Get current content
            content_lines = file_item.value.get("content", [])
            current_content = "\n".join(content_lines) if isinstance(content_lines, list) else str(content_lines)

            # Count memories (each bullet point = 1 memory)
            memory_count = current_content.count("\n- ")

            # Skip if already small enough
            if memory_count <= self.max_memories:
                return

            # Format the prompt with runtime values
            prompt = TRIM_SYSTEM_PROMPT.format(
                max_memories=self.max_memories,
                file_key=file_item.key,
                current_content=current_content
            )

            response = self.llm.invoke(prompt)
            trimmed = response.content.strip()

            # Remove markdown code blocks if present (we do not want this)
            if "```" in trimmed:
                trimmed = trimmed.replace("```markdown", "").replace("```", "").strip()

            # Save trimmed version
            self.store.put(
                ("filesystem",),
                file_item.key,
                {
                    "content": trimmed.split("\n"),
                    "created_at": file_item.value.get("created_at", datetime.now().isoformat()),
                    "modified_at": datetime.now().isoformat(),
                }
            )

            print(f"🧹 Trimmed {file_item.key}: {memory_count} → {self.max_memories} memories")

        except Exception as e:
            print(f"⚠️ Failed to trim {file_item.key}: {e}")


# =============================================================================
# CREATE AGENT
# =============================================================================

agent_llm = ChatOpenAI(model="gpt-5-2025-08-07", max_retries=3)

def create_research_agent():
    """Create the deep research agent."""
    return create_deep_agent(
        tools=[web_search],
        system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
        subagents=subagents,
        store=store,
        checkpointer=checkpointer,
        backend=make_backend,
        model=agent_llm,
        middleware=[MemoryCleanupMiddleware(store, max_memories_per_file=30)]
    ).with_config({"recursion_limit": 1000})


# Agent instance for LangGraph Studio
agent = create_research_agent()