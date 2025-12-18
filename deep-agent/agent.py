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
from dotenv import load_dotenv
from tavily import TavilyClient
from daytona_sdk import Daytona
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware

from prompts.main_agent import PROMPT as MAIN_AGENT_SYSTEM_PROMPT
from prompts.analysis_agent import PROMPT as ANALYSIS_AGENT_SYSTEM_PROMPT
from prompts.web_research_agent import PROMPT as WEB_RESEARCH_AGENT_SYSTEM_PROMPT
from prompts.credibility_agent import PROMPT as CREDIBILITY_AGENT_SYSTEM_PROMPT

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
        
        if response.result:
            output_parts.append(f"Output:\n{response.result}")
        
        # Check for generated files
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
# SUB-AGENT CONFIGURATIONS
# =============================================================================

sub_agent_llm = ChatOpenAI(model="gpt-5.1-2025-11-13", max_retries=3)

common_middleware = [
    SummarizationMiddleware(
        model=sub_agent_llm,
        max_tokens_before_summary=120000,
        messages_to_keep=20,
    ),
    ToolCallLimitMiddleware(run_limit=15),
]


subagents = [
    {
        "name": "analysis-agent",
        "description": """Data analysis specialist for processing data, creating visualizations,
statistical analysis, and trend identification. Use when you need charts,
graphs, calculations, or any code-based analysis.""",
        "system_prompt": ANALYSIS_AGENT_SYSTEM_PROMPT,
        "tools": [execute_python_code],
        "middleware": common_middleware,
        "model": sub_agent_llm
    },
    {
        "name": "web-research-agent",
        "description": """Web research specialist for searching the internet, gathering information,
finding sources, and collecting raw data on topics. Use for initial
research and fact-finding. Always call with ONE focused research topic. For multiple topics, call multiple times in parallel""",
        "system_prompt": WEB_RESEARCH_AGENT_SYSTEM_PROMPT,
        "tools": [web_search],
        "middleware": common_middleware,
        "model": sub_agent_llm
    },
    {
        "name": "credibility-agent",
        "description": """Credibility and fact-checking specialist. Use to verify research outputs,
check source reliability, validate claims, and ensure findings are
trustworthy and defensible. ALWAYS use before finalizing reports.""",
        "system_prompt": CREDIBILITY_AGENT_SYSTEM_PROMPT,
        "tools": [web_search],
        "middleware": common_middleware,
        "model": sub_agent_llm
    }
]

# =============================================================================
# BACKEND (Ephemeral + Persistent Memory)
# =============================================================================

def make_backend(runtime):
    """Ephemeral storage by default, persistent for /memories/"""
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


# =============================================================================
# CREATE AGENT
# =============================================================================

agent_llm = ChatOpenAI(model="gpt-5-2025-08-07", max_retries=3)

def create_research_agent():
    """Create the deep research agent."""
    return create_deep_agent(
        tools=[web_search],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        store=store,
        checkpointer=checkpointer,
        backend=make_backend,
        model=agent_llm
    ).with_config({"recursion_limit": 1000})


# Agent instance for LangGraph Studio
agent = create_research_agent()