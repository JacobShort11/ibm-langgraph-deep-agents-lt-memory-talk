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
# SYSTEM PROMPTS
# =============================================================================

MAIN_AGENT_SYSTEM_PROMPT = """You are an expert research orchestrator. Your job is to conduct thorough, 
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

**At the start of each research task:**
1. Read your memory files to recall past learnings
2. Apply those lessons to your current approach

**After completing research:**
1. Update memory files with new learnings about sources, methods, etc.

## Your Working Files

Use the file system for organizing your work:
- `/research/` - Current research notes and drafts
- `/outputs/` - Final deliverables for the user
- `/data/` - Raw data and intermediate analysis

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

ANALYSIS_AGENT_PROMPT = """You are a data analysis specialist. Your role is to:

1. **Analyze Data**: Process datasets, identify patterns, calculate statistics
2. **Create Visualizations**: Generate clear, informative charts and graphs
3. **Spot Trends**: Identify meaningful trends and anomalies in data
4. **Statistical Analysis**: Perform appropriate statistical tests

## Tools Available

You have `execute_python_code` for running Python with:
- pandas, numpy for data manipulation
- matplotlib, seaborn, plotly for visualization  
- scipy, sklearn for statistical analysis

## Best Practices

- Always explain your analysis approach before running code
- Include error handling in your code
- Save visualizations to files (e.g., `/outputs/chart.png`)
- Provide clear interpretation of results
- Note any data quality issues or limitations

## Output Format

When you complete analysis, return:
1. Key findings (3-5 bullet points)
2. Visualizations created (with file paths)
3. Confidence level in the analysis
4. Any caveats or limitations

Keep your response focused - save detailed data to files if needed.
"""

WEB_RESEARCH_AGENT_PROMPT = """You are a web research specialist. Your role is to:

1. **Find Information**: Search for relevant, reliable sources
2. **Gather Data**: Collect facts, statistics, and quotes
3. **Document Sources**: Keep detailed records of where information came from
4. **Assess Initial Quality**: Note if sources seem reliable or questionable

## Tools Available

You have `web_search` for searching the web with options for:
- General, news, or finance-focused searches
- Configurable result count
- Optional raw page content

## Research Strategy

1. Start with broad queries to understand the landscape
2. Refine with specific queries for details
3. Search for counter-arguments and alternative viewpoints
4. Look for primary sources (original research, official reports)

## Best Practices

- Use multiple search queries to triangulate information
- Note the date of sources (recency matters)
- Distinguish between news, opinion, and research
- Save raw search results to files for later reference
- Flag any sources that seem unreliable

## Output Format

Return your findings as:
1. Summary of what you found
2. Key facts with source URLs
3. List of sources used with brief reliability notes
4. Gaps in the research (what you couldn't find)

Keep responses concise - save detailed notes to `/research/` files.
"""

CREDIBILITY_AGENT_PROMPT = """You are a credibility and fact-checking specialist. Your role is to:

1. **Verify Claims**: Check if claims are supported by reliable evidence
2. **Assess Sources**: Evaluate the trustworthiness of sources used
3. **Check Consistency**: Look for contradictions or inconsistencies
4. **Identify Bias**: Note potential biases in sources or analysis

## Your Task

When given research outputs to review, you should:

1. Read the content carefully
2. Identify major claims that need verification
3. Use web_search to find corroborating or contradicting evidence
4. Assess whether the research answers the original question
5. Rate overall trustworthiness and defensibility

## Credibility Criteria

**High Credibility Sources:**
- Peer-reviewed research
- Official government/institutional data
- Established news organizations (for current events)
- Primary sources and original documents

**Lower Credibility Sources:**
- Blogs and opinion pieces (unless expert)
- Social media
- Sites with heavy advertising
- Sources with clear conflicts of interest

## Output Format

Provide your assessment as:

### Claim Verification
- [Claim 1]: VERIFIED / PARTIALLY VERIFIED / UNVERIFIED / CONTRADICTED
  - Evidence: [brief explanation]
- [Claim 2]: ...

### Source Assessment  
- Overall source quality: [1-5 rating]
- Concerns: [any issues found]

### Answer Quality
- Does it answer the original question? [Yes/Partially/No]
- Missing elements: [what's missing]

### Recommendations
- Suggested corrections or additions
- Areas needing more research

### Final Verdict
- Trustworthy and defensible? [Yes/With caveats/Needs work]
"""


# =============================================================================
# SUB-AGENT CONFIGURATIONS
# =============================================================================

subagents = [
    {
        "name": "analysis-agent",
        "description": """Data analysis specialist for processing data, creating visualizations,
statistical analysis, and trend identification. Use when you need charts,
graphs, calculations, or any code-based analysis.""",
        "system_prompt": ANALYSIS_AGENT_PROMPT,
        "tools": [execute_python_code],
    },
    {
        "name": "web-research-agent",
        "description": """Web research specialist for searching the internet, gathering information,
finding sources, and collecting raw data on topics. Use for initial
research and fact-finding.""",
        "system_prompt": WEB_RESEARCH_AGENT_PROMPT,
        "tools": [web_search],
    },
    {
        "name": "credibility-agent",
        "description": """Credibility and fact-checking specialist. Use to verify research outputs,
check source reliability, validate claims, and ensure findings are
trustworthy and defensible. ALWAYS use before finalizing reports.""",
        "system_prompt": CREDIBILITY_AGENT_PROMPT,
        "tools": [web_search],
    },
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

def create_research_agent():
    """Create the deep research agent."""
    return create_deep_agent(
        tools=[web_search],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        store=store,
        checkpointer=checkpointer,
        backend=make_backend,
    )


# Agent instance for LangGraph Studio
agent = create_research_agent()


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import uuid
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Deep Research Agent")
    print(f"Thread: {thread_id[:8]}...")
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the latest AI agent trends?"}]},
        config=config
    )
    
    print("\nResponse:")
    print(result["messages"][-1].content)
