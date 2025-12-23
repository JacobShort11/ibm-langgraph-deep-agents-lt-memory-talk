"""Integration tests for sub-agent configuration."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _load_main_agent_with_stubs():
    """Reload main_agent with heavy dependencies mocked."""
    for name in [
        "agents.analysis_agent",
        "agents.web_research_agent",
        "agents.credibility_agent",
        "agents.main_agent",
        "middleware.memory_backend",
    ]:
        sys.modules.pop(name, None)

    patchers = [
        patch("deepagents.graph.create_agent"),
        patch("deepagents.create_deep_agent"),
        patch("langchain_openai.ChatOpenAI"),
    ]
    started = [p.start() for p in patchers]

    started[0].side_effect = [
        MagicMock(name="analysis-graph"),
        MagicMock(name="web-graph"),
        MagicMock(name="credibility-graph"),
    ]
    deep_agent = MagicMock()
    deep_agent.with_config.return_value = "configured"
    started[1].return_value = deep_agent

    import middleware.memory_backend as memory_backend
    importlib.reload(memory_backend)
    import agents.main_agent as main_agent
    importlib.reload(main_agent)

    return main_agent, patchers


@pytest.mark.integration
def test_subagent_entries_have_expected_shape():
    main_agent, patchers = _load_main_agent_with_stubs()

    try:
        assert len(main_agent.subagents) == 3
        names = {s["name"] for s in main_agent.subagents}
        assert names == {"analysis-agent", "web-research-agent", "credibility-agent"}

        for sub in main_agent.subagents:
            assert sub["description"]
            assert callable(sub["runnable"])
    finally:
        for patcher in patchers:
            patcher.stop()


@pytest.mark.integration
def test_subagent_runnables_match_graphs():
    main_agent, patchers = _load_main_agent_with_stubs()

    try:
        runnable_ids = {id(sub["runnable"]) for sub in main_agent.subagents}
        assert len(runnable_ids) == 3
        assert all(callable(sub["runnable"]) for sub in main_agent.subagents)
    finally:
        for patcher in patchers:
            patcher.stop()
