"""Integration tests for sub-agent configuration."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _load_main_agent_with_stubs():
    """Reload main_agent with heavy dependencies mocked."""
    # Clear cached modules to force fresh import
    for name in list(sys.modules.keys()):
        if name.startswith(("agents.", "middleware.")):
            sys.modules.pop(name, None)

    # Track sub-agent graphs as they're created
    subagent_graphs = []

    def create_agent_side_effect(*args, **kwargs):
        graph = MagicMock(name=f"subgraph-{len(subagent_graphs)}")
        subagent_graphs.append(graph)
        return graph

    # Create mock for the main deep agent
    mock_deep_agent = MagicMock()
    mock_deep_agent.with_config.return_value = MagicMock(name="configured")

    patchers = [
        patch("deepagents.graph.create_agent", side_effect=create_agent_side_effect),
        patch("deepagents.create_deep_agent", return_value=mock_deep_agent),
        patch("deepagents.FilesystemMiddleware"),
        patch("langchain.agents.middleware.ToolCallLimitMiddleware"),
        patch("langchain_openai.ChatOpenAI"),
    ]

    for p in patchers:
        p.start()

    # Now import the modules
    import middleware.memory_backend as memory_backend
    importlib.reload(memory_backend)

    import agents.main_agent as main_agent
    importlib.reload(main_agent)

    return main_agent, subagent_graphs, patchers


@pytest.mark.integration
def test_subagent_entries_have_expected_shape():
    main_agent, subagent_graphs, patchers = _load_main_agent_with_stubs()

    try:
        assert len(main_agent.subagents) == 3
        names = {s["name"] for s in main_agent.subagents}
        assert names == {"analysis-agent", "web-research-agent", "credibility-agent"}

        for sub in main_agent.subagents:
            assert sub["description"]
            assert "runnable" in sub
    finally:
        for patcher in patchers:
            patcher.stop()


@pytest.mark.integration
def test_subagent_runnables_are_unique():
    main_agent, subagent_graphs, patchers = _load_main_agent_with_stubs()

    try:
        runnable_ids = {id(sub["runnable"]) for sub in main_agent.subagents}
        # Each subagent should have a unique runnable
        assert len(runnable_ids) == 3
    finally:
        for patcher in patchers:
            patcher.stop()
