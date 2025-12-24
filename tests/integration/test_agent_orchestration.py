"""Integration tests for the main orchestration module."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _reload_with_patches():
    """Reload main_agent with heavy dependencies stubbed out."""
    # Clear cached modules to force fresh import
    for name in list(sys.modules.keys()):
        if name.startswith(("agents.", "middleware.")):
            sys.modules.pop(name, None)

    # Create mock for sub-agent graphs
    mock_subagent_graph = MagicMock(name="subgraph")

    # Create mock for the main deep agent
    mock_deep_agent = MagicMock()
    mock_configured = MagicMock(name="configured-agent")
    mock_deep_agent.with_config.return_value = mock_configured

    patchers = [
        patch("deepagents.graph.create_agent", return_value=mock_subagent_graph),
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

    return main_agent, mock_deep_agent, mock_configured, patchers


@pytest.mark.integration
def test_main_agent_graph_configured_with_recursion_limit():
    main_agent, mock_deep_agent, mock_configured, patchers = _reload_with_patches()

    try:
        # Verify with_config was called with recursion_limit (may be called multiple times due to module reloading)
        assert mock_deep_agent.with_config.call_count >= 1
        mock_deep_agent.with_config.assert_called_with({"recursion_limit": 1000})
        # The main_agent_graph should be the configured agent
        assert main_agent.main_agent_graph is mock_configured
        # Verify we have 3 subagents defined
        assert len(main_agent.subagents) == 3
    finally:
        for patcher in patchers:
            patcher.stop()
