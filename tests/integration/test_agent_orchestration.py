"""Integration tests for the main orchestration module."""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _reload_with_patches():
    """Reload main_agent with heavy dependencies stubbed out."""
    # Reset modules so patches apply during import
    for name in [
        "agents.main_agent",
        "agents.analysis_agent",
        "agents.web_research_agent",
        "agents.credibility_agent",
        "middleware.memory_backend",
    ]:
        sys.modules.pop(name, None)

    patchers = [
        patch("deepagents.graph.create_agent"),
        patch("deepagents.create_deep_agent"),
        patch("langchain_openai.ChatOpenAI"),
    ]

    started = [p.start() for p in patchers]

    started[0].return_value = MagicMock(name="subgraph")  # create_agent
    deep_agent = MagicMock()
    deep_agent.with_config.return_value = "configured-agent"
    started[1].return_value = deep_agent  # create_deep_agent

    import middleware.memory_backend as memory_backend
    importlib.reload(memory_backend)

    import agents.main_agent as main_agent
    importlib.reload(main_agent)

    return main_agent, deep_agent, patchers


@pytest.mark.integration
def test_create_research_agent_uses_recursion_limit():
    main_agent, deep_agent, patchers = _reload_with_patches()

    try:
        result = main_agent.create_research_agent()

        assert deep_agent.with_config.call_count >= 1
        for call in deep_agent.with_config.call_args_list:
            assert call.args[0] == {"recursion_limit": 1000}
        assert result == "configured-agent"
        assert len(main_agent.subagents) == 3
    finally:
        for patcher in patchers:
            patcher.stop()


@pytest.mark.integration
def test_clear_scratchpad_resets_directories(tmp_path, monkeypatch):
    # Reload with patches to avoid side effects
    main_agent, _, patchers = _reload_with_patches()

    try:
        monkeypatch.setattr(main_agent, "SCRATCHPAD_DIR", tmp_path)
        monkeypatch.setattr(main_agent, "SCRATCHPAD_SUBDIRS", ["data", "images"])

        # Seed directories with files
        for subdir in main_agent.SCRATCHPAD_SUBDIRS:
            path = tmp_path / subdir
            path.mkdir(parents=True, exist_ok=True)
            (path / "temp.txt").write_text("junk")

        main_agent.clear_scratchpad()

        for subdir in main_agent.SCRATCHPAD_SUBDIRS:
            path = tmp_path / subdir
            assert path.exists()
            assert not any(path.iterdir())
    finally:
        for patcher in patchers:
            patcher.stop()
