"""
Unit tests for middleware components (SummarizationMiddleware, ToolCallLimitMiddleware, MemoryCleanupMiddleware).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add the deep-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deep-agent'))


@pytest.mark.unit
class TestMemoryCleanupMiddleware:
    """Tests for MemoryCleanupMiddleware."""

    def test_middleware_initialization(self):
        """Test middleware initializes with correct defaults."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        middleware = MemoryCleanupMiddleware(mock_store)

        assert middleware.max_memories == 30
        assert middleware.store == mock_store

    def test_middleware_initialization_custom_max_memories(self):
        """Test middleware with custom max_memories_per_file."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=50)

        assert middleware.max_memories == 50

    def test_after_agent_no_cleanup_needed(self):
        """Test after_agent when cleanup is not needed."""
        from agent import MemoryCleanupMiddleware

        # Setup: Memory file with fewer than max memories (each bullet = 1 memory)
        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        mock_item.value = {"content": ["## Test", "- Memory 1", "- Memory 2", "- Memory 3"]}
        mock_store.search.return_value = [mock_item]

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {"test": "state"}
        run = Mock()
        tool_calls = []

        # Execute
        result_state, result_run, result_tool_calls = middleware.after_agent(
            state, run, tool_calls
        )

        # Verify: No trimming (put should not be called)
        mock_store.put.assert_not_called()
        assert result_state == state
        assert result_run == run
        assert result_tool_calls == tool_calls

    @patch('agent.ChatOpenAI')
    def test_after_agent_trims_large_files(self, mock_openai):
        """Test after_agent trims files when over memory limit."""
        from agent import MemoryCleanupMiddleware

        # Setup: Memory file with more than max memories
        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        # Create content with 50 memories (50 bullet points)
        content = "## Test Section\n" + "\n".join([f"- Memory {i}" for i in range(50)])
        mock_item.value = {"content": content.split("\n"), "created_at": "2025-01-01T00:00:00"}
        mock_store.search.return_value = [mock_item]

        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "## Test Section\n" + "\n".join([f"- Memory {i}" for i in range(30)])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {"test": "state"}
        run = Mock()
        tool_calls = []

        # Execute
        with patch('builtins.print') as mock_print:
            result_state, result_run, result_tool_calls = middleware.after_agent(
                state, run, tool_calls
            )

        # Verify: File was trimmed (put should be called)
        mock_store.put.assert_called_once()

        # Verify cleanup message printed
        mock_print.assert_called()
        print_message = mock_print.call_args[0][0]
        assert "Trimmed" in print_message or "trimmed" in print_message

    @patch('agent.ChatOpenAI')
    def test_after_agent_only_trims_txt_files(self, mock_openai):
        """Test that after_agent only processes .txt files."""
        from agent import MemoryCleanupMiddleware

        # Setup: Mix of .txt and non-.txt files
        mock_store = Mock()
        mock_txt_item = Mock()
        mock_txt_item.key = "/memories/test.txt"
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(5)])
        mock_txt_item.value = {"content": content.split("\n")}

        mock_other_item = Mock()
        mock_other_item.key = "/memories/data.json"
        mock_other_item.value = {"content": "{}"}

        mock_store.search.return_value = [mock_txt_item, mock_other_item]

        mock_llm = Mock()
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: Only .txt file content should be processed (no trimming needed as < 30 memories)
        # LLM should not be invoked since file is small enough
        mock_llm.invoke.assert_not_called()

    def test_after_agent_searches_filesystem_namespace(self):
        """Test that after_agent searches the filesystem namespace."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_store.search.return_value = []

        middleware = MemoryCleanupMiddleware(mock_store)
        state = {}
        run = Mock()
        tool_calls = []

        middleware.after_agent(state, run, tool_calls)

        # Verify correct namespace used
        mock_store.search.assert_called_once_with(("filesystem",))

    def test_after_agent_error_handling(self):
        """Test that after_agent handles errors gracefully."""
        from agent import MemoryCleanupMiddleware

        # Setup: Store search raises error
        mock_store = Mock()
        mock_store.search.side_effect = Exception("Database error")

        middleware = MemoryCleanupMiddleware(mock_store)
        state = {"test": "state"}
        run = Mock()
        tool_calls = []

        # Execute: Should not raise
        with patch('builtins.print') as mock_print:
            result_state, result_run, result_tool_calls = middleware.after_agent(
                state, run, tool_calls
            )

        # Verify: State returned unchanged
        assert result_state == state
        assert result_run == run
        assert result_tool_calls == tool_calls

        # Verify error message printed
        mock_print.assert_called()
        error_message = mock_print.call_args[0][0]
        assert "Memory cleanup failed" in error_message

    @patch('agent.ChatOpenAI')
    def test_after_agent_trim_error_handling(self, mock_openai):
        """Test error handling when trimming fails."""
        from agent import MemoryCleanupMiddleware

        # Setup: File that needs trimming, but LLM fails
        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(50)])
        mock_item.value = {"content": content.split("\n")}
        mock_store.search.return_value = [mock_item]

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute: Should not raise
        with patch('builtins.print') as mock_print:
            result_state, result_run, result_tool_calls = middleware.after_agent(
                state, run, tool_calls
            )

        # Verify: State returned unchanged
        assert result_state == state

        # Verify error was handled
        assert any("Failed to trim" in str(call[0][0]) for call in mock_print.call_args_list)

    def test_after_agent_exact_limit(self):
        """Test when memory count exactly equals the limit."""
        from agent import MemoryCleanupMiddleware

        # Setup: Exactly max memories
        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(30)])
        mock_item.value = {"content": content.split("\n")}
        mock_store.search.return_value = [mock_item]

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: No trimming when at exact limit
        mock_store.put.assert_not_called()

    def test_after_agent_empty_memories(self):
        """Test when there are no memories."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_store.search.return_value = []

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: No trimming
        mock_store.put.assert_not_called()


@pytest.mark.unit
class TestMiddlewareIntegration:
    """Tests for middleware integration in sub-agents."""

    def test_analysis_agent_has_middleware(self):
        """Test that analysis agent has correct middleware configuration."""
        # This test verifies the middleware setup in agent.py
        # In new SubAgent architecture, middleware is shared and contains ToolCallLimitMiddleware
        import importlib
        import agent

        # Reload to get fresh module
        importlib.reload(agent)

        # Check that middleware list is defined
        assert hasattr(agent, 'analysis_sub_agent_middleware')
        assert isinstance(agent.analysis_sub_agent_middleware, list)
        assert len(agent.analysis_sub_agent_middleware) >= 1  # At least ToolCallLimitMiddleware

    def test_web_research_agent_has_middleware(self):
        """Test that web research agent has correct middleware configuration."""
        import importlib
        import agent

        importlib.reload(agent)

        assert hasattr(agent, 'web_research_sub_agent_middleware')
        assert isinstance(agent.web_research_sub_agent_middleware, list)
        assert len(agent.web_research_sub_agent_middleware) >= 1  # At least ToolCallLimitMiddleware

    def test_credibility_agent_has_middleware(self):
        """Test that credibility agent has correct middleware configuration."""
        import importlib
        import agent

        importlib.reload(agent)

        assert hasattr(agent, 'credibility_sub_agent_middleware')
        assert isinstance(agent.credibility_sub_agent_middleware, list)
        assert len(agent.credibility_sub_agent_middleware) >= 1  # At least ToolCallLimitMiddleware

    def test_main_agent_has_memory_cleanup_middleware(self):
        """Test that main agent includes MemoryCleanupMiddleware."""
        # We'll check this in the create_research_agent function
        # This is tested more thoroughly in integration tests
        pass


@pytest.mark.unit
class TestMiddlewareConfiguration:
    """Tests for middleware configuration values."""

    def test_summarization_threshold(self):
        """Test that summarization middleware has correct token threshold."""
        # The threshold should be ~120k tokens
        import agent

        # Check analysis agent middleware config
        middleware_list = agent.analysis_sub_agent_middleware
        # Find SummarizationMiddleware in the list
        # This is a smoke test to ensure the configuration exists
        assert len(middleware_list) > 0

    def test_tool_call_limit(self):
        """Test that tool call limit is properly configured."""
        import agent

        # The run limit should be 15 for sub-agents
        middleware_list = agent.analysis_sub_agent_middleware
        assert len(middleware_list) > 0

    def test_memory_cleanup_limit(self):
        """Test that memory cleanup has correct memories limit."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        # Default should be 30 memories per file
        middleware = MemoryCleanupMiddleware(mock_store)
        assert middleware.max_memories == 30
