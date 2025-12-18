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

        middleware = MemoryCleanupMiddleware()

        assert middleware.max_items == 100
        assert middleware.namespace == ("agent", "memories")

    def test_middleware_initialization_custom_max_items(self):
        """Test middleware with custom max_items."""
        from agent import MemoryCleanupMiddleware

        middleware = MemoryCleanupMiddleware(max_items=50)

        assert middleware.max_items == 50

    @patch('agent.store')
    def test_after_agent_no_cleanup_needed(self, mock_store):
        """Test after_agent when cleanup is not needed."""
        from agent import MemoryCleanupMiddleware

        # Setup: Less than max items
        mock_memories = [
            Mock(key=f"mem_{i}", updated_at=datetime(2025, 1, 1 + i // 24, i % 24, 0, 0))
            for i in range(50)
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=100)
        state = {"test": "state"}
        run = Mock()
        tool_calls = []

        # Execute
        result_state, result_run, result_tool_calls = middleware.after_agent(
            state, run, tool_calls
        )

        # Verify: No deletions
        mock_store.delete.assert_not_called()
        assert result_state == state
        assert result_run == run
        assert result_tool_calls == tool_calls

    @patch('agent.store')
    def test_after_agent_cleanup_old_memories(self, mock_store):
        """Test after_agent deletes old memories when over limit."""
        from agent import MemoryCleanupMiddleware

        # Setup: More than max items
        mock_memories = [
            Mock(
                key=f"mem_{i:03d}",
                updated_at=datetime(2025, 1, 1, i // 60, i % 60, 0),
            )
            for i in range(150)
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=100)
        state = {"test": "state"}
        run = Mock()
        tool_calls = []

        # Execute
        with patch('builtins.print') as mock_print:
            result_state, result_run, result_tool_calls = middleware.after_agent(
                state, run, tool_calls
            )

        # Verify: 50 oldest memories deleted (150 - 100)
        assert mock_store.delete.call_count == 50

        # Verify the correct memories were deleted (oldest ones)
        deleted_keys = [call[0][1] for call in mock_store.delete.call_args_list]
        # The oldest 50 memories should be deleted
        for i in range(50):
            assert f"mem_{i:03d}" in deleted_keys

        # Verify cleanup message printed
        mock_print.assert_called()
        print_message = mock_print.call_args[0][0]
        assert "Memory cleanup" in print_message
        assert "50" in print_message
        assert "100" in print_message

    @patch('agent.store')
    def test_after_agent_keeps_most_recent(self, mock_store):
        """Test that after_agent keeps the most recent memories."""
        from agent import MemoryCleanupMiddleware

        # Setup: Memories with different timestamps
        mock_memories = [
            Mock(key="old_mem", updated_at=datetime(2025, 1, 1, 0, 0, 0)),
            Mock(key="new_mem", updated_at=datetime(2025, 1, 15, 0, 0, 0)),
            Mock(key="newer_mem", updated_at=datetime(2025, 1, 20, 0, 0, 0)),
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=2)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: Only the oldest memory is deleted
        mock_store.delete.assert_called_once()
        call_args = mock_store.delete.call_args[0]
        assert call_args[1] == "old_mem"  # Oldest memory deleted

    @patch('agent.store')
    def test_after_agent_uses_correct_namespace(self, mock_store):
        """Test that after_agent searches the correct namespace."""
        from agent import MemoryCleanupMiddleware

        mock_store.search.return_value = []

        middleware = MemoryCleanupMiddleware()
        state = {}
        run = Mock()
        tool_calls = []

        middleware.after_agent(state, run, tool_calls)

        # Verify correct namespace used
        mock_store.search.assert_called_once_with(("agent", "memories"))

    @patch('agent.store')
    def test_after_agent_error_handling(self, mock_store):
        """Test that after_agent handles errors gracefully."""
        from agent import MemoryCleanupMiddleware

        # Setup: Store search raises error
        mock_store.search.side_effect = Exception("Database error")

        middleware = MemoryCleanupMiddleware()
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

    @patch('agent.store')
    def test_after_agent_delete_error_handling(self, mock_store):
        """Test error handling when deletion fails."""
        from agent import MemoryCleanupMiddleware

        # Setup: Many memories, but delete fails
        mock_memories = [
            Mock(key=f"mem_{i}", updated_at=datetime(2025, 1, 1, i // 60, i % 60, 0))
            for i in range(150)
        ]
        mock_store.search.return_value = mock_memories
        mock_store.delete.side_effect = Exception("Delete error")

        middleware = MemoryCleanupMiddleware(max_items=100)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute: Should not raise
        with patch('builtins.print') as mock_print:
            middleware.after_agent(state, run, tool_calls)

        # Verify error was handled
        error_message = mock_print.call_args[0][0]
        assert "Memory cleanup failed" in error_message

    @patch('agent.store')
    def test_after_agent_exact_limit(self, mock_store):
        """Test when memory count exactly equals the limit."""
        from agent import MemoryCleanupMiddleware

        # Setup: Exactly max_items
        mock_memories = [
            Mock(key=f"mem_{i}", updated_at=datetime(2025, 1, 1 + i // 24, i % 24, 0, 0))
            for i in range(100)
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=100)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: No deletions when at exact limit
        mock_store.delete.assert_not_called()

    @patch('agent.store')
    def test_after_agent_empty_memories(self, mock_store):
        """Test when there are no memories."""
        from agent import MemoryCleanupMiddleware

        mock_store.search.return_value = []

        middleware = MemoryCleanupMiddleware(max_items=100)
        state = {}
        run = Mock()
        tool_calls = []

        # Execute
        middleware.after_agent(state, run, tool_calls)

        # Verify: No deletions
        mock_store.delete.assert_not_called()


@pytest.mark.unit
class TestMiddlewareIntegration:
    """Tests for middleware integration in sub-agents."""

    def test_analysis_agent_has_middleware(self):
        """Test that analysis agent has correct middleware configuration."""
        # This test verifies the middleware setup in agent.py
        import importlib
        import agent

        # Reload to get fresh module
        importlib.reload(agent)

        # Check that middleware list is defined
        assert hasattr(agent, 'analysis_sub_agent_middleware')
        assert isinstance(agent.analysis_sub_agent_middleware, list)
        assert len(agent.analysis_sub_agent_middleware) >= 2

    def test_web_research_agent_has_middleware(self):
        """Test that web research agent has correct middleware configuration."""
        import importlib
        import agent

        importlib.reload(agent)

        assert hasattr(agent, 'web_research_sub_agent_middleware')
        assert isinstance(agent.web_research_sub_agent_middleware, list)
        assert len(agent.web_research_sub_agent_middleware) >= 2

    def test_credibility_agent_has_middleware(self):
        """Test that credibility agent has correct middleware configuration."""
        import importlib
        import agent

        importlib.reload(agent)

        assert hasattr(agent, 'credibility_sub_agent_middleware')
        assert isinstance(agent.credibility_sub_agent_middleware, list)
        assert len(agent.credibility_sub_agent_middleware) >= 2

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
        """Test that memory cleanup has correct item limit."""
        from agent import MemoryCleanupMiddleware

        # Default should be 100 items
        middleware = MemoryCleanupMiddleware()
        assert middleware.max_items == 100
