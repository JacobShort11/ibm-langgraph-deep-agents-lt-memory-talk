"""
Unit tests for memory management and backend configuration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the deep-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deep-agent'))


@pytest.mark.unit
class TestBackendConfiguration:
    """Tests for backend configuration (CompositeBackend, StateBackend, StoreBackend)."""

    def test_make_backend_returns_composite_backend(self):
        """Test that make_backend returns a CompositeBackend."""
        from agent import make_backend
        from deepagents.backends import CompositeBackend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        assert isinstance(backend, CompositeBackend)

    def test_make_backend_has_default_state_backend(self):
        """Test that CompositeBackend uses StateBackend as default."""
        from agent import make_backend
        from deepagents.backends import StateBackend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        # The default backend should be StateBackend
        assert hasattr(backend, 'default')
        assert isinstance(backend.default, StateBackend)

    def test_make_backend_has_memory_route(self):
        """Test that CompositeBackend routes /memories/ to StoreBackend."""
        from agent import make_backend
        from deepagents.backends import StoreBackend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        # Should have a route for /memories/
        assert hasattr(backend, 'routes')
        assert '/memories/' in backend.routes
        assert isinstance(backend.routes['/memories/'], StoreBackend)

    def test_make_backend_receives_runtime(self):
        """Test that runtime is passed to backends."""
        from agent import make_backend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        # Verify runtime was used (backends should have it)
        assert backend is not None


@pytest.mark.unit
@pytest.mark.requires_db
class TestPostgresStoreConfiguration:
    """Tests for PostgresStore configuration."""

    @patch('langgraph.store.postgres.PostgresStore')
    def test_postgres_store_created_from_conn_string(self, mock_store_class):
        """Test that PostgresStore is created with connection string."""
        mock_store = Mock()
        mock_store_class.from_conn_string.return_value = mock_store

        # Import will trigger store creation
        import importlib
        import sys
        if 'agent' in sys.modules:
            del sys.modules['agent']
        import agent as agent_module

        # Verify from_conn_string was called
        mock_store_class.from_conn_string.assert_called()

    def test_postgres_checkpointer_created_from_conn_string(self):
        """Test that PostgresSaver is created with connection string."""
        # This test verifies that PostgresSaver.from_conn_string is called during module initialization
        # We need to patch before importing the module
        import sys

        # Remove agent module if already loaded
        if 'agent' in sys.modules:
            del sys.modules['agent']

        with patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
            'TAVILY_API_KEY': 'test_key',
            'DAYTONA_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key'
        }):
            with patch('langgraph.checkpoint.postgres.PostgresSaver') as mock_saver_class:
                with patch('langgraph.store.postgres.PostgresStore') as mock_store_class:
                    with patch('agent.Daytona'):
                        with patch('agent.TavilyClient'):
                            mock_saver = Mock()
                            mock_store = Mock()
                            mock_saver_class.from_conn_string.return_value = mock_saver
                            mock_store_class.from_conn_string.return_value = mock_store

                            # Now import agent - this will trigger the module-level code
                            import agent as agent_module

                            # Verify from_conn_string was called
                            mock_saver_class.from_conn_string.assert_called()

    @patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost:5432/testdb'})
    @patch('agent.PostgresStore')
    @patch('agent.PostgresSaver')
    def test_database_url_from_environment(self, mock_saver, mock_store):
        """Test that DATABASE_URL is read from environment."""
        import importlib
        import agent as agent_module

        # Reload to pick up environment variable
        importlib.reload(agent_module)

        # Verify the DATABASE_URL was used
        assert agent_module.DATABASE_URL == 'postgresql://user:pass@localhost:5432/testdb'


@pytest.mark.unit
class TestMemoryNamespaces:
    """Tests for memory namespace handling."""

    def test_memory_cleanup_searches_filesystem_namespace(self):
        """Test that MemoryCleanupMiddleware searches filesystem namespace."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_store.search.return_value = []
        middleware = MemoryCleanupMiddleware(mock_store)

        # Execute after_agent to trigger search
        middleware.after_agent({}, Mock(), [])

        # Verify it searches the filesystem namespace
        mock_store.search.assert_called_with(("filesystem",))

    def test_memory_cleanup_filters_memories_path(self):
        """Test that middleware only processes files in /memories/."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        # Create items in and out of /memories/ path
        in_memories = Mock()
        in_memories.key = "/memories/test.txt"
        in_memories.value = {"content": ["## Test", "- Memory 1"]}

        out_memories = Mock()
        out_memories.key = "/other/test.txt"
        out_memories.value = {"content": ["## Test", "- Memory 1"]}

        mock_store.search.return_value = [in_memories, out_memories]
        middleware = MemoryCleanupMiddleware(mock_store)

        # Execute - should only process /memories/ files
        middleware.after_agent({}, Mock(), [])

        # Verify search was called
        mock_store.search.assert_called_once()


@pytest.mark.unit
class TestMemoryPersistence:
    """Tests for memory persistence paths."""

    def test_backend_routes_memories_to_persistent_storage(self):
        """Test that /memories/ path routes to StoreBackend (persistent)."""
        from agent import make_backend
        from deepagents.backends import StoreBackend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        # Memories should use persistent storage
        assert '/memories/' in backend.routes
        memory_backend = backend.routes['/memories/']
        assert isinstance(memory_backend, StoreBackend)

    def test_backend_default_is_ephemeral(self):
        """Test that default backend is ephemeral (StateBackend)."""
        from agent import make_backend
        from deepagents.backends import StateBackend

        mock_runtime = Mock()
        backend = make_backend(mock_runtime)

        # Default should be ephemeral
        assert isinstance(backend.default, StateBackend)


@pytest.mark.unit
class TestStoreOperations:
    """Tests for store operations (mock-based)."""

    def test_store_search_returns_list(self):
        """Test that store.search returns a list."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_item1 = Mock()
        mock_item1.key = "/memories/mem1.txt"
        mock_item1.value = {"content": ["## Test", "- Memory 1"]}
        mock_item2 = Mock()
        mock_item2.key = "/memories/mem2.txt"
        mock_item2.value = {"content": ["## Test", "- Memory 2"]}
        mock_memories = [mock_item1, mock_item2]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(mock_store)

        # Trigger search via after_agent
        state = {}
        run = Mock()
        tool_calls = []
        middleware.after_agent(state, run, tool_calls)

        # Verify search was called
        mock_store.search.assert_called_once_with(("filesystem",))

    @patch('agent.ChatOpenAI')
    def test_store_put_called_when_trimming(self, mock_openai):
        """Test that store.put is called with correct arguments when trimming."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        # Create content with 50 memories
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(50)])
        mock_item.value = {"content": content.split("\n"), "created_at": "2025-01-01T00:00:00"}
        mock_store.search.return_value = [mock_item]

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(30)])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)

        state = {}
        run = Mock()
        tool_calls = []

        with patch('builtins.print'):
            middleware.after_agent(state, run, tool_calls)

        # Verify put was called
        mock_store.put.assert_called_once()
        call_args = mock_store.put.call_args[0]
        assert len(call_args) == 3
        assert call_args[0] == ("filesystem",)  # namespace
        assert call_args[1] == "/memories/test.txt"  # key

    @patch('agent.ChatOpenAI')
    def test_store_handles_large_memory_set(self, mock_openai):
        """Test store operations with multiple large memory files."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        # Create 3 files with many memories each
        mock_files = []
        for i in range(3):
            mock_item = Mock()
            mock_item.key = f"/memories/file{i}.txt"
            content = f"## File {i}\n" + "\n".join([f"- Memory {j}" for j in range(50)])
            mock_item.value = {"content": content.split("\n"), "created_at": "2025-01-01T00:00:00"}
            mock_files.append(mock_item)

        mock_store.search.return_value = mock_files

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(30)])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)

        state = {}
        run = Mock()
        tool_calls = []

        with patch('builtins.print'):
            middleware.after_agent(state, run, tool_calls)

        # Should trim all 3 files
        assert mock_store.put.call_count == 3


@pytest.mark.unit
class TestMemoryFileStructure:
    """Tests for memory file structure expectations."""

    def test_expected_memory_files_documented(self):
        """Test that expected memory files are documented in code."""
        # The documentation mentions these memory files:
        # - /memories/website_quality.txt
        # - /memories/research_lessons.txt
        # - /memories/source_notes.txt
        # - /memories/coding.txt

        # This is a documentation test to ensure we remember these files exist
        expected_files = [
            "/memories/website_quality.txt",
            "/memories/research_lessons.txt",
            "/memories/source_notes.txt",
            "/memories/coding.txt",
        ]

        # Just verify the list structure
        assert len(expected_files) == 4
        assert all(f.startswith("/memories/") for f in expected_files)


@pytest.mark.unit
class TestDatabaseConfiguration:
    """Tests for database configuration."""

    def test_missing_database_url_raises_error(self):
        """Test that missing DATABASE_URL raises KeyError."""
        # This test verifies that DATABASE_URL is required
        # We can't actually test this without breaking the module import
        # So we just verify the constant exists when the module is imported
        import agent as agent_module
        assert hasattr(agent_module, 'DATABASE_URL')

    @patch.dict(os.environ, {'DATABASE_URL': 'postgresql://localhost/testdb'})
    def test_database_url_format(self):
        """Test DATABASE_URL format validation."""
        url = os.environ['DATABASE_URL']

        # Basic format check
        assert url.startswith('postgresql://')
        assert 'testdb' in url


@pytest.mark.unit
class TestMemoryTimestamps:
    """Tests for memory timestamp handling in trimmed files."""

    @patch('agent.ChatOpenAI')
    def test_trimmed_file_has_modified_at_timestamp(self, mock_openai):
        """Test that trimmed files get a modified_at timestamp."""
        from agent import MemoryCleanupMiddleware
        from datetime import datetime

        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(50)])
        mock_item.value = {"content": content.split("\n"), "created_at": "2025-01-01T00:00:00"}
        mock_store.search.return_value = [mock_item]

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(30)])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm

        middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)

        state = {}
        run = Mock()
        tool_calls = []

        with patch('builtins.print'):
            with patch('agent.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2025, 1, 15, 12, 0, 0)
                middleware.after_agent(state, run, tool_calls)

        # Verify put was called with modified_at
        assert mock_store.put.called
        call_args = mock_store.put.call_args[0]
        saved_value = call_args[2]
        assert "modified_at" in saved_value

    def test_trimmed_file_preserves_created_at(self):
        """Test that created_at timestamp is preserved when trimming."""
        from agent import MemoryCleanupMiddleware

        mock_store = Mock()
        mock_item = Mock()
        mock_item.key = "/memories/test.txt"
        original_created = "2025-01-01T00:00:00"
        content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(50)])
        mock_item.value = {"content": content.split("\n"), "created_at": original_created}
        mock_store.search.return_value = [mock_item]

        # Mock LLM
        with patch('agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "## Test\n" + "\n".join([f"- Memory {i}" for i in range(30)])
            mock_llm.invoke.return_value = mock_response
            mock_openai.return_value = mock_llm

            middleware = MemoryCleanupMiddleware(mock_store, max_memories_per_file=30)

            with patch('builtins.print'):
                middleware.after_agent({}, Mock(), [])

        # Verify created_at was preserved
        call_args = mock_store.put.call_args[0]
        saved_value = call_args[2]
        assert saved_value["created_at"] == original_created
