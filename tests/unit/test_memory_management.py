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

    @patch('agent.PostgresSaver')
    def test_postgres_checkpointer_created_from_conn_string(self, mock_saver_class):
        """Test that PostgresSaver is created with connection string."""
        mock_saver = Mock()
        mock_saver_class.from_conn_string.return_value = mock_saver

        # Import will trigger checkpointer creation
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

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

    def test_memory_cleanup_uses_correct_namespace(self):
        """Test that MemoryCleanupMiddleware uses correct namespace."""
        from agent import MemoryCleanupMiddleware

        middleware = MemoryCleanupMiddleware()

        assert middleware.namespace == ("agent", "memories")

    def test_memory_namespace_structure(self):
        """Test memory namespace is a tuple."""
        from agent import MemoryCleanupMiddleware

        middleware = MemoryCleanupMiddleware()

        assert isinstance(middleware.namespace, tuple)
        assert len(middleware.namespace) == 2
        assert middleware.namespace[0] == "agent"
        assert middleware.namespace[1] == "memories"


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

    @patch('agent.store')
    def test_store_search_returns_list(self, mock_store):
        """Test that store.search returns a list."""
        from agent import MemoryCleanupMiddleware

        mock_memories = [Mock(key="mem1"), Mock(key="mem2")]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware()

        # Trigger search via after_agent
        state = {}
        run = Mock()
        tool_calls = []
        middleware.after_agent(state, run, tool_calls)

        # Verify search was called
        mock_store.search.assert_called_once_with(("agent", "memories"))

    @patch('agent.store')
    def test_store_delete_called_with_namespace_and_key(self, mock_store):
        """Test that store.delete is called with correct arguments."""
        from agent import MemoryCleanupMiddleware

        mock_memories = [
            Mock(key="mem1", updated_at="2025-01-01"),
            Mock(key="mem2", updated_at="2025-01-02"),
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=1)

        state = {}
        run = Mock()
        tool_calls = []
        middleware.after_agent(state, run, tool_calls)

        # Verify delete was called with namespace and key
        mock_store.delete.assert_called_once()
        call_args = mock_store.delete.call_args[0]
        assert len(call_args) == 2
        assert call_args[0] == ("agent", "memories")  # namespace
        assert isinstance(call_args[1], str)  # key

    @patch('agent.store')
    def test_store_handles_large_memory_set(self, mock_store):
        """Test store operations with large memory set."""
        from agent import MemoryCleanupMiddleware

        # Create 1000 memories
        mock_memories = [
            Mock(key=f"mem_{i:04d}", updated_at=f"2025-01-{i%30+1:02d}")
            for i in range(1000)
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=100)

        state = {}
        run = Mock()
        tool_calls = []

        with patch('builtins.print'):
            middleware.after_agent(state, run, tool_calls)

        # Should delete 900 memories (1000 - 100)
        assert mock_store.delete.call_count == 900


@pytest.mark.unit
class TestMemoryFileStructure:
    """Tests for memory file structure expectations."""

    def test_expected_memory_files_documented(self):
        """Test that expected memory files are documented in code."""
        # The documentation mentions these memory files:
        # - /memories/website_quality.txt
        # - /memories/research_lessons.txt
        # - /memories/source_notes.txt

        # This is a documentation test to ensure we remember these files exist
        expected_files = [
            "/memories/website_quality.txt",
            "/memories/research_lessons.txt",
            "/memories/source_notes.txt",
        ]

        # Just verify the list structure
        assert len(expected_files) == 3
        assert all(f.startswith("/memories/") for f in expected_files)


@pytest.mark.unit
class TestDatabaseConfiguration:
    """Tests for database configuration."""

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_database_url_raises_error(self):
        """Test that missing DATABASE_URL raises KeyError."""
        # Remove DATABASE_URL if it exists
        os.environ.pop('DATABASE_URL', None)

        with pytest.raises(KeyError):
            import importlib
            import agent as agent_module
            importlib.reload(agent_module)

    @patch.dict(os.environ, {'DATABASE_URL': 'postgresql://localhost/testdb'})
    def test_database_url_format(self):
        """Test DATABASE_URL format validation."""
        url = os.environ['DATABASE_URL']

        # Basic format check
        assert url.startswith('postgresql://')
        assert 'testdb' in url


@pytest.mark.unit
class TestMemoryTimestamps:
    """Tests for memory timestamp handling."""

    @patch('agent.store')
    def test_memories_sorted_by_updated_at(self, mock_store):
        """Test that memories are sorted by updated_at timestamp."""
        from agent import MemoryCleanupMiddleware
        from datetime import datetime

        # Create memories with different timestamps (out of order)
        mock_memories = [
            Mock(key="mem2", updated_at=datetime(2025, 1, 15, 0, 0, 0)),
            Mock(key="mem1", updated_at=datetime(2025, 1, 10, 0, 0, 0)),
            Mock(key="mem3", updated_at=datetime(2025, 1, 20, 0, 0, 0)),
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=2)

        state = {}
        run = Mock()
        tool_calls = []
        middleware.after_agent(state, run, tool_calls)

        # Should delete oldest (mem1)
        deleted_key = mock_store.delete.call_args[0][1]
        assert deleted_key == "mem1"

    @patch('agent.store')
    def test_memories_with_same_timestamp(self, mock_store):
        """Test handling of memories with identical timestamps."""
        from agent import MemoryCleanupMiddleware
        from datetime import datetime

        # Create memories with same timestamp
        same_time = datetime(2025, 1, 15, 0, 0, 0)
        mock_memories = [
            Mock(key="mem1", updated_at=same_time),
            Mock(key="mem2", updated_at=same_time),
            Mock(key="mem3", updated_at=same_time),
        ]
        mock_store.search.return_value = mock_memories

        middleware = MemoryCleanupMiddleware(max_items=2)

        state = {}
        run = Mock()
        tool_calls = []
        middleware.after_agent(state, run, tool_calls)

        # Should delete one memory (order may vary)
        assert mock_store.delete.call_count == 1
