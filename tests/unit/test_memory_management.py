"""Unit tests for memory backend and cleanup middleware."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestMemoryBackend:
    """Tests for middleware.memory_backend utilities."""

    def test_make_backend_routes_memories(self):
        from middleware.memory_backend import make_backend
        from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

        runtime = MagicMock()
        backend = make_backend(runtime)

        assert isinstance(backend, CompositeBackend)
        assert isinstance(backend.default, StateBackend)
        assert "/memories/" in backend.routes
        assert isinstance(backend.routes["/memories/"], StoreBackend)


@pytest.mark.unit
class TestMemoryCleanupMiddleware:
    """Tests for MemoryCleanupMiddleware behavior."""

    def test_searches_filesystem_namespace(self):
        from middleware.memory_cleanup import MemoryCleanupMiddleware

        store = MagicMock()
        store.search.return_value = []
        middleware = MemoryCleanupMiddleware(store)

        middleware.after_agent(state={}, runtime=MagicMock())

        store.search.assert_called_once_with(("filesystem",))

    def test_skips_small_files(self):
        from middleware.memory_cleanup import MemoryCleanupMiddleware

        store = MagicMock()
        item = MagicMock()
        item.key = "/memories/test.txt"
        item.value = {"content": ["## Test", "- a", "- b"]}
        store.search.return_value = [item]

        middleware = MemoryCleanupMiddleware(store, max_memories_per_file=5)
        middleware.after_agent(state={}, runtime=MagicMock())

        store.put.assert_not_called()

    def test_trims_when_over_limit(self):
        from middleware.memory_cleanup import MemoryCleanupMiddleware

        store = MagicMock()
        item = MagicMock()
        item.key = "/memories/test.txt"
        content = "## Test\n" + "\n".join(f"- Memory {i}" for i in range(10))
        item.value = {"content": content.split("\n"), "created_at": "2025-01-01T00:00:00"}
        store.search.return_value = [item]

        trimmed_response = MagicMock()
        trimmed_response.content = "## Test\n- Trimmed 1\n- Trimmed 2"

        with patch("middleware.memory_cleanup.ChatOpenAI") as mock_chat:
            llm = MagicMock()
            llm.invoke.return_value = trimmed_response
            mock_chat.return_value = llm

            middleware = MemoryCleanupMiddleware(store, max_memories_per_file=2)
            middleware.after_agent(state={}, runtime=MagicMock())

        store.put.assert_called_once()
        args, kwargs = store.put.call_args
        assert args[1] == "/memories/test.txt"
        assert "- Trimmed 1" in "\n".join(args[2]["content"])

    def test_only_processes_txt_files(self):
        from middleware.memory_cleanup import MemoryCleanupMiddleware

        store = MagicMock()
        txt_item = MagicMock()
        txt_item.key = "/memories/test.txt"
        txt_item.value = {"content": ["## Section", "- item"]}

        other_item = MagicMock()
        other_item.key = "/memories/data.json"
        other_item.value = {"content": "{}"}

        store.search.return_value = [txt_item, other_item]

        with patch("middleware.memory_cleanup.ChatOpenAI") as mock_chat:
            middleware = MemoryCleanupMiddleware(store, max_memories_per_file=1)
            middleware.after_agent(state={}, runtime=MagicMock())

        mock_chat.return_value.invoke.assert_not_called()

    def test_error_is_swallowed(self, capsys):
        from middleware.memory_cleanup import MemoryCleanupMiddleware

        store = MagicMock()
        store.search.side_effect = RuntimeError("boom")

        middleware = MemoryCleanupMiddleware(store)
        middleware.after_agent(state={}, runtime=MagicMock())

        output = capsys.readouterr().out
        assert "Memory cleanup failed" in output
