"""Pytest configuration and shared fixtures for Deep Research Agent tests."""

import os
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Set minimal environment variables and import path for tests."""
    project_root = Path(__file__).resolve().parents[1]
    agents_dir = project_root / "deep-agent"

    if str(agents_dir) not in sys.path:
        sys.path.insert(0, str(agents_dir))

    os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    os.environ.setdefault("TAVILY_API_KEY", "test_tavily_key")
    os.environ.setdefault("DAYTONA_API_KEY", "test_daytona_key")
    os.environ.setdefault("OPENAI_API_KEY", "test_openai_key")

    yield


@pytest.fixture(scope="session", autouse=True)
def stub_postgres_store():
    """Stub PostgresStore to avoid real database connections during imports."""
    context_manager = MagicMock()
    context_manager.__enter__.return_value = MagicMock(name="store")

    with patch("langgraph.store.postgres.base.PostgresStore") as mock_store_base:
        with patch("langgraph.store.postgres.PostgresStore") as mock_store_public:
            mock_store_base.from_conn_string.return_value = context_manager
            mock_store_public.from_conn_string.return_value = context_manager
            yield
@pytest.fixture
def temp_env_var():
    """Context manager for temporary environment variables."""
    original_env = os.environ.copy()

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = value

    yield _set_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_agent_state():
    """Mock agent state for testing."""
    return {
        "messages": [],
        "context": {},
        "memory": {},
        "tool_calls": [],
        "run_id": "test_run_123",
    }


# =============================================================================
# SKIP MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may call external services)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_db: Tests that require PostgreSQL database"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests that require external API keys"
    )
