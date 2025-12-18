"""
Pytest configuration and shared fixtures for Deep Research Agent tests.
"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables before any tests run."""
    os.environ["TESTING"] = "1"
    os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    os.environ.setdefault("TAVILY_API_KEY", "test_tavily_key")
    os.environ.setdefault("DAYTONA_API_KEY", "test_daytona_key")
    os.environ.setdefault("OPENAI_API_KEY", "test_openai_key")
    yield
    # Cleanup after all tests
    os.environ.pop("TESTING", None)


# =============================================================================
# MOCK EXTERNAL SERVICES
# =============================================================================

@pytest.fixture
def mock_tavily_client():
    """Mock Tavily client for web search tests."""
    with patch("deep-agent.agent.TavilyClient") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Default search response
        mock_instance.search.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "Test content 1",
                    "score": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Test content 2",
                    "score": 0.89,
                },
            ],
            "query": "test query",
        }

        yield mock_instance


@pytest.fixture
def mock_daytona():
    """Mock Daytona client for code execution tests."""
    with patch("deep-agent.agent.Daytona") as mock_daytona_class:
        mock_daytona_instance = Mock()
        mock_sandbox = Mock()

        # Configure sandbox behavior
        mock_sandbox.process.code_run.return_value = Mock(
            result="Code executed successfully",
            exit_code=0,
        )
        mock_sandbox.fs.list_files.return_value = []

        mock_daytona_instance.create.return_value = mock_sandbox
        mock_daytona_instance.remove.return_value = None
        mock_daytona_class.return_value = mock_daytona_instance

        yield mock_daytona_instance, mock_sandbox


@pytest.fixture
def mock_postgres_store():
    """Mock PostgresStore for memory tests."""
    with patch("deep-agent.agent.PostgresStore") as mock_store_class:
        mock_store = Mock()

        # Mock store methods
        mock_store.from_conn_string.return_value = mock_store
        mock_store.search.return_value = []
        mock_store.put.return_value = None
        mock_store.delete.return_value = None
        mock_store.get.return_value = None

        mock_store_class.from_conn_string.return_value = mock_store

        yield mock_store


@pytest.fixture
def mock_postgres_checkpointer():
    """Mock PostgresSaver for checkpointing tests."""
    with patch("deep-agent.agent.PostgresSaver") as mock_saver_class:
        mock_saver = Mock()
        mock_saver.from_conn_string.return_value = mock_saver
        mock_saver_class.from_conn_string.return_value = mock_saver

        yield mock_saver


@pytest.fixture
def mock_openai_chat():
    """Mock ChatOpenAI for LLM tests."""
    with patch("deep-agent.agent.ChatOpenAI") as mock_chat_class:
        mock_chat = Mock()
        mock_chat_class.return_value = mock_chat

        # Mock response
        mock_response = Mock()
        mock_response.content = "Test LLM response"
        mock_chat.invoke.return_value = mock_response

        yield mock_chat


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_web_search_results():
    """Sample web search results for testing."""
    return {
        "results": [
            {
                "title": "Python Best Practices 2025",
                "url": "https://realpython.com/best-practices",
                "content": "Comprehensive guide to Python best practices including testing, documentation, and code quality.",
                "score": 0.98,
                "published_date": "2025-01-15",
            },
            {
                "title": "Unit Testing in Python",
                "url": "https://docs.python.org/testing",
                "content": "Official Python documentation on unittest and testing frameworks.",
                "score": 0.95,
                "published_date": "2024-12-01",
            },
            {
                "title": "Pytest Tutorial",
                "url": "https://pytest.org/tutorial",
                "content": "Learn pytest from scratch with examples and best practices.",
                "score": 0.92,
                "published_date": "2024-11-20",
            },
        ],
        "query": "python testing best practices",
        "response_time": 0.45,
    }


@pytest.fixture
def sample_code_execution_result():
    """Sample code execution result for testing."""
    return {
        "result": "Mean: 50.5\nStd Dev: 28.87",
        "exit_code": 0,
        "files": ["/home/daytona/outputs/histogram.png"],
    }


@pytest.fixture
def sample_memory_items():
    """Sample memory items for testing."""
    return [
        Mock(
            key="memory_001",
            value={"content": "Research finding 1", "source": "web"},
            updated_at=datetime(2025, 1, 15, 10, 0, 0),
            namespace=("agent", "memories"),
        ),
        Mock(
            key="memory_002",
            value={"content": "Research finding 2", "source": "analysis"},
            updated_at=datetime(2025, 1, 15, 11, 0, 0),
            namespace=("agent", "memories"),
        ),
        Mock(
            key="memory_003",
            value={"content": "Research finding 3", "source": "credibility"},
            updated_at=datetime(2025, 1, 15, 12, 0, 0),
            namespace=("agent", "memories"),
        ),
    ]


# =============================================================================
# HELPER FIXTURES
# =============================================================================

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
