"""
Unit tests for tool functions (web_search, execute_python_code).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add the deep-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deep-agent'))


@pytest.mark.unit
class TestWebSearch:
    """Tests for the web_search tool function."""

    @patch('agent.tavily_client')
    def test_web_search_basic(self, mock_client):
        """Test basic web search functionality."""
        from agent import web_search

        # Setup mock response
        mock_client.search.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2"},
            ]
        }

        # Execute search
        result = web_search("test query")

        # Verify
        mock_client.search.assert_called_once_with(
            "test query",
            max_results=5,
            topic="general"
        )
        assert "results" in result
        assert len(result["results"]) == 2

    @patch('agent.tavily_client')
    def test_web_search_with_max_results(self, mock_client):
        """Test web search with custom max_results parameter."""
        from agent import web_search

        mock_client.search.return_value = {"results": []}

        web_search("test query", max_results=10)

        mock_client.search.assert_called_once_with(
            "test query",
            max_results=10,
            topic="general"
        )

    @patch('agent.tavily_client')
    def test_web_search_with_topic(self, mock_client):
        """Test web search with different topic types."""
        from agent import web_search

        mock_client.search.return_value = {"results": []}

        # Test different topics
        for topic in ["general", "news", "finance"]:
            mock_client.reset_mock()
            web_search("test query", topic=topic)
            mock_client.search.assert_called_once_with(
                "test query",
                max_results=5,
                topic=topic
            )

    @patch('agent.tavily_client')
    def test_web_search_returns_dict(self, mock_client):
        """Test that web_search returns a dictionary."""
        from agent import web_search

        mock_client.search.return_value = {"results": [], "query": "test"}

        result = web_search("test query")

        assert isinstance(result, dict)

    @patch('agent.tavily_client')
    def test_web_search_handles_empty_results(self, mock_client):
        """Test web search with no results."""
        from agent import web_search

        mock_client.search.return_value = {"results": []}

        result = web_search("obscure query")

        assert result["results"] == []

    @patch('agent.tavily_client')
    def test_web_search_error_handling(self, mock_client):
        """Test web search error handling."""
        from agent import web_search

        # Simulate API error
        mock_client.search.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            web_search("test query")

        assert "API Error" in str(exc_info.value)


@pytest.mark.unit
class TestExecutePythonCode:
    """Tests for the execute_python_code tool function."""

    @patch('agent.daytona')
    def test_execute_python_code_basic(self, mock_daytona):
        """Test basic Python code execution."""
        from agent import execute_python_code

        # Setup mock sandbox
        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result="42")
        mock_sandbox.fs.list_files.return_value = []
        mock_daytona.create.return_value = mock_sandbox

        # Execute code
        result = execute_python_code("print(6 * 7)")

        # Verify sandbox was created and used
        mock_daytona.create.assert_called_once()
        mock_sandbox.process.code_run.assert_called_once()
        mock_daytona.remove.assert_called_once_with(mock_sandbox)

        assert "42" in result

    @patch('agent.daytona')
    def test_execute_python_code_with_output(self, mock_daytona):
        """Test code execution with output."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(
            result="Mean: 50.5\nStd: 28.87"
        )
        mock_sandbox.fs.list_files.return_value = []
        mock_daytona.create.return_value = mock_sandbox

        result = execute_python_code("import numpy as np\nprint(np.mean([1,2,3]))")

        assert "Output:" in result
        assert "Mean: 50.5" in result

    @patch('agent.daytona')
    def test_execute_python_code_with_files(self, mock_daytona):
        """Test code execution that generates files."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result="Plot saved")
        mock_sandbox.fs.list_files.return_value = ["chart.png", "data.csv"]
        mock_daytona.create.return_value = mock_sandbox

        result = execute_python_code("plt.savefig('/home/daytona/outputs/chart.png')")

        assert "Generated files:" in result
        assert "chart.png" in result
        assert "data.csv" in result

    @patch('agent.daytona')
    def test_execute_python_code_no_output(self, mock_daytona):
        """Test code execution with no output."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result=None)
        mock_sandbox.fs.list_files.return_value = []
        mock_daytona.create.return_value = mock_sandbox

        result = execute_python_code("x = 5")

        assert result == "Code executed successfully"

    @patch('agent.daytona')
    def test_execute_python_code_includes_setup(self, mock_daytona):
        """Test that setup code is included in execution."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result="OK")
        mock_sandbox.fs.list_files.return_value = []
        mock_daytona.create.return_value = mock_sandbox

        execute_python_code("print('test')")

        # Verify setup code was included
        call_args = mock_sandbox.process.code_run.call_args[0][0]
        assert "import pandas as pd" in call_args
        assert "import numpy as np" in call_args
        assert "import matplotlib.pyplot as plt" in call_args
        assert "import seaborn as sns" in call_args
        assert "os.makedirs('/home/daytona/outputs', exist_ok=True)" in call_args
        assert "print('test')" in call_args

    @patch('agent.daytona')
    def test_execute_python_code_cleanup_on_success(self, mock_daytona):
        """Test sandbox cleanup after successful execution."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result="OK")
        mock_sandbox.fs.list_files.return_value = []
        mock_daytona.create.return_value = mock_sandbox

        execute_python_code("print('test')")

        # Verify cleanup
        mock_daytona.remove.assert_called_once_with(mock_sandbox)

    @patch('agent.daytona')
    def test_execute_python_code_cleanup_on_error(self, mock_daytona):
        """Test sandbox cleanup after execution error."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.side_effect = Exception("Execution error")
        mock_daytona.create.return_value = mock_sandbox

        with pytest.raises(Exception):
            execute_python_code("raise ValueError('test')")

        # Verify cleanup still happens
        mock_daytona.remove.assert_called_once_with(mock_sandbox)

    @patch('agent.daytona')
    def test_execute_python_code_fs_list_error(self, mock_daytona):
        """Test handling of file listing errors."""
        from agent import execute_python_code

        mock_sandbox = Mock()
        mock_sandbox.process.code_run.return_value = Mock(result="OK")
        mock_sandbox.fs.list_files.side_effect = Exception("FS error")
        mock_daytona.create.return_value = mock_sandbox

        # Should not raise, just skip file listing
        result = execute_python_code("print('test')")

        assert "OK" in result
        # Should not mention files
        assert "Generated files:" not in result


@pytest.mark.unit
class TestToolDocstrings:
    """Tests to verify tool docstrings are correct."""

    def test_web_search_docstring(self):
        """Test web_search has proper docstring."""
        from agent import web_search

        assert web_search.__doc__ is not None
        assert "Search the web" in web_search.__doc__
        assert "query" in web_search.__doc__
        assert "max_results" in web_search.__doc__

    def test_execute_python_code_docstring(self):
        """Test execute_python_code has proper docstring."""
        from agent import execute_python_code

        assert execute_python_code.__doc__ is not None
        assert "Execute Python code" in execute_python_code.__doc__
        assert "sandbox" in execute_python_code.__doc__.lower()
        assert "pandas" in execute_python_code.__doc__


@pytest.mark.unit
class TestToolTypeAnnotations:
    """Tests for tool function type annotations."""

    def test_web_search_annotations(self):
        """Test web_search has correct type annotations."""
        from agent import web_search
        import inspect

        sig = inspect.signature(web_search)

        assert sig.parameters['query'].annotation == str
        assert sig.parameters['max_results'].annotation == int
        assert sig.parameters['max_results'].default == 5
        assert sig.return_annotation == dict

    def test_execute_python_code_annotations(self):
        """Test execute_python_code has correct type annotations."""
        from agent import execute_python_code
        import inspect

        sig = inspect.signature(execute_python_code)

        assert sig.parameters['code'].annotation == str
        assert sig.return_annotation == str
