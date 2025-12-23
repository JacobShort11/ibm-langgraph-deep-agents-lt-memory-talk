"""Unit tests for tool functions."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestWebSearch:
    """Tests for the web_search tool."""

    def test_web_search_calls_client(self):
        from tools.web_search import web_search

        with patch("tools.web_search.tavily_client") as mock_client:
            mock_client.search.return_value = {"results": []}

            result = web_search("test query", max_results=3, topic="news")

            mock_client.search.assert_called_once_with("test query", max_results=3, topic="news")
            assert isinstance(result, dict)


@pytest.mark.unit
class TestExecutePythonCode:
    """Tests for the execute_python_code tool."""

    def test_executes_code_with_setup_and_cleans_up(self):
        from tools.code_execution import execute_python_code

        sandbox = MagicMock()
        sandbox.process.code_run.return_value = MagicMock(result="done")
        sandbox.fs.list_files.return_value = []

        with patch("tools.code_execution.daytona") as mock_daytona:
            mock_daytona.create.return_value = sandbox

            result = execute_python_code("print('hello')")

            mock_daytona.create.assert_called_once()
            mock_daytona.delete.assert_called_once_with(sandbox)
            assert "Output:" in result

            assert sandbox.process.code_run.call_count >= 1
            call_code = sandbox.process.code_run.call_args_list[-1][0][0]
            assert "import pandas as pd" in call_code
            assert "print('hello')" in call_code

    def test_downloads_generated_files(self, tmp_path, monkeypatch):
        from tools.code_execution import execute_python_code, PLOTS_DIR

        sandbox = MagicMock()
        sandbox.process.code_run.return_value = MagicMock(result=None)
        file_one = MagicMock()
        file_one.name = "chart.png"
        file_two = MagicMock()
        file_two.name = "table.csv"
        sandbox.fs.list_files.return_value = [file_one, file_two]

        monkeypatch.setattr("tools.code_execution.PLOTS_DIR", tmp_path)

        with patch("tools.code_execution.daytona") as mock_daytona:
            mock_daytona.create.return_value = sandbox
            result = execute_python_code("pass")

        sandbox.fs.download_file.assert_any_call("/home/daytona/outputs/chart.png", str(tmp_path / "chart.png"))
        sandbox.fs.download_file.assert_any_call("/home/daytona/outputs/table.csv", str(tmp_path / "table.csv"))
        assert "Generated files" in result
        assert "Plots saved" in result
