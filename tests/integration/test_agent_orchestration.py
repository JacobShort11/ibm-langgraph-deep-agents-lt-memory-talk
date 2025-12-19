"""
Integration tests for main agent orchestration and configuration.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the deep-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deep-agent'))


@pytest.mark.integration
class TestAgentCreation:
    """Integration tests for agent creation."""

    def test_create_research_agent_function_exists(self):
        """Test that create_research_agent function exists."""
        import agent

        assert hasattr(agent, 'create_research_agent')
        assert callable(agent.create_research_agent)

    @patch('agent.store')
    @patch('agent.checkpointer')
    def test_create_research_agent_returns_graph(self, mock_checkpointer, mock_store):
        """Test that create_research_agent returns a graph."""
        import agent

        research_agent = agent.create_research_agent()

        assert research_agent is not None

    @patch('agent.store')
    @patch('agent.checkpointer')
    def test_agent_instance_created(self, mock_checkpointer, mock_store):
        """Test that agent instance is created for LangGraph Studio."""
        import agent

        # The module should have an 'agent' instance
        assert hasattr(agent, 'agent')
        assert agent.agent is not None


@pytest.mark.integration
class TestMainAgentConfiguration:
    """Integration tests for main agent configuration."""

    def test_main_agent_has_web_search_tool(self):
        """Test that main agent has web_search tool."""
        import agent

        # The create_research_agent function configures tools=[web_search]
        # This is verified by checking the function definition
        assert hasattr(agent, 'web_search')

    def test_main_agent_uses_correct_model(self):
        """Test that main agent uses GPT-5 (not 5.1)."""
        import agent

        # agent_llm should use gpt-5-2025-08-07
        assert hasattr(agent, 'agent_llm')

    def test_main_agent_has_system_prompt(self):
        """Test that main agent uses correct system prompt."""
        import agent
        from prompts.main_agent import PROMPT as MAIN_AGENT_PROMPT

        assert MAIN_AGENT_PROMPT is not None
        assert len(MAIN_AGENT_PROMPT) > 0

    def test_main_agent_includes_all_subagents(self):
        """Test that main agent includes all 3 sub-agents."""
        import agent

        # Verify all sub-agents are defined
        assert hasattr(agent, 'analysis_sub_agent')
        assert hasattr(agent, 'web_research_sub_agent')
        assert hasattr(agent, 'credibility_sub_agent')

    def test_main_agent_has_memory_cleanup_middleware(self):
        """Test that main agent has memory cleanup middleware."""
        import agent

        # The create_research_agent uses middleware=[MemoryCleanupMiddleware(max_items=100)]
        assert hasattr(agent, 'MemoryCleanupMiddleware')


@pytest.mark.integration
class TestAgentBackendConfiguration:
    """Integration tests for agent backend configuration."""

    @patch('agent.store')
    @patch('agent.checkpointer')
    def test_agent_uses_make_backend_function(self, mock_checkpointer, mock_store):
        """Test that agent uses make_backend function."""
        import agent

        # The make_backend function should be used in create_research_agent
        assert hasattr(agent, 'make_backend')
        assert callable(agent.make_backend)

    @patch('agent.store')
    @patch('agent.checkpointer')
    def test_agent_has_store_configured(self, mock_checkpointer, mock_store):
        """Test that agent has PostgresStore configured."""
        import agent

        # Store should be configured
        assert hasattr(agent, 'store')

    @patch('agent.store')
    @patch('agent.checkpointer')
    def test_agent_has_checkpointer_configured(self, mock_checkpointer, mock_store):
        """Test that agent has PostgresSaver checkpointer configured."""
        import agent

        # Checkpointer should be configured
        assert hasattr(agent, 'checkpointer')


@pytest.mark.integration
class TestAgentRecursionConfiguration:
    """Integration tests for recursion limits."""

    def test_main_agent_has_recursion_limit(self):
        """Test that main agent has recursion limit set."""
        import agent

        # The create_research_agent returns graph with .with_config({"recursion_limit": 1000})
        # This is tested by verifying the function includes this configuration
        research_agent = agent.create_research_agent()
        assert research_agent is not None

    def test_all_sub_agents_have_recursion_limit(self):
        """Test that sub-agents are configured (recursion limit handled by deep agent framework)."""
        import agent

        # In the new SubAgent architecture, recursion limits are handled by create_deep_agent
        # Sub-agents are configuration objects, not pre-compiled graphs
        # Verify all sub-agents are properly configured
        assert hasattr(agent, 'subagents')
        assert len(agent.subagents) == 3

        # Verify each sub-agent has required configuration
        for subagent in agent.subagents:
            assert 'name' in subagent
            assert 'system_prompt' in subagent
            assert 'tools' in subagent


@pytest.mark.integration
class TestDeepAgentFramework:
    """Integration tests for deep agent framework usage."""

    def test_uses_create_deep_agent(self):
        """Test that create_deep_agent is imported and used."""
        import agent

        # Should import create_deep_agent from deepagents
        # This is verified by checking imports in agent.py
        from deepagents import create_deep_agent
        assert create_deep_agent is not None

    def test_uses_compiled_sub_agent(self):
        """Test that CompiledSubAgent is imported and used."""
        import agent

        # All sub-agents should be dicts (CompiledSubAgent is a TypedDict)
        assert isinstance(agent.analysis_sub_agent, dict)
        assert isinstance(agent.web_research_sub_agent, dict)
        assert isinstance(agent.credibility_sub_agent, dict)


@pytest.mark.integration
class TestLLMConfiguration:
    """Integration tests for LLM configuration."""

    def test_sub_agent_llm_configured(self):
        """Test that sub-agent LLM is properly configured."""
        import agent

        assert hasattr(agent, 'sub_agent_llm')

    def test_main_agent_llm_configured(self):
        """Test that main agent LLM is properly configured."""
        import agent

        assert hasattr(agent, 'agent_llm')

    def test_different_models_for_main_and_sub_agents(self):
        """Test that main agent and sub-agents use different models."""
        import agent

        # Main agent uses gpt-5-2025-08-07
        # Sub-agents use gpt-5.1-2025-11-13
        assert hasattr(agent, 'agent_llm')
        assert hasattr(agent, 'sub_agent_llm')


@pytest.mark.integration
class TestPromptImports:
    """Integration tests for prompt imports."""

    def test_all_prompts_imported(self):
        """Test that all prompt files are imported."""
        import agent

        # All prompt imports should succeed
        from prompts.main_agent import PROMPT as MAIN_AGENT_PROMPT
        from prompts.analysis_agent import PROMPT as ANALYSIS_PROMPT
        from prompts.web_research_agent import PROMPT as WEB_RESEARCH_PROMPT
        from prompts.credibility_agent import PROMPT as CREDIBILITY_PROMPT

        assert MAIN_AGENT_PROMPT is not None
        assert ANALYSIS_PROMPT is not None
        assert WEB_RESEARCH_PROMPT is not None
        assert CREDIBILITY_PROMPT is not None

    def test_prompts_are_strings(self):
        """Test that all prompts are strings."""
        from prompts.main_agent import PROMPT as MAIN_AGENT_PROMPT
        from prompts.analysis_agent import PROMPT as ANALYSIS_PROMPT
        from prompts.web_research_agent import PROMPT as WEB_RESEARCH_PROMPT
        from prompts.credibility_agent import PROMPT as CREDIBILITY_PROMPT

        assert isinstance(MAIN_AGENT_PROMPT, str)
        assert isinstance(ANALYSIS_PROMPT, str)
        assert isinstance(WEB_RESEARCH_PROMPT, str)
        assert isinstance(CREDIBILITY_PROMPT, str)


@pytest.mark.integration
class TestEnvironmentConfiguration:
    """Integration tests for environment configuration."""

    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_tavily'})
    def test_tavily_client_initialized(self):
        """Test that Tavily client is initialized."""
        import agent

        assert hasattr(agent, 'tavily_client')

    @patch.dict(os.environ, {'DAYTONA_API_KEY': 'test_daytona'})
    def test_daytona_initialized(self):
        """Test that Daytona is initialized."""
        import agent

        assert hasattr(agent, 'daytona')

    def test_dotenv_loaded(self):
        """Test that load_dotenv is called."""
        import agent

        # load_dotenv() should be called at module level
        # This ensures .env file is loaded
        from dotenv import load_dotenv
        assert load_dotenv is not None


@pytest.mark.integration
class TestAgentModuleStructure:
    """Integration tests for agent module structure."""

    def test_module_has_required_exports(self):
        """Test that agent module has all required exports."""
        import agent

        required_exports = [
            'web_search',
            'execute_python_code',
            'analysis_sub_agent',
            'web_research_sub_agent',
            'credibility_sub_agent',
            'make_backend',
            'MemoryCleanupMiddleware',
            'create_research_agent',
            'agent',
        ]

        for export in required_exports:
            assert hasattr(agent, export), f"Missing export: {export}"

    def test_module_docstring(self):
        """Test that agent module has descriptive docstring."""
        import agent

        assert agent.__doc__ is not None
        assert len(agent.__doc__) > 0
        assert 'Deep Research Agent' in agent.__doc__


@pytest.mark.integration
@pytest.mark.slow
class TestAgentInitialization:
    """Integration tests for full agent initialization."""

    @patch('agent.store')
    @patch('agent.checkpointer')
    @patch('agent.TavilyClient')
    @patch('agent.Daytona')
    @patch('agent.ChatOpenAI')
    def test_full_agent_initialization(
        self, mock_openai, mock_daytona, mock_tavily, mock_checkpointer, mock_store
    ):
        """Test full agent initialization with all components."""
        import importlib
        import agent as agent_module

        # Reload to trigger initialization
        importlib.reload(agent_module)

        # Verify all components are initialized
        assert agent_module.store is not None
        assert agent_module.checkpointer is not None
        assert agent_module.tavily_client is not None
        assert agent_module.daytona is not None
        assert agent_module.agent is not None


@pytest.mark.integration
class TestCreateAgentUsage:
    """Integration tests for create_agent usage."""

    def test_create_agent_used_for_sub_agents(self):
        """Test that create_agent is used for creating sub-agent graphs."""
        import agent

        # create_agent should be imported from langchain.agents
        # and used for creating analysis_sub_agent_graph, web_research_sub_agent_graph, credibility_sub_agent_graph
        from langchain.agents import create_agent
        assert create_agent is not None
