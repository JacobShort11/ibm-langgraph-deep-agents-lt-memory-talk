"""
Integration tests for sub-agent creation and configuration.

Updated for SubAgent-based architecture where sub-agents are defined as
configuration objects (SubAgent TypedDicts) rather than pre-compiled graphs.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the deep-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deep-agent'))


@pytest.mark.integration
class TestAnalysisSubAgent:
    """Integration tests for Analysis Sub-Agent."""

    def test_analysis_sub_agent_exists(self):
        """Test that analysis sub-agent is properly defined."""
        import agent

        assert hasattr(agent, 'analysis_sub_agent')
        assert agent.analysis_sub_agent is not None

    def test_analysis_sub_agent_name(self):
        """Test analysis sub-agent has correct name."""
        import agent

        assert agent.analysis_sub_agent['name'] == "analysis-agent"

    def test_analysis_sub_agent_description(self):
        """Test analysis sub-agent has description."""
        import agent

        description = agent.analysis_sub_agent['description']
        assert description is not None
        assert "data analysis" in description.lower() or "analysis" in description.lower()

    def test_analysis_sub_agent_has_system_prompt(self):
        """Test analysis sub-agent has system_prompt (SubAgent architecture)."""
        import agent

        # SubAgent objects have 'system_prompt' instead of 'runnable'
        assert 'system_prompt' in agent.analysis_sub_agent
        assert agent.analysis_sub_agent['system_prompt'] is not None
        assert len(agent.analysis_sub_agent['system_prompt']) > 0

    def test_analysis_sub_agent_tools(self):
        """Test analysis sub-agent is configured with correct tools."""
        import agent

        # SubAgent has 'tools' field
        assert 'tools' in agent.analysis_sub_agent
        assert len(agent.analysis_sub_agent['tools']) > 0
        # First tool should be execute_python_code
        assert agent.analysis_sub_agent['tools'][0].__name__ == 'execute_python_code'

    def test_analysis_sub_agent_middleware_configured(self):
        """Test analysis sub-agent has middleware."""
        import agent

        middleware = agent.analysis_sub_agent_middleware
        assert isinstance(middleware, list)
        assert len(middleware) >= 1  # At minimum has ToolCallLimitMiddleware

    def test_analysis_sub_agent_has_model(self):
        """Test analysis sub-agent has model configured."""
        import agent

        # SubAgent has 'model' field
        assert 'model' in agent.analysis_sub_agent
        assert agent.analysis_sub_agent['model'] is not None


@pytest.mark.integration
class TestWebResearchSubAgent:
    """Integration tests for Web Research Sub-Agent."""

    def test_web_research_sub_agent_exists(self):
        """Test that web research sub-agent is properly defined."""
        import agent

        assert hasattr(agent, 'web_research_sub_agent')
        assert agent.web_research_sub_agent is not None

    def test_web_research_sub_agent_name(self):
        """Test web research sub-agent has correct name."""
        import agent

        assert agent.web_research_sub_agent['name'] == "web-research-agent"

    def test_web_research_sub_agent_description(self):
        """Test web research sub-agent has description."""
        import agent

        description = agent.web_research_sub_agent['description']
        assert description is not None
        assert "web" in description.lower() or "research" in description.lower()

    def test_web_research_sub_agent_has_system_prompt(self):
        """Test web research sub-agent has system_prompt."""
        import agent

        assert 'system_prompt' in agent.web_research_sub_agent
        assert agent.web_research_sub_agent['system_prompt'] is not None

    def test_web_research_sub_agent_middleware_configured(self):
        """Test web research sub-agent has middleware."""
        import agent

        middleware = agent.web_research_sub_agent_middleware
        assert isinstance(middleware, list)
        assert len(middleware) >= 1


@pytest.mark.integration
class TestCredibilitySubAgent:
    """Integration tests for Credibility Sub-Agent."""

    def test_credibility_sub_agent_exists(self):
        """Test that credibility sub-agent is properly defined."""
        import agent

        assert hasattr(agent, 'credibility_sub_agent')
        assert agent.credibility_sub_agent is not None

    def test_credibility_sub_agent_name(self):
        """Test credibility sub-agent has correct name."""
        import agent

        assert agent.credibility_sub_agent['name'] == "credibility-agent"

    def test_credibility_sub_agent_description(self):
        """Test credibility sub-agent has description."""
        import agent

        description = agent.credibility_sub_agent['description']
        assert description is not None
        assert "credibility" in description.lower() or "fact" in description.lower()

    def test_credibility_sub_agent_has_system_prompt(self):
        """Test credibility sub-agent has system_prompt."""
        import agent

        assert 'system_prompt' in agent.credibility_sub_agent
        assert agent.credibility_sub_agent['system_prompt'] is not None

    def test_credibility_sub_agent_middleware_configured(self):
        """Test credibility sub-agent has middleware."""
        import agent

        middleware = agent.credibility_sub_agent_middleware
        assert isinstance(middleware, list)
        assert len(middleware) >= 1


@pytest.mark.integration
class TestSubAgentToolAssignments:
    """Integration tests for sub-agent tool assignments."""

    def test_analysis_agent_has_code_execution(self):
        """Test that analysis agent has execute_python_code tool."""
        import agent

        # Check the tools field in SubAgent
        assert 'tools' in agent.analysis_sub_agent
        tools = agent.analysis_sub_agent['tools']
        assert any(tool.__name__ == 'execute_python_code' for tool in tools)

    def test_web_research_agent_has_web_search(self):
        """Test that web research agent has web_search tool."""
        import agent

        assert 'tools' in agent.web_research_sub_agent
        tools = agent.web_research_sub_agent['tools']
        assert any(tool.__name__ == 'web_search' for tool in tools)

    def test_credibility_agent_has_web_search(self):
        """Test that credibility agent has web_search tool."""
        import agent

        assert 'tools' in agent.credibility_sub_agent
        tools = agent.credibility_sub_agent['tools']
        assert any(tool.__name__ == 'web_search' for tool in tools)


@pytest.mark.integration
class TestSubAgentLLMConfiguration:
    """Integration tests for sub-agent LLM configuration."""

    def test_sub_agents_use_gpt5(self):
        """Test that sub-agents use GPT-5 model."""
        import agent

        # sub_agent_llm should be configured with gpt-5.1-2025-11-13
        assert hasattr(agent, 'sub_agent_llm')
        assert agent.sub_agent_llm.model_name == "gpt-5.1-2025-11-13"

    def test_sub_agents_have_retry_configuration(self):
        """Test that sub-agents have max_retries configured."""
        import agent

        # ChatOpenAI should be initialized with max_retries=3
        assert hasattr(agent, 'sub_agent_llm')
        assert agent.sub_agent_llm.max_retries == 3


@pytest.mark.integration
class TestSubAgentPrompts:
    """Integration tests for sub-agent system prompts."""

    def test_analysis_agent_has_system_prompt(self):
        """Test that analysis agent uses correct system prompt."""
        import agent
        from prompts.analysis_agent import PROMPT as ANALYSIS_PROMPT

        # Verify the prompt is imported and used
        assert ANALYSIS_PROMPT is not None
        assert len(ANALYSIS_PROMPT) > 0
        # Verify it's actually used in the SubAgent
        assert agent.analysis_sub_agent['system_prompt'] == ANALYSIS_PROMPT

    def test_web_research_agent_has_system_prompt(self):
        """Test that web research agent uses correct system prompt."""
        import agent
        from prompts.web_research_agent import PROMPT as WEB_RESEARCH_PROMPT

        assert WEB_RESEARCH_PROMPT is not None
        assert len(WEB_RESEARCH_PROMPT) > 0
        assert agent.web_research_sub_agent['system_prompt'] == WEB_RESEARCH_PROMPT

    def test_credibility_agent_has_system_prompt(self):
        """Test that credibility agent uses correct system prompt."""
        import agent
        from prompts.credibility_agent import PROMPT as CREDIBILITY_PROMPT

        assert CREDIBILITY_PROMPT is not None
        assert len(CREDIBILITY_PROMPT) > 0
        assert agent.credibility_sub_agent['system_prompt'] == CREDIBILITY_PROMPT


@pytest.mark.integration
class TestSubAgentConfiguration:
    """Integration tests for SubAgent instances (new architecture)."""

    def test_all_sub_agents_are_subagent_dicts(self):
        """Test that all sub-agents are SubAgent TypedDicts."""
        import agent

        # SubAgent is a TypedDict, so instances are dicts
        assert isinstance(agent.analysis_sub_agent, dict)
        assert isinstance(agent.web_research_sub_agent, dict)
        assert isinstance(agent.credibility_sub_agent, dict)

    def test_subagents_have_required_fields(self):
        """Test that SubAgent instances have name, description, system_prompt, tools, model, middleware."""
        import agent

        required_fields = ['name', 'description', 'system_prompt', 'tools', 'model', 'middleware']

        for sub_agent in [
            agent.analysis_sub_agent,
            agent.web_research_sub_agent,
            agent.credibility_sub_agent,
        ]:
            for field in required_fields:
                assert field in sub_agent, f"Missing field '{field}' in sub-agent {sub_agent['name']}"

            assert isinstance(sub_agent['name'], str)
            assert isinstance(sub_agent['description'], str)
            assert isinstance(sub_agent['system_prompt'], str)
            assert isinstance(sub_agent['tools'], list)
            assert isinstance(sub_agent['middleware'], list)


@pytest.mark.integration
class TestSubAgentDescriptions:
    """Integration tests for sub-agent description quality."""

    def test_analysis_agent_description_mentions_capabilities(self):
        """Test analysis agent description mentions key capabilities."""
        import agent

        description = agent.analysis_sub_agent['description'].lower()
        # Should mention data, analysis, or visualization
        assert any(keyword in description for keyword in [
            'data', 'analysis', 'visualization', 'chart', 'graph'
        ])

    def test_web_research_agent_description_mentions_capabilities(self):
        """Test web research agent description mentions key capabilities."""
        import agent

        description = agent.web_research_sub_agent['description'].lower()
        # Should mention web, search, or research
        assert any(keyword in description for keyword in [
            'web', 'search', 'research', 'information', 'internet'
        ])

    def test_credibility_agent_description_mentions_capabilities(self):
        """Test credibility agent description mentions key capabilities."""
        import agent

        description = agent.credibility_sub_agent['description'].lower()
        # Should mention fact-checking, verification, or credibility
        assert any(keyword in description for keyword in [
            'credibility', 'fact', 'verify', 'check', 'source', 'trust'
        ])


@pytest.mark.integration
class TestSubAgentMiddlewareStacks:
    """Integration tests for complete middleware stacks."""

    def test_middleware_includes_tool_limit(self):
        """Test all sub-agents have tool call limit middleware."""
        import agent
        from langchain.agents.middleware import ToolCallLimitMiddleware

        # Each should have ToolCallLimitMiddleware
        for middleware_list in [
            agent.analysis_sub_agent_middleware,
            agent.web_research_sub_agent_middleware,
            agent.credibility_sub_agent_middleware,
        ]:
            assert len(middleware_list) >= 1
            # Check that at least one middleware is ToolCallLimitMiddleware
            assert any(isinstance(m, ToolCallLimitMiddleware) for m in middleware_list)

    def test_subagents_list_exists(self):
        """Test that subagents list is properly defined."""
        import agent

        assert hasattr(agent, 'subagents')
        assert isinstance(agent.subagents, list)
        assert len(agent.subagents) == 3

    def test_subagents_list_contains_all_agents(self):
        """Test that subagents list contains all three sub-agents."""
        import agent

        agent_names = [sa['name'] for sa in agent.subagents]
        assert 'analysis-agent' in agent_names
        assert 'web-research-agent' in agent_names
        assert 'credibility-agent' in agent_names
