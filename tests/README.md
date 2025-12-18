# Deep Research Agent Test Suite

This test suite verifies that all components of the Deep Research Agent work correctly.

## What Gets Tested

The tests check:
- **Tool functions** - web search and Python code execution
- **Middleware** - memory cleanup and limits
- **Memory management** - storage and database connections
- **Sub-agents** - the 3 specialized agents (analysis, web research, credibility)
- **Main agent** - overall orchestration and configuration

## Test Organization

```
tests/
├── conftest.py                    # Test setup and fake data
├── unit/                          # Fast tests for individual pieces
│   ├── test_tools.py             # Web search & code execution
│   ├── test_middleware.py        # Memory cleanup logic
│   └── test_memory_management.py # Database & storage
└── integration/                   # Tests for how pieces work together
    ├── test_subagents.py         # 3 sub-agents configuration
    └── test_agent_orchestration.py # Main agent setup
```

## What Each File Does

### `conftest.py`
Sets up the test environment with fake versions of external services (no real API calls):
- Fake web search results
- Fake code execution sandbox
- Fake database
- Fake LLM responses

### `unit/test_tools.py` (~20 tests)
Tests the two main tool functions:
- `web_search()` - checks it calls the search API correctly
- `execute_python_code()` - checks code runs safely in sandbox

### `unit/test_middleware.py` (~15 tests)
Tests the memory cleanup system that prevents too many memories from building up.

### `unit/test_memory_management.py` (~18 tests)
Tests how memories are stored:
- Temporary vs permanent storage
- Database connections
- Memory cleanup by timestamp

### `integration/test_subagents.py` (~35 tests)
Tests all 3 sub-agents are configured correctly:
- Analysis agent has code execution tool
- Web research agent has search tool
- Credibility agent has search tool
- All have proper middleware and settings

### `integration/test_agent_orchestration.py` (~30 tests)
Tests the main agent setup:
- Includes all 3 sub-agents
- Uses correct LLM models
- Has memory cleanup enabled
- Environment variables loaded

## Running Tests

```bash
# Install dependencies first
pip install -r requirements.txt

# Run all tests
pytest

# Run with details
pytest -v

# Run only fast tests
pytest -m unit

# Run specific file
pytest tests/unit/test_tools.py
```

## What Happens When Tests Run

All external services are replaced with fake versions, so:
- No real API calls are made
- No actual money is spent
- Tests run fast (~10 seconds total)
- Your actual environment isn't affected

## Expected Output

When everything works, you'll see:
```
======================== test session starts ========================
collected 120 items

tests/unit/test_tools.py ......................                [ 18%]
tests/unit/test_middleware.py ...............                 [ 30%]
tests/unit/test_memory_management.py .........                [ 45%]
tests/integration/test_subagents.py ..........                [ 75%]
tests/integration/test_agent_orchestration.py .........       [100%]

======================== 120 passed in 10.23s =======================
```

## Common Issues

**Import errors?**
- Make sure you're in the project root directory
- Run `pip install -r requirements.txt`

**Tests failing?**
- Check the error message for which component failed
- Look at the test file to see what's being checked
