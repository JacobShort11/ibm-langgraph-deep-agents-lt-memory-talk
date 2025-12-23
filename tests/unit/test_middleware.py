"""Lightweight middleware sanity checks."""

import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
def test_memory_cleanup_defaults():
    from middleware.memory_cleanup import MemoryCleanupMiddleware

    store = MagicMock()
    middleware = MemoryCleanupMiddleware(store)

    assert middleware.max_memories == 30
    assert middleware.store is store


@pytest.mark.unit
def test_trim_prompt_contains_limit_placeholder():
    from middleware.memory_cleanup import TRIM_SYSTEM_PROMPT

    assert "{max_memories}" in TRIM_SYSTEM_PROMPT
    assert "{current_content}" in TRIM_SYSTEM_PROMPT
