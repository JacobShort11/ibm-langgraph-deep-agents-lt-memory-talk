"""
Memory Backend Configuration - PostgreSQL store and backend routing.

Provides shared resources for persistent long-term memory:
- PostgreSQL store for memory files (persistent across sessions)
- Backend factory that routes /memories/ to persistent storage
- Checkpointing is handled by in-memory MemorySaver (per-session state)
"""

import os
from dotenv import load_dotenv
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.postgres import PostgresStore

load_dotenv()


# =============================================================================
# DATABASE (Required)
# =============================================================================

DATABASE_URL = os.environ["DATABASE_URL"]

# Context manager needs to be entered to get the actual store object
# This will be managed by the application lifecycle
_store_cm = PostgresStore.from_conn_string(DATABASE_URL)
store = _store_cm.__enter__()


# =============================================================================
# BACKEND (Ephemeral + Persistent Memory)
# =============================================================================

def make_backend(runtime):
    """
    Persistent for /memories/, ephemeral for everything else.

    This ensures that:
    - Files in /memories/ are stored in PostgreSQL (persistent across sessions)
    - All other files are stored in graph state (ephemeral, cleared between runs)
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )
