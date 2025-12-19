"""
Memory Backend Configuration - PostgreSQL store and backend routing.

Provides shared resources for persistent long-term memory:
- PostgreSQL store for memory files
- PostgreSQL checkpointer for conversation state
- Backend factory that routes /memories/ to persistent storage
"""

import os
from dotenv import load_dotenv
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()


# =============================================================================
# DATABASE (Required)
# =============================================================================

DATABASE_URL = os.environ["DATABASE_URL"]
store = PostgresStore.from_conn_string(DATABASE_URL)
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)


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
