"""Memory backend routing.

Routes /memories/ to the runtime-provided store (LangGraph-managed
persistence) and keeps everything else ephemeral in state.
"""

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


def make_backend(runtime):
    """
    Persistent for /memories/, ephemeral for everything else.

    The runtime supplies the store; LangGraph API handles persistence so
    we don't provision our own store here.
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )
