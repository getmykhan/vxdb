"""Agent-focused layer for vxdb.

vxdb as a data structure you *allocate*, not a service you *connect to*. The first
primitive is :class:`WorkingMemory`, an ephemeral, in-process scratchpad an agent
can consult on every step of its loop.
"""

from vxdb.agent.memory import Recall, WorkingMemory, scratch

__all__ = ["WorkingMemory", "Recall", "scratch"]
