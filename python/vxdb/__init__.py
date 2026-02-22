"""vxdb — A lightweight, high-performance vector store."""

from vxdb._vxdb import Database, Collection
from vxdb.client import Client

__all__ = ["Database", "Collection", "Client"]
__version__ = "0.1.0"
