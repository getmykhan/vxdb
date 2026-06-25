"""vxdb — A lightweight, high-performance vector store."""

from importlib.metadata import PackageNotFoundError, version

from vxdb.client import Client
from vxdb.embedded import Collection, Database
from vxdb.embedding import EmbeddingFunction

__all__ = ["Database", "Collection", "Client", "EmbeddingFunction"]

try:
    # Single-source the version from installed package metadata so it can never
    # drift from pyproject/Cargo again.
    __version__ = version("vxdb")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = "0.0.0+source"
