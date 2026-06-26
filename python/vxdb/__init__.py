"""vxdb — A lightweight, high-performance vector store."""

from vxdb.client import Client
from vxdb.embedded import Collection, Database
from vxdb.embedding import EmbeddingFunction

__all__ = ["Database", "Collection", "Client", "EmbeddingFunction"]


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` lazily, on first access only.

    Computing it eagerly at import time via ``importlib.metadata`` adds ~40 ms
    to ``import vxdb`` (it imports the metadata/email machinery and scans
    ``site-packages`` ``.dist-info``). Deferring it keeps import startup at a
    few milliseconds while still single-sourcing the version from package
    metadata — so it can never drift from pyproject/Cargo. The result is cached
    into module globals, so repeated ``vxdb.__version__`` access is free.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            resolved = version("vxdb")
        except PackageNotFoundError:  # running from a source tree without an installed dist
            resolved = "0.0.0+source"
        globals()["__version__"] = resolved
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
