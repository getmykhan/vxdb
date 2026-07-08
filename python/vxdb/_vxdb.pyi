"""Type stubs for the vxdb native extension (_vxdb)."""

from typing import Any, TypedDict

class SearchResult(TypedDict):
    id: str
    score: float
    metadata: dict[str, Any]
    document: str | None

class Collection:
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = ...,
        documents: list[str] | None = ...,
    ) -> None: ...
    def query(
        self,
        vector: list[float],
        top_k: int = ...,
        filter: dict[str, Any] | None = ...,
        ef_search: int | None = ...,
    ) -> list[SearchResult]: ...
    def hybrid_query(
        self,
        vector: list[float],
        query: str,
        top_k: int = ...,
        alpha: float = ...,
    ) -> list[SearchResult]: ...
    def keyword_search(self, query: str, top_k: int = ...) -> list[SearchResult]: ...
    def delete(self, ids: list[str]) -> list[bool]: ...
    def count(self) -> int: ...

class Database:
    def __init__(self, path: str | None = ...) -> None: ...
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = ...,
        index: str = ...,
    ) -> Collection: ...
    def get_collection(self, name: str) -> Collection: ...
    def list_collections(self) -> list[str]: ...
    def delete_collection(self, name: str) -> None: ...
