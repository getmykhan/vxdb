"""HTTP client for vxdb server mode."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class VexClientError(Exception):
    """Raised when the server returns an error."""
    pass


class RemoteCollection:
    """Proxy for a collection on a remote vxdb server."""

    def __init__(self, client: "Client", name: str):
        self._client = client
        self.name = name

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ids": ids, "vectors": vectors}
        if metadata is not None:
            payload["metadata"] = metadata
        if documents is not None:
            payload["documents"] = documents
        return self._client._post(f"/collections/{self.name}/upsert", payload)

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"vector": vector, "top_k": top_k}
        if filter is not None:
            payload["filter"] = filter
        resp = self._client._post(f"/collections/{self.name}/query", payload)
        return resp["results"]

    def hybrid_query(
        self,
        vector: List[float],
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "vector": vector,
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
        }
        resp = self._client._post(f"/collections/{self.name}/hybrid", payload)
        return resp["results"]

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        resp = self._client._post(f"/collections/{self.name}/keyword", payload)
        return resp["results"]

    def delete(self, ids: List[str]) -> List[bool]:
        resp = self._client._post(f"/collections/{self.name}/delete", {"ids": ids})
        return resp["deleted"]

    def count(self) -> int:
        resp = self._client._get(f"/collections/{self.name}/count")
        return resp["count"]

    def __repr__(self) -> str:
        return f"RemoteCollection(name='{self.name}')"


class Client:
    """HTTP client for a remote vxdb server.

    Example::

        client = Client("http://localhost:8080")
        coll = client.create_collection("docs", dimension=384)
        coll.upsert(ids=["a"], vectors=[[0.1, ...]])
        results = coll.query(vector=[0.1, ...], top_k=5)
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        if httpx is None:
            raise ImportError(
                "httpx is required for server mode. Install it with: "
                "pip install 'vxdb[server]'"
            )
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=30.0)

    def _post(self, path: str, json: Any) -> Any:
        resp = self._http.post(path, json=json)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise VexClientError(f"{resp.status_code}: {detail}")
        if resp.status_code == 204:
            return {}
        return resp.json()

    def _get(self, path: str) -> Any:
        resp = self._http.get(path)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise VexClientError(f"{resp.status_code}: {detail}")
        return resp.json()

    def _delete(self, path: str) -> None:
        resp = self._http.delete(path)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise VexClientError(f"{resp.status_code}: {detail}")

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index: str = "flat",
    ) -> RemoteCollection:
        self._post("/collections", {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index": index,
        })
        return RemoteCollection(self, name)

    def get_collection(self, name: str) -> RemoteCollection:
        return RemoteCollection(self, name)

    def list_collections(self) -> List[str]:
        resp = self._get("/collections")
        return resp["collections"]

    def delete_collection(self, name: str) -> None:
        self._delete(f"/collections/{name}")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client(base_url='{self._base_url}')"
