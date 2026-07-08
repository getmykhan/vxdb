pub mod flat;
pub mod hnsw;

use crate::error::VexResult;
use crate::types::{Metadata, SearchResult, VectorData};

pub trait VectorIndex: Send + Sync {
    fn insert(&mut self, id: String, vector: VectorData, metadata: Metadata) -> VexResult<()>;

    /// Search with an optional per-query `ef_search` override.
    ///
    /// Higher `ef_search` explores more of the graph, trading latency for recall.
    /// `None` uses the index's configured default. Exact indexes (flat) ignore
    /// the hint and always return exact results.
    fn search_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: Option<usize>,
    ) -> VexResult<Vec<SearchResult>>;

    /// Search with the index's configured default `ef_search`.
    fn search(&self, query: &[f32], top_k: usize) -> VexResult<Vec<SearchResult>> {
        self.search_ef(query, top_k, None)
    }

    fn delete(&mut self, id: &str) -> VexResult<bool>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn contains(&self, id: &str) -> bool;
}
