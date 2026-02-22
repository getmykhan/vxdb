pub mod flat;
pub mod hnsw;

use crate::error::VexResult;
use crate::types::{Metadata, SearchResult, VectorData};

pub trait VectorIndex: Send + Sync {
    fn insert(&mut self, id: String, vector: VectorData, metadata: Metadata) -> VexResult<()>;
    fn search(&self, query: &[f32], top_k: usize) -> VexResult<Vec<SearchResult>>;
    fn delete(&mut self, id: &str) -> VexResult<bool>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn contains(&self, id: &str) -> bool;
}
