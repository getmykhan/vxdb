use std::collections::HashMap;

use crate::distance::DistanceMetric;
use crate::error::{VexError, VexResult};
use crate::types::{Metadata, SearchResult, VectorData};

use super::VectorIndex;

pub struct FlatIndex {
    dimension: usize,
    metric: Box<dyn DistanceMetric>,
    vectors: Vec<VectorData>,
    ids: Vec<String>,
    metadata: Vec<Metadata>,
    id_to_idx: HashMap<String, usize>,
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: Box<dyn DistanceMetric>) -> Self {
        Self {
            dimension,
            metric,
            vectors: Vec::new(),
            ids: Vec::new(),
            metadata: Vec::new(),
            id_to_idx: HashMap::new(),
        }
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&mut self, id: String, vector: VectorData, metadata: Metadata) -> VexResult<()> {
        if vector.len() != self.dimension {
            return Err(VexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if vector.is_empty() {
            return Err(VexError::EmptyVector);
        }

        if let Some(&idx) = self.id_to_idx.get(&id) {
            self.vectors[idx] = vector;
            self.metadata[idx] = metadata;
            return Ok(());
        }

        let idx = self.vectors.len();
        self.id_to_idx.insert(id.clone(), idx);
        self.ids.push(id);
        self.vectors.push(vector);
        self.metadata.push(metadata);
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> VexResult<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(VexError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.metric.distance(query, v)))
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results = scored
            .into_iter()
            .map(|(idx, dist)| SearchResult {
                id: self.ids[idx].clone(),
                score: dist,
                metadata: self.metadata[idx].clone(),
            })
            .collect();

        Ok(results)
    }

    fn delete(&mut self, id: &str) -> VexResult<bool> {
        let idx = match self.id_to_idx.remove(id) {
            Some(idx) => idx,
            None => return Ok(false),
        };

        let last = self.vectors.len() - 1;
        if idx != last {
            self.vectors.swap(idx, last);
            self.ids.swap(idx, last);
            self.metadata.swap(idx, last);
            self.id_to_idx.insert(self.ids[idx].clone(), idx);
        }

        self.vectors.pop();
        self.ids.pop();
        self.metadata.pop();
        Ok(true)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn contains(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::CosineDistance;
    use std::collections::HashMap;

    fn make_index() -> FlatIndex {
        FlatIndex::new(3, Box::new(CosineDistance))
    }

    #[test]
    fn test_insert_and_len() {
        let mut idx = make_index();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());

        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
        assert!(idx.contains("a"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut idx = make_index();
        let result = idx.insert("a".into(), vec![1.0, 0.0], HashMap::new());
        assert!(matches!(result, Err(VexError::DimensionMismatch { expected: 3, got: 2 })));
    }

    #[test]
    fn test_upsert_overwrites() {
        let mut idx = make_index();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("a".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_search_exact_match() {
        let mut idx = make_index();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();
        idx.insert("c".into(), vec![1.0, 0.1, 0.0], HashMap::new()).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].score < 0.01); // near-zero distance for exact match
    }

    #[test]
    fn test_search_top_k_bounds() {
        let mut idx = make_index();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1); // only 1 vector, even though top_k=10
    }

    #[test]
    fn test_delete() {
        let mut idx = make_index();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();

        assert!(idx.delete("a").unwrap());
        assert_eq!(idx.len(), 1);
        assert!(!idx.contains("a"));
        assert!(idx.contains("b"));

        assert!(!idx.delete("nonexistent").unwrap());
    }

    #[test]
    fn test_delete_and_search() {
        let mut idx = make_index();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();
        idx.insert("c".into(), vec![0.0, 0.0, 1.0], HashMap::new()).unwrap();

        idx.delete("a").unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "a"));
    }
}
