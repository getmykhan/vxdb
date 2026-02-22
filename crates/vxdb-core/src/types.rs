use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type VectorData = Vec<f32>;
pub type Metadata = HashMap<String, serde_json::Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetricKind {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    Flat,
    Hnsw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub vector: VectorData,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetricKind,
    pub index_kind: IndexKind,
}

impl CollectionConfig {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            metric: DistanceMetricKind::Cosine,
            index_kind: IndexKind::Flat,
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetricKind) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_index(mut self, index_kind: IndexKind) -> Self {
        self.index_kind = index_kind;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document {
            id: "test-1".into(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: HashMap::from([("color".into(), serde_json::json!("red"))]),
        };
        assert_eq!(doc.id, "test-1");
        assert_eq!(doc.vector.len(), 3);
        assert_eq!(doc.metadata["color"], "red");
    }

    #[test]
    fn test_document_serde_roundtrip() {
        let doc = Document {
            id: "serde-1".into(),
            vector: vec![0.5, -1.0, 3.14],
            metadata: HashMap::from([
                ("tag".into(), serde_json::json!("important")),
                ("score".into(), serde_json::json!(42)),
            ]),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let recovered: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.id, doc.id);
        assert_eq!(recovered.vector, doc.vector);
        assert_eq!(recovered.metadata, doc.metadata);
    }

    #[test]
    fn test_search_result_ordering() {
        let mut results = vec![
            SearchResult { id: "a".into(), score: 0.8, metadata: HashMap::new() },
            SearchResult { id: "b".into(), score: 0.95, metadata: HashMap::new() },
            SearchResult { id: "c".into(), score: 0.6, metadata: HashMap::new() },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(results[0].id, "b");
        assert_eq!(results[1].id, "a");
        assert_eq!(results[2].id, "c");
    }

    #[test]
    fn test_collection_config_builder() {
        let cfg = CollectionConfig::new("my_docs", 384)
            .with_metric(DistanceMetricKind::Euclidean)
            .with_index(IndexKind::Hnsw);
        assert_eq!(cfg.name, "my_docs");
        assert_eq!(cfg.dimension, 384);
        assert_eq!(cfg.metric, DistanceMetricKind::Euclidean);
        assert_eq!(cfg.index_kind, IndexKind::Hnsw);
    }

    #[test]
    fn test_collection_config_defaults() {
        let cfg = CollectionConfig::new("default_test", 128);
        assert_eq!(cfg.metric, DistanceMetricKind::Cosine);
        assert_eq!(cfg.index_kind, IndexKind::Flat);
    }

    #[test]
    fn test_distance_metric_kind_serde() {
        let kinds = vec![
            DistanceMetricKind::Cosine,
            DistanceMetricKind::Euclidean,
            DistanceMetricKind::DotProduct,
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let recovered: DistanceMetricKind = serde_json::from_str(&json).unwrap();
            assert_eq!(recovered, kind);
        }
    }
}
