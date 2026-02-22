use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;

use crate::distance::metric_for_kind;
use crate::error::{VexError, VexResult};
use crate::filter::Filter;
use crate::hybrid::{self, Bm25Index};
use crate::index::flat::FlatIndex;
use crate::index::hnsw::{HnswConfig, HnswIndex};
use crate::index::VectorIndex;
use crate::storage::persistent::PersistentStorage;
use crate::storage::wal::{WalEntry, WriteAheadLog};
use crate::types::{CollectionConfig, DistanceMetricKind, IndexKind, Metadata, SearchResult, VectorData};

pub struct Collection {
    pub config: CollectionConfig,
    index: Box<dyn VectorIndex>,
    text_index: Bm25Index,
    storage: Option<PersistentStorage>,
}

impl Collection {
    pub fn new(config: CollectionConfig) -> Self {
        let index = Self::make_index(&config);
        Self {
            config,
            index,
            text_index: Bm25Index::new(),
            storage: None,
        }
    }

    /// Create a new persistent collection, writing config to disk.
    pub fn create_persistent(config: CollectionConfig, path: &Path) -> VexResult<Self> {
        fs::create_dir_all(path)?;
        let config_json = serde_json::to_string_pretty(&config)?;
        fs::write(path.join("config.json"), config_json)?;

        let storage = PersistentStorage::create(path, config.dimension)?;
        let index = Self::make_index(&config);
        Ok(Self {
            config,
            index,
            text_index: Bm25Index::new(),
            storage: Some(storage),
        })
    }

    /// Open an existing persistent collection from disk, reloading all data.
    pub fn open(path: &Path) -> VexResult<Self> {
        let config_bytes = fs::read(path.join("config.json"))?;
        let config: CollectionConfig = serde_json::from_slice(&config_bytes)?;

        let storage = PersistentStorage::open(path)?;
        let entries = storage.load_all()?;

        let mut index = Self::make_index(&config);
        let mut text_index = Bm25Index::new();

        for (id, vector, metadata, document) in entries {
            index.insert(id.clone(), vector, metadata)?;
            if let Some(doc) = &document {
                text_index.insert(&id, doc);
            }
        }

        Ok(Self {
            config,
            index,
            text_index,
            storage: Some(storage),
        })
    }

    fn make_index(config: &CollectionConfig) -> Box<dyn VectorIndex> {
        let metric = metric_for_kind(config.metric);
        match config.index_kind {
            IndexKind::Flat => Box::new(FlatIndex::new(config.dimension, metric)),
            IndexKind::Hnsw => Box::new(HnswIndex::new(
                config.dimension,
                metric,
                HnswConfig::default(),
            )),
        }
    }

    pub fn upsert(&mut self, id: String, vector: VectorData, metadata: Metadata) -> VexResult<()> {
        if let Some(storage) = &mut self.storage {
            storage.save_entry(&id, &vector, &metadata, None)?;
        }
        self.index.insert(id, vector, metadata)
    }

    pub fn upsert_with_doc(
        &mut self,
        id: String,
        vector: VectorData,
        metadata: Metadata,
        document: &str,
    ) -> VexResult<()> {
        if let Some(storage) = &mut self.storage {
            storage.save_entry(&id, &vector, &metadata, Some(document))?;
        }
        self.index.insert(id.clone(), vector, metadata)?;
        self.text_index.insert(&id, document);
        Ok(())
    }

    pub fn upsert_batch(
        &mut self,
        ids: Vec<String>,
        vectors: Vec<VectorData>,
        metadata: Vec<Metadata>,
    ) -> VexResult<()> {
        if ids.len() != vectors.len() || ids.len() != metadata.len() {
            return Err(VexError::Internal(
                "ids, vectors, and metadata must have the same length".into(),
            ));
        }
        if let Some(storage) = &mut self.storage {
            let batch: Vec<_> = ids
                .iter()
                .zip(vectors.iter())
                .zip(metadata.iter())
                .map(|((id, v), m)| (id.clone(), v.clone(), m.clone(), None))
                .collect();
            storage.save_batch(&batch)?;
        }
        for ((id, vec), meta) in ids.into_iter().zip(vectors).zip(metadata) {
            self.index.insert(id, vec, meta)?;
        }
        Ok(())
    }

    pub fn upsert_batch_with_docs(
        &mut self,
        ids: Vec<String>,
        vectors: Vec<VectorData>,
        metadata: Vec<Metadata>,
        documents: Vec<String>,
    ) -> VexResult<()> {
        if ids.len() != vectors.len() || ids.len() != metadata.len() || ids.len() != documents.len() {
            return Err(VexError::Internal(
                "ids, vectors, metadata, and documents must have the same length".into(),
            ));
        }
        if let Some(storage) = &mut self.storage {
            let batch: Vec<_> = ids
                .iter()
                .zip(vectors.iter())
                .zip(metadata.iter())
                .zip(documents.iter())
                .map(|(((id, v), m), d)| (id.clone(), v.clone(), m.clone(), Some(d.clone())))
                .collect();
            storage.save_batch(&batch)?;
        }
        for (((id, vec), meta), doc) in ids.into_iter().zip(vectors).zip(metadata).zip(documents) {
            self.index.insert(id.clone(), vec, meta)?;
            self.text_index.insert(&id, &doc);
        }
        Ok(())
    }

    pub fn query(&self, vector: &[f32], top_k: usize) -> VexResult<Vec<SearchResult>> {
        self.index.search(vector, top_k)
    }

    pub fn query_with_filter(
        &self,
        vector: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> VexResult<Vec<SearchResult>> {
        let fetch_k = (top_k * 10).max(100);
        let candidates = self.index.search(vector, fetch_k)?;
        let filtered: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(top_k)
            .collect();
        Ok(filtered)
    }

    pub fn hybrid_query(
        &self,
        vector: &[f32],
        query_text: &str,
        top_k: usize,
        alpha: f32,
    ) -> VexResult<Vec<SearchResult>> {
        let fetch_k = (top_k * 10).max(100);

        let vector_results = self.index.search(vector, fetch_k)?;
        let keyword_results = self.text_index.search(query_text, fetch_k);

        let fused = hybrid::reciprocal_rank_fusion(&vector_results, &keyword_results, top_k, 60, alpha);

        let all_results = self.index.search(vector, fetch_k.max(self.index.len()))?;
        let meta_map: HashMap<String, Metadata> = all_results
            .into_iter()
            .map(|r| (r.id, r.metadata))
            .collect();

        let results = fused
            .into_iter()
            .map(|(id, score)| SearchResult {
                metadata: meta_map.get(&id).cloned().unwrap_or_default(),
                id,
                score,
            })
            .collect();

        Ok(results)
    }

    pub fn keyword_search(&self, query: &str, top_k: usize) -> VexResult<Vec<SearchResult>> {
        let results = self.text_index.search(query, top_k);

        let all_results = if !results.is_empty() {
            let dim = self.config.dimension;
            let dummy = vec![0.0f32; dim];
            self.index.search(&dummy, self.index.len()).unwrap_or_default()
        } else {
            Vec::new()
        };
        let meta_map: HashMap<String, Metadata> = all_results
            .into_iter()
            .map(|r| (r.id, r.metadata))
            .collect();

        let out = results
            .into_iter()
            .map(|(id, score)| SearchResult {
                metadata: meta_map.get(&id).cloned().unwrap_or_default(),
                id,
                score: score as f32,
            })
            .collect();

        Ok(out)
    }

    pub fn delete(&mut self, id: &str) -> VexResult<bool> {
        if let Some(storage) = &self.storage {
            storage.delete_entry(id)?;
        }
        self.text_index.remove(id);
        self.index.delete(id)
    }

    pub fn count(&self) -> usize {
        self.index.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.index.contains(id)
    }
}

pub struct Database {
    collections: RwLock<HashMap<String, RwLock<Collection>>>,
    base_path: Option<PathBuf>,
    wal: Option<RwLock<WriteAheadLog>>,
}

impl Database {
    /// Create an in-memory database (no persistence).
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            base_path: None,
            wal: None,
        }
    }

    /// Open or create a persistent database at the given path.
    pub fn open(path: &Path) -> VexResult<Self> {
        fs::create_dir_all(path)?;
        let collections_dir = path.join("collections");
        fs::create_dir_all(&collections_dir)?;

        let mut collections = HashMap::new();

        // Load existing collections from disk
        if collections_dir.exists() {
            for entry in fs::read_dir(&collections_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    let coll_path = entry.path();
                    if coll_path.join("config.json").exists() {
                        let coll = Collection::open(&coll_path)?;
                        let name = coll.config.name.clone();
                        collections.insert(name, RwLock::new(coll));
                    }
                }
            }
        }

        // Open WAL and replay un-checkpointed entries
        let wal_path = path.join("vxdb.wal");
        let wal = if wal_path.exists() {
            WriteAheadLog::open(&wal_path)?
        } else {
            WriteAheadLog::create(&wal_path)?
        };

        let entries = wal.replay()?;

        // Replay WAL entries into in-memory state + disk storage
        for entry in &entries {
            match entry {
                WalEntry::CreateCollection {
                    name,
                    dimension,
                    metric,
                    index_kind,
                } => {
                    if !collections.contains_key(name) {
                        let config = CollectionConfig::new(name, *dimension)
                            .with_metric(parse_metric_str(metric))
                            .with_index(parse_index_str(index_kind));
                        let coll_path = collections_dir.join(name);
                        let coll = Collection::create_persistent(config, &coll_path)?;
                        collections.insert(name.clone(), RwLock::new(coll));
                    }
                }
                WalEntry::DeleteCollection { name } => {
                    collections.remove(name);
                    let coll_path = collections_dir.join(name);
                    if coll_path.exists() {
                        let _ = fs::remove_dir_all(&coll_path);
                    }
                }
                WalEntry::Upsert {
                    collection,
                    id,
                    vector,
                    metadata,
                    document,
                } => {
                    if let Some(coll_lock) = collections.get(collection) {
                        let mut coll = coll_lock.write();
                        if let Some(doc) = document {
                            coll.upsert_with_doc(id.clone(), vector.clone(), metadata.clone(), doc)?;
                        } else {
                            coll.upsert(id.clone(), vector.clone(), metadata.clone())?;
                        }
                    }
                }
                WalEntry::Delete { collection, id } => {
                    if let Some(coll_lock) = collections.get(collection) {
                        let mut coll = coll_lock.write();
                        coll.delete(id)?;
                    }
                }
                WalEntry::Checkpoint => {}
            }
        }

        // Checkpoint to clear replayed entries
        let mut wal = wal;
        if !entries.is_empty() {
            wal.checkpoint()?;
        }

        Ok(Self {
            collections: RwLock::new(collections),
            base_path: Some(path.to_path_buf()),
            wal: Some(RwLock::new(wal)),
        })
    }

    pub fn create_collection(&self, config: CollectionConfig) -> VexResult<()> {
        let mut collections = self.collections.write();
        if collections.contains_key(&config.name) {
            return Err(VexError::CollectionAlreadyExists(config.name.clone()));
        }

        if let Some(wal) = &self.wal {
            let mut wal = wal.write();
            wal.append(&WalEntry::CreateCollection {
                name: config.name.clone(),
                dimension: config.dimension,
                metric: format!("{:?}", config.metric),
                index_kind: format!("{:?}", config.index_kind),
            })?;
        }

        let name = config.name.clone();
        let coll = if let Some(base) = &self.base_path {
            let coll_path = base.join("collections").join(&name);
            Collection::create_persistent(config, &coll_path)?
        } else {
            Collection::new(config)
        };

        collections.insert(name, RwLock::new(coll));
        Ok(())
    }

    pub fn list_collections(&self) -> Vec<String> {
        let collections = self.collections.read();
        collections.keys().cloned().collect()
    }

    pub fn delete_collection(&self, name: &str) -> VexResult<()> {
        let mut collections = self.collections.write();
        if collections.remove(name).is_none() {
            return Err(VexError::CollectionNotFound(name.into()));
        }

        if let Some(wal) = &self.wal {
            let mut wal = wal.write();
            wal.append(&WalEntry::DeleteCollection {
                name: name.into(),
            })?;
        }

        if let Some(base) = &self.base_path {
            let coll_path = base.join("collections").join(name);
            if coll_path.exists() {
                fs::remove_dir_all(&coll_path)?;
            }
        }

        Ok(())
    }

    pub fn with_collection<F, R>(&self, name: &str, f: F) -> VexResult<R>
    where
        F: FnOnce(&Collection) -> VexResult<R>,
    {
        let collections = self.collections.read();
        let coll = collections
            .get(name)
            .ok_or_else(|| VexError::CollectionNotFound(name.into()))?;
        let coll = coll.read();
        f(&coll)
    }

    pub fn with_collection_mut<F, R>(&self, name: &str, f: F) -> VexResult<R>
    where
        F: FnOnce(&mut Collection) -> VexResult<R>,
    {
        let collections = self.collections.read();
        let coll = collections
            .get(name)
            .ok_or_else(|| VexError::CollectionNotFound(name.into()))?;
        let mut coll = coll.write();
        f(&mut coll)
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_metric_str(s: &str) -> DistanceMetricKind {
    match s {
        "Cosine" => DistanceMetricKind::Cosine,
        "Euclidean" => DistanceMetricKind::Euclidean,
        "DotProduct" => DistanceMetricKind::DotProduct,
        _ => DistanceMetricKind::Cosine,
    }
}

fn parse_index_str(s: &str) -> IndexKind {
    match s {
        "Flat" => IndexKind::Flat,
        "Hnsw" => IndexKind::Hnsw,
        _ => IndexKind::Flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CollectionConfig, DistanceMetricKind};
    use std::collections::HashMap;

    fn make_db() -> Database {
        Database::new()
    }

    fn sample_config(name: &str) -> CollectionConfig {
        CollectionConfig::new(name, 3).with_metric(DistanceMetricKind::Cosine)
    }

    #[test]
    fn test_create_and_list_collections() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();
        db.create_collection(sample_config("images")).unwrap();

        let mut names = db.list_collections();
        names.sort();
        assert_eq!(names, vec!["docs", "images"]);
    }

    #[test]
    fn test_create_duplicate_collection_fails() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();
        let result = db.create_collection(sample_config("docs"));
        assert!(matches!(result, Err(VexError::CollectionAlreadyExists(_))));
    }

    #[test]
    fn test_delete_collection() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();
        db.delete_collection("docs").unwrap();
        assert!(db.list_collections().is_empty());
    }

    #[test]
    fn test_delete_nonexistent_collection() {
        let db = make_db();
        let result = db.delete_collection("nope");
        assert!(matches!(result, Err(VexError::CollectionNotFound(_))));
    }

    #[test]
    fn test_upsert_and_query() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new())?;
            c.upsert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new())?;
            c.upsert("c".into(), vec![1.0, 0.1, 0.0], HashMap::new())?;
            Ok(())
        })
        .unwrap();

        let results = db
            .with_collection("docs", |c| c.query(&[1.0, 0.0, 0.0], 2))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_batch_upsert() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert_batch(
                vec!["a".into(), "b".into(), "c".into()],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                vec![HashMap::new(), HashMap::new(), HashMap::new()],
            )
        })
        .unwrap();

        let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_delete_vector() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new())?;
            c.upsert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new())?;
            Ok(())
        })
        .unwrap();

        let deleted = db
            .with_collection_mut("docs", |c| c.delete("a"))
            .unwrap();
        assert!(deleted);

        let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_query_nonexistent_collection() {
        let db = make_db();
        let result = db.with_collection("nope", |c| c.query(&[1.0], 1));
        assert!(matches!(result, Err(VexError::CollectionNotFound(_))));
    }

    #[test]
    fn test_metadata_preserved() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        let meta = HashMap::from([("color".into(), serde_json::json!("red"))]);
        db.with_collection_mut("docs", |c| {
            c.upsert("a".into(), vec![1.0, 0.0, 0.0], meta)
        })
        .unwrap();

        let results = db
            .with_collection("docs", |c| c.query(&[1.0, 0.0, 0.0], 1))
            .unwrap();
        assert_eq!(results[0].metadata["color"], "red");
    }

    #[test]
    fn test_upsert_overwrites() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new())?;
            c.upsert("a".into(), vec![0.0, 1.0, 0.0], HashMap::new())?;
            Ok(())
        })
        .unwrap();

        let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 1);

        // Query toward [0,1,0] -- should find "a" now pointing there
        let results = db
            .with_collection("docs", |c| c.query(&[0.0, 1.0, 0.0], 1))
            .unwrap();
        assert_eq!(results[0].id, "a");
        assert!(results[0].score < 0.01);
    }

    #[test]
    fn test_filtered_query() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert(
                "a".into(),
                vec![1.0, 0.0, 0.0],
                HashMap::from([("color".into(), serde_json::json!("red"))]),
            )?;
            c.upsert(
                "b".into(),
                vec![0.9, 0.1, 0.0],
                HashMap::from([("color".into(), serde_json::json!("blue"))]),
            )?;
            c.upsert(
                "c".into(),
                vec![0.8, 0.2, 0.0],
                HashMap::from([("color".into(), serde_json::json!("red"))]),
            )?;
            Ok(())
        })
        .unwrap();

        let filter =
            crate::filter::Filter::parse(&serde_json::json!({"color": {"$eq": "red"}})).unwrap();

        let results = db
            .with_collection("docs", |c| c.query_with_filter(&[1.0, 0.0, 0.0], 10, &filter))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.metadata["color"] == "red"));
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "c");
    }

    #[test]
    fn test_filtered_query_with_numeric_range() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert(
                "cheap".into(),
                vec![1.0, 0.0, 0.0],
                HashMap::from([("price".into(), serde_json::json!(10))]),
            )?;
            c.upsert(
                "mid".into(),
                vec![0.9, 0.1, 0.0],
                HashMap::from([("price".into(), serde_json::json!(50))]),
            )?;
            c.upsert(
                "expensive".into(),
                vec![0.8, 0.2, 0.0],
                HashMap::from([("price".into(), serde_json::json!(200))]),
            )?;
            Ok(())
        })
        .unwrap();

        let filter =
            crate::filter::Filter::parse(&serde_json::json!({"price": {"$lte": 50}})).unwrap();

        let results = db
            .with_collection("docs", |c| c.query_with_filter(&[1.0, 0.0, 0.0], 10, &filter))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| {
            r.metadata["price"].as_i64().unwrap() <= 50
        }));
    }

    #[test]
    fn test_euclidean_collection() {
        let db = make_db();
        let config = CollectionConfig::new("euc", 2).with_metric(DistanceMetricKind::Euclidean);
        db.create_collection(config).unwrap();

        db.with_collection_mut("euc", |c| {
            c.upsert("origin".into(), vec![0.0, 0.0], HashMap::new())?;
            c.upsert("near".into(), vec![1.0, 0.0], HashMap::new())?;
            c.upsert("far".into(), vec![10.0, 10.0], HashMap::new())?;
            Ok(())
        })
        .unwrap();

        let results = db
            .with_collection("euc", |c| c.query(&[0.0, 0.0], 3))
            .unwrap();
        assert_eq!(results[0].id, "origin");
        assert_eq!(results[1].id, "near");
        assert_eq!(results[2].id, "far");
    }

    #[test]
    fn test_upsert_with_documents() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert_batch_with_docs(
                vec!["a".into(), "b".into(), "c".into()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.5, 0.5, 0.0]],
                vec![HashMap::new(), HashMap::new(), HashMap::new()],
                vec![
                    "machine learning for image recognition".into(),
                    "cooking recipes for pasta and pizza".into(),
                    "deep learning neural networks".into(),
                ],
            )
        })
        .unwrap();

        let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_keyword_search() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert_batch_with_docs(
                vec!["ml".into(), "cook".into(), "dl".into()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.5, 0.5, 0.0]],
                vec![HashMap::new(), HashMap::new(), HashMap::new()],
                vec![
                    "machine learning for image recognition".into(),
                    "cooking recipes for pasta and pizza".into(),
                    "deep learning neural networks".into(),
                ],
            )
        })
        .unwrap();

        let results = db
            .with_collection("docs", |c| c.keyword_search("machine learning", 10))
            .unwrap();

        assert!(!results.is_empty());
        // "ml" should rank highest for "machine learning"
        assert_eq!(results[0].id, "ml");
    }

    #[test]
    fn test_hybrid_query() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert_batch_with_docs(
                vec!["vec_close".into(), "text_match".into(), "both".into()],
                vec![
                    vec![1.0, 0.0, 0.0],  // closest to query vector
                    vec![0.0, 1.0, 0.0],  // far from query vector
                    vec![0.8, 0.2, 0.0],  // somewhat close to query vector
                ],
                vec![HashMap::new(), HashMap::new(), HashMap::new()],
                vec![
                    "unrelated content about cooking".into(),
                    "machine learning and artificial intelligence".into(),
                    "machine learning for image processing".into(),
                ],
            )
        })
        .unwrap();

        // Pure vector search: vec_close should win
        let vec_results = db
            .with_collection("docs", |c| c.query(&[1.0, 0.0, 0.0], 3))
            .unwrap();
        assert_eq!(vec_results[0].id, "vec_close");

        // Pure keyword search: text_match should win
        let kw_results = db
            .with_collection("docs", |c| c.keyword_search("machine learning", 3))
            .unwrap();
        assert!(kw_results.iter().any(|r| r.id == "text_match"));

        // Hybrid (alpha=0.5): "both" has decent vector AND keyword match
        let hybrid = db
            .with_collection("docs", |c| {
                c.hybrid_query(&[1.0, 0.0, 0.0], "machine learning", 3, 0.5)
            })
            .unwrap();
        assert_eq!(hybrid.len(), 3);
        // "both" should rank well because it appears in both result sets
        let both_rank = hybrid.iter().position(|r| r.id == "both").unwrap();
        assert!(both_rank <= 1, "expected 'both' in top 2, got rank {}", both_rank);
    }

    #[test]
    fn test_hybrid_alpha_extremes() {
        let db = make_db();
        db.create_collection(sample_config("docs")).unwrap();

        db.with_collection_mut("docs", |c| {
            c.upsert_batch_with_docs(
                vec!["vec_only".into(), "text_only".into()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]],
                vec![HashMap::new(), HashMap::new()],
                vec!["unrelated document".into(), "quantum computing research".into()],
            )
        })
        .unwrap();

        // alpha=1.0 (pure vector): vec_only should win
        let results = db
            .with_collection("docs", |c| {
                c.hybrid_query(&[1.0, 0.0, 0.0], "quantum computing", 2, 1.0)
            })
            .unwrap();
        assert_eq!(results[0].id, "vec_only");

        // alpha=0.0 (pure keyword): text_only should win
        let results = db
            .with_collection("docs", |c| {
                c.hybrid_query(&[1.0, 0.0, 0.0], "quantum computing", 2, 0.0)
            })
            .unwrap();
        assert_eq!(results[0].id, "text_only");
    }

    // ---- Persistence tests ----

    #[test]
    fn test_persistent_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        // Create, insert, drop
        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();

            db.with_collection_mut("docs", |c| {
                c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::from([("color".into(), serde_json::json!("red"))]))?;
                c.upsert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new())?;
                Ok(())
            })
            .unwrap();

            let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
            assert_eq!(count, 2);
        }

        // Reopen and verify data survived
        {
            let db = Database::open(&db_path).unwrap();
            let names = db.list_collections();
            assert_eq!(names, vec!["docs"]);

            let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
            assert_eq!(count, 2);

            let results = db
                .with_collection("docs", |c| c.query(&[1.0, 0.0, 0.0], 2))
                .unwrap();
            assert_eq!(results[0].id, "a");
            assert_eq!(results[0].metadata["color"], "red");
        }
    }

    #[test]
    fn test_persistent_delete_survives_restart() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();
            db.with_collection_mut("docs", |c| {
                c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new())?;
                c.upsert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new())?;
                Ok(())
            })
            .unwrap();
            db.with_collection_mut("docs", |c| { c.delete("a")?; Ok(()) }).unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
            assert_eq!(count, 1);
            let has_a = db.with_collection("docs", |c| Ok(c.contains("a"))).unwrap();
            assert!(!has_a);
            let has_b = db.with_collection("docs", |c| Ok(c.contains("b"))).unwrap();
            assert!(has_b);
        }
    }

    #[test]
    fn test_persistent_documents_and_bm25() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();
            db.with_collection_mut("docs", |c| {
                c.upsert_batch_with_docs(
                    vec!["ml".into(), "cook".into()],
                    vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                    vec![HashMap::new(), HashMap::new()],
                    vec![
                        "machine learning for image recognition".into(),
                        "cooking recipes for pasta".into(),
                    ],
                )
            })
            .unwrap();
        }

        // Reopen: BM25 index should be rebuilt from stored documents
        {
            let db = Database::open(&db_path).unwrap();
            let results = db
                .with_collection("docs", |c| c.keyword_search("machine learning", 10))
                .unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].id, "ml");
        }
    }

    #[test]
    fn test_persistent_multiple_collections() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();
            db.create_collection(
                CollectionConfig::new("images", 3).with_metric(DistanceMetricKind::Euclidean),
            )
            .unwrap();

            db.with_collection_mut("docs", |c| {
                c.upsert("d1".into(), vec![1.0, 0.0, 0.0], HashMap::new())
            })
            .unwrap();
            db.with_collection_mut("images", |c| {
                c.upsert("i1".into(), vec![0.0, 1.0, 0.0], HashMap::new())
            })
            .unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let mut names = db.list_collections();
            names.sort();
            assert_eq!(names, vec!["docs", "images"]);

            let d = db.with_collection("docs", |c| Ok(c.count())).unwrap();
            assert_eq!(d, 1);
            let i = db.with_collection("images", |c| Ok(c.count())).unwrap();
            assert_eq!(i, 1);
        }
    }

    #[test]
    fn test_persistent_delete_collection() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();
            db.create_collection(sample_config("temp")).unwrap();
            db.with_collection_mut("temp", |c| {
                c.upsert("t1".into(), vec![1.0, 0.0, 0.0], HashMap::new())
            })
            .unwrap();
            db.delete_collection("temp").unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let names = db.list_collections();
            assert_eq!(names, vec!["docs"]);
        }
    }

    #[test]
    fn test_persistent_upsert_overwrites() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("testdb");

        {
            let db = Database::open(&db_path).unwrap();
            db.create_collection(sample_config("docs")).unwrap();
            db.with_collection_mut("docs", |c| {
                c.upsert("a".into(), vec![1.0, 0.0, 0.0], HashMap::from([("v".into(), serde_json::json!(1))]))?;
                c.upsert("a".into(), vec![0.0, 1.0, 0.0], HashMap::from([("v".into(), serde_json::json!(2))]))?;
                Ok(())
            })
            .unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let count = db.with_collection("docs", |c| Ok(c.count())).unwrap();
            assert_eq!(count, 1);

            let results = db
                .with_collection("docs", |c| c.query(&[0.0, 1.0, 0.0], 1))
                .unwrap();
            assert_eq!(results[0].id, "a");
            assert_eq!(results[0].metadata["v"], 2);
        }
    }
}
