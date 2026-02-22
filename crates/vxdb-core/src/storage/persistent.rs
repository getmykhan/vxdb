use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection};

use crate::error::{VexError, VexResult};
use crate::storage::mmap::MmapVectorStorage;
use crate::types::{Metadata, VectorData};

/// Per-collection persistent storage combining mmap vectors + SQLite metadata.
/// The SQLite Connection is wrapped in a Mutex for thread safety (Send + Sync).
pub struct PersistentStorage {
    vectors: MmapVectorStorage,
    conn: Mutex<Connection>,
    base_path: PathBuf,
}

impl PersistentStorage {
    pub fn create(path: &Path, dimension: usize) -> VexResult<Self> {
        fs::create_dir_all(path)?;

        let vectors = MmapVectorStorage::create(&path.join("vectors.vex"), dimension)?;

        let conn = Connection::open(path.join("store.db"))
            .map_err(|e| VexError::Internal(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                vector_idx INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                document TEXT
            );
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;",
        )
        .map_err(|e| VexError::Internal(e.to_string()))?;

        Ok(Self {
            vectors,
            conn: Mutex::new(conn),
            base_path: path.to_path_buf(),
        })
    }

    pub fn open(path: &Path) -> VexResult<Self> {
        let vectors = MmapVectorStorage::open(&path.join("vectors.vex"))?;

        let conn = Connection::open(path.join("store.db"))
            .map_err(|e| VexError::Internal(e.to_string()))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;",
        )
        .map_err(|e| VexError::Internal(e.to_string()))?;

        Ok(Self {
            vectors,
            conn: Mutex::new(conn),
            base_path: path.to_path_buf(),
        })
    }

    pub fn save_entry(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: &Metadata,
        document: Option<&str>,
    ) -> VexResult<()> {
        let vector_idx = self.vectors.append(vector)?;
        let meta_json = serde_json::to_string(metadata)?;

        let conn = self.conn.lock().map_err(|e| VexError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO entries (id, vector_idx, metadata, document) VALUES (?1, ?2, ?3, ?4)",
            params![id, vector_idx as i64, meta_json, document],
        )
        .map_err(|e| VexError::Internal(e.to_string()))?;

        Ok(())
    }

    pub fn save_batch(
        &mut self,
        entries: &[(String, VectorData, Metadata, Option<String>)],
    ) -> VexResult<()> {
        let vectors: Vec<Vec<f32>> = entries.iter().map(|(_, v, _, _)| v.clone()).collect();
        let indices = self.vectors.append_batch(&vectors)?;

        let conn = self.conn.lock().map_err(|e| VexError::Internal(e.to_string()))?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| VexError::Internal(e.to_string()))?;

        for (i, (id, _, metadata, document)) in entries.iter().enumerate() {
            let meta_json = serde_json::to_string(metadata)?;
            tx.execute(
                "INSERT OR REPLACE INTO entries (id, vector_idx, metadata, document) VALUES (?1, ?2, ?3, ?4)",
                params![id, indices[i] as i64, meta_json, document.as_deref()],
            )
            .map_err(|e| VexError::Internal(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn delete_entry(&self, id: &str) -> VexResult<bool> {
        let conn = self.conn.lock().map_err(|e| VexError::Internal(e.to_string()))?;
        let affected = conn
            .execute("DELETE FROM entries WHERE id = ?1", params![id])
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(affected > 0)
    }

    /// Load all stored entries. Returns (id, vector, metadata, document).
    pub fn load_all(&self) -> VexResult<Vec<(String, VectorData, Metadata, Option<String>)>> {
        let conn = self.conn.lock().map_err(|e| VexError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, vector_idx, metadata, document FROM entries")
            .map_err(|e| VexError::Internal(e.to_string()))?;

        let rows: Vec<_> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let vector_idx: i64 = row.get(1)?;
                let meta_json: String = row.get(2)?;
                let document: Option<String> = row.get(3)?;
                Ok((id, vector_idx, meta_json, document))
            })
            .map_err(|e| VexError::Internal(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| VexError::Internal(e.to_string()))?;

        let mut entries = Vec::new();
        for (id, vector_idx, meta_json, document) in rows {
            let vector = self.vectors.get(vector_idx as usize)?;
            let metadata: Metadata = serde_json::from_str(&meta_json)?;
            entries.push((id, vector, metadata, document));
        }

        Ok(entries)
    }

    pub fn path(&self) -> &Path {
        &self.base_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_create_save_and_load() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("coll");

        {
            let mut store = PersistentStorage::create(&path, 3).unwrap();
            let meta = HashMap::from([("color".into(), serde_json::json!("red"))]);
            store
                .save_entry("a", &[1.0, 2.0, 3.0], &meta, Some("hello world"))
                .unwrap();
            store
                .save_entry("b", &[4.0, 5.0, 6.0], &HashMap::new(), None)
                .unwrap();
        }

        {
            let store = PersistentStorage::open(&path).unwrap();
            let entries = store.load_all().unwrap();
            assert_eq!(entries.len(), 2);

            let a = entries.iter().find(|(id, _, _, _)| id == "a").unwrap();
            assert_eq!(a.1, vec![1.0, 2.0, 3.0]);
            assert_eq!(a.2["color"], "red");
            assert_eq!(a.3.as_deref(), Some("hello world"));

            let b = entries.iter().find(|(id, _, _, _)| id == "b").unwrap();
            assert_eq!(b.1, vec![4.0, 5.0, 6.0]);
            assert!(b.3.is_none());
        }
    }

    #[test]
    fn test_upsert_overwrites_sqlite_row() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("coll");

        let mut store = PersistentStorage::create(&path, 2).unwrap();
        store
            .save_entry("a", &[1.0, 2.0], &HashMap::new(), None)
            .unwrap();
        store
            .save_entry("a", &[3.0, 4.0], &HashMap::new(), None)
            .unwrap();

        let entries = store.load_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, vec![3.0, 4.0]);
    }

    #[test]
    fn test_delete_entry() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("coll");

        let mut store = PersistentStorage::create(&path, 2).unwrap();
        store
            .save_entry("a", &[1.0, 2.0], &HashMap::new(), None)
            .unwrap();
        store
            .save_entry("b", &[3.0, 4.0], &HashMap::new(), None)
            .unwrap();

        assert!(store.delete_entry("a").unwrap());
        assert!(!store.delete_entry("a").unwrap());

        let entries = store.load_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "b");
    }

    #[test]
    fn test_batch_save() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("coll");

        let mut store = PersistentStorage::create(&path, 2).unwrap();
        let batch = vec![
            ("a".into(), vec![1.0, 2.0], HashMap::new(), None),
            (
                "b".into(),
                vec![3.0, 4.0],
                HashMap::from([("k".into(), serde_json::json!("v"))]),
                Some("doc text".into()),
            ),
        ];
        store.save_batch(&batch).unwrap();

        let entries = store.load_all().unwrap();
        assert_eq!(entries.len(), 2);
    }
}
