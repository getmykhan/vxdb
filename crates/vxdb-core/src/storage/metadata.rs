use std::path::Path;

use rusqlite::{params, Connection};

use crate::error::{VexError, VexResult};
use crate::types::Metadata;

pub struct MetadataStore {
    conn: Connection,
}

impl MetadataStore {
    pub fn create(path: &Path) -> VexResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| VexError::Internal(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;",
        )
        .map_err(|e| VexError::Internal(e.to_string()))?;

        Ok(Self { conn })
    }

    pub fn open(path: &Path) -> VexResult<Self> {
        Self::create(path)
    }

    pub fn put(&self, id: &str, metadata: &Metadata) -> VexResult<()> {
        let json = serde_json::to_string(metadata)?;
        self.conn
            .execute(
                "INSERT OR REPLACE INTO metadata (id, data) VALUES (?1, ?2)",
                params![id, json],
            )
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn put_batch(&self, items: &[(String, Metadata)]) -> VexResult<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| VexError::Internal(e.to_string()))?;

        for (id, meta) in items {
            let json = serde_json::to_string(meta)?;
            tx.execute(
                "INSERT OR REPLACE INTO metadata (id, data) VALUES (?1, ?2)",
                params![id, json],
            )
            .map_err(|e| VexError::Internal(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> VexResult<Option<Metadata>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM metadata WHERE id = ?1")
            .map_err(|e| VexError::Internal(e.to_string()))?;

        let result = stmt
            .query_row(params![id], |row| {
                let data: String = row.get(0)?;
                Ok(data)
            })
            .ok();

        match result {
            Some(json) => {
                let meta: Metadata = serde_json::from_str(&json)?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    pub fn delete(&self, id: &str) -> VexResult<bool> {
        let affected = self
            .conn
            .execute("DELETE FROM metadata WHERE id = ?1", params![id])
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(affected > 0)
    }

    pub fn get_all_ids(&self) -> VexResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM metadata")
            .map_err(|e| VexError::Internal(e.to_string()))?;

        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| VexError::Internal(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }

    pub fn count(&self) -> VexResult<usize> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM metadata", [], |row| row.get(0))
            .map_err(|e| VexError::Internal(e.to_string()))?;
        Ok(count)
    }

    /// Query metadata with a raw SQL WHERE clause on JSON fields.
    /// Used internally by the filter engine.
    pub fn query_ids_where(&self, where_clause: &str, params_vec: &[&dyn rusqlite::types::ToSql]) -> VexResult<Vec<String>> {
        let sql = format!("SELECT id FROM metadata WHERE {}", where_clause);
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| VexError::Internal(e.to_string()))?;

        let ids: Vec<String> = stmt
            .query_map(rusqlite::params_from_iter(params_vec), |row| row.get(0))
            .map_err(|e| VexError::Internal(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_put_get() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");

        let store = MetadataStore::create(&path).unwrap();
        let meta = HashMap::from([
            ("color".into(), serde_json::json!("red")),
            ("size".into(), serde_json::json!(42)),
        ]);
        store.put("doc1", &meta).unwrap();

        let retrieved = store.get("doc1").unwrap().unwrap();
        assert_eq!(retrieved["color"], "red");
        assert_eq!(retrieved["size"], 42);
    }

    #[test]
    fn test_get_nonexistent() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");
        let store = MetadataStore::create(&path).unwrap();
        assert!(store.get("nope").unwrap().is_none());
    }

    #[test]
    fn test_upsert() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");
        let store = MetadataStore::create(&path).unwrap();

        let meta1 = HashMap::from([("v".into(), serde_json::json!(1))]);
        store.put("a", &meta1).unwrap();

        let meta2 = HashMap::from([("v".into(), serde_json::json!(2))]);
        store.put("a", &meta2).unwrap();

        assert_eq!(store.count().unwrap(), 1);
        let retrieved = store.get("a").unwrap().unwrap();
        assert_eq!(retrieved["v"], 2);
    }

    #[test]
    fn test_delete() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");
        let store = MetadataStore::create(&path).unwrap();

        store
            .put("a", &HashMap::from([("x".into(), serde_json::json!(1))]))
            .unwrap();
        assert!(store.delete("a").unwrap());
        assert!(!store.delete("a").unwrap());
        assert!(store.get("a").unwrap().is_none());
    }

    #[test]
    fn test_batch_put() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");
        let store = MetadataStore::create(&path).unwrap();

        let items: Vec<(String, Metadata)> = (0..100)
            .map(|i| {
                (
                    format!("doc{}", i),
                    HashMap::from([("i".into(), serde_json::json!(i))]),
                )
            })
            .collect();

        store.put_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 100);
    }

    #[test]
    fn test_persistence_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");

        {
            let store = MetadataStore::create(&path).unwrap();
            store
                .put("a", &HashMap::from([("val".into(), serde_json::json!("hello"))]))
                .unwrap();
            store
                .put("b", &HashMap::from([("val".into(), serde_json::json!("world"))]))
                .unwrap();
        }

        {
            let store = MetadataStore::open(&path).unwrap();
            assert_eq!(store.count().unwrap(), 2);
            let a = store.get("a").unwrap().unwrap();
            assert_eq!(a["val"], "hello");
            let b = store.get("b").unwrap().unwrap();
            assert_eq!(b["val"], "world");
        }
    }

    #[test]
    fn test_get_all_ids() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.db");
        let store = MetadataStore::create(&path).unwrap();

        store.put("a", &HashMap::new()).unwrap();
        store.put("b", &HashMap::new()).unwrap();
        store.put("c", &HashMap::new()).unwrap();

        let mut ids = store.get_all_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }
}
