use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::VexResult;
use crate::types::{Metadata, VectorData};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    Upsert {
        collection: String,
        id: String,
        vector: VectorData,
        metadata: Metadata,
        #[serde(default)]
        document: Option<String>,
    },
    Delete {
        collection: String,
        id: String,
    },
    CreateCollection {
        name: String,
        dimension: usize,
        metric: String,
        index_kind: String,
    },
    DeleteCollection {
        name: String,
    },
    Checkpoint,
}

pub struct WriteAheadLog {
    path: PathBuf,
    file: File,
    entry_count: usize,
}

impl WriteAheadLog {
    pub fn create(path: &Path) -> VexResult<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
            entry_count: 0,
        })
    }

    pub fn open(path: &Path) -> VexResult<Self> {
        let entry_count = if path.exists() {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            reader.lines().count()
        } else {
            0
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
            entry_count,
        })
    }

    pub fn append(&mut self, entry: &WalEntry) -> VexResult<()> {
        let json = serde_json::to_string(entry)?;
        writeln!(self.file, "{}", json)?;
        self.file.flush()?;
        self.entry_count += 1;
        Ok(())
    }

    pub fn replay(&self) -> VexResult<Vec<WalEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WalEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    // Truncated entry -- stop replay here (crash recovery)
                    eprintln!("WAL: skipping corrupted entry: {}", e);
                    break;
                }
            }
        }

        // Only return entries after the last checkpoint
        let last_checkpoint = entries
            .iter()
            .rposition(|e| matches!(e, WalEntry::Checkpoint));

        match last_checkpoint {
            Some(pos) => Ok(entries[pos + 1..].to_vec()),
            None => Ok(entries),
        }
    }

    pub fn checkpoint(&mut self) -> VexResult<()> {
        self.append(&WalEntry::Checkpoint)?;
        // Truncate the WAL file, keeping only the checkpoint marker
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.entry_count = 0;
        Ok(())
    }

    pub fn entry_count(&self) -> usize {
        self.entry_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_append() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        let mut wal = WriteAheadLog::create(&path).unwrap();
        assert_eq!(wal.entry_count(), 0);

        wal.append(&WalEntry::Upsert {
            collection: "docs".into(),
            id: "a".into(),
            vector: vec![1.0, 2.0],
            metadata: HashMap::new(),
            document: None,
        })
        .unwrap();

        assert_eq!(wal.entry_count(), 1);
    }

    #[test]
    fn test_replay() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        {
            let mut wal = WriteAheadLog::create(&path).unwrap();
            wal.append(&WalEntry::Upsert {
                collection: "docs".into(),
                id: "a".into(),
                vector: vec![1.0, 2.0],
                metadata: HashMap::from([("k".into(), serde_json::json!("v"))]),
                document: None,
            })
            .unwrap();
            wal.append(&WalEntry::Delete {
                collection: "docs".into(),
                id: "b".into(),
            })
            .unwrap();
        }

        {
            let wal = WriteAheadLog::open(&path).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 2);
            assert!(matches!(&entries[0], WalEntry::Upsert { id, .. } if id == "a"));
            assert!(matches!(&entries[1], WalEntry::Delete { id, .. } if id == "b"));
        }
    }

    #[test]
    fn test_checkpoint_clears_wal() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        let mut wal = WriteAheadLog::create(&path).unwrap();
        wal.append(&WalEntry::Upsert {
            collection: "docs".into(),
            id: "a".into(),
            vector: vec![1.0],
            metadata: HashMap::new(),
            document: None,
        })
        .unwrap();
        wal.append(&WalEntry::Upsert {
            collection: "docs".into(),
            id: "b".into(),
            vector: vec![2.0],
            metadata: HashMap::new(),
            document: None,
        })
        .unwrap();

        wal.checkpoint().unwrap();

        // After checkpoint, replay should return empty
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_replay_after_checkpoint_only_returns_new() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        {
            let mut wal = WriteAheadLog::create(&path).unwrap();
            wal.append(&WalEntry::Upsert {
                collection: "docs".into(),
                id: "old".into(),
                vector: vec![1.0],
                metadata: HashMap::new(),
                document: None,
            })
            .unwrap();
            wal.append(&WalEntry::Checkpoint).unwrap();
            wal.append(&WalEntry::Upsert {
                collection: "docs".into(),
                id: "new".into(),
                vector: vec![2.0],
                metadata: HashMap::new(),
                document: None,
            })
            .unwrap();
        }

        {
            let wal = WriteAheadLog::open(&path).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 1);
            assert!(matches!(&entries[0], WalEntry::Upsert { id, .. } if id == "new"));
        }
    }

    #[test]
    fn test_corrupted_entry_recovery() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        // Write valid entries then corrupt the file
        {
            let mut wal = WriteAheadLog::create(&path).unwrap();
            wal.append(&WalEntry::Upsert {
                collection: "docs".into(),
                id: "good".into(),
                vector: vec![1.0],
                metadata: HashMap::new(),
                document: None,
            })
            .unwrap();
        }

        // Append corrupted data
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(f, "{{invalid json truncated").unwrap();
        }

        {
            let wal = WriteAheadLog::open(&path).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 1);
            assert!(matches!(&entries[0], WalEntry::Upsert { id, .. } if id == "good"));
        }
    }

    #[test]
    fn test_collection_operations() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.wal");

        let mut wal = WriteAheadLog::create(&path).unwrap();
        wal.append(&WalEntry::CreateCollection {
            name: "test".into(),
            dimension: 128,
            metric: "cosine".into(),
            index_kind: "flat".into(),
        })
        .unwrap();
        wal.append(&WalEntry::DeleteCollection {
            name: "test".into(),
        })
        .unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[0], WalEntry::CreateCollection { name, .. } if name == "test"));
        assert!(matches!(&entries[1], WalEntry::DeleteCollection { name } if name == "test"));
    }
}
