use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};

use crate::error::{VexError, VexResult};

const HEADER_SIZE: usize = 16; // 4 bytes magic + 4 bytes dimension + 4 bytes count + 4 bytes reserved
const MAGIC: [u8; 4] = *b"VEXV";

/// Memory-mapped vector storage.
/// File layout: [HEADER (16 bytes)] [vector_0: dim*f32] [vector_1: dim*f32] ...
pub struct MmapVectorStorage {
    path: PathBuf,
    dimension: usize,
    count: usize,
    file: File,
}

impl MmapVectorStorage {
    pub fn create(path: &Path, dimension: usize) -> VexResult<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4..8].copy_from_slice(&(dimension as u32).to_le_bytes());
        header[8..12].copy_from_slice(&0u32.to_le_bytes());
        file.write_all(&header)?;
        file.flush()?;

        Ok(Self {
            path: path.to_path_buf(),
            dimension,
            count: 0,
            file,
        })
    }

    pub fn open(path: &Path) -> VexResult<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER_SIZE {
            return Err(VexError::Internal("file too small for header".into()));
        }
        if &mmap[0..4] != &MAGIC {
            return Err(VexError::Internal("invalid magic bytes".into()));
        }

        let dimension = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;

        let expected_size = HEADER_SIZE + count * dimension * 4;
        if mmap.len() < expected_size {
            return Err(VexError::Internal("file truncated".into()));
        }

        Ok(Self {
            path: path.to_path_buf(),
            dimension,
            count,
            file,
        })
    }

    fn vector_byte_size(&self) -> usize {
        self.dimension * std::mem::size_of::<f32>()
    }

    fn vector_offset(&self, index: usize) -> usize {
        HEADER_SIZE + index * self.vector_byte_size()
    }

    pub fn append(&mut self, vector: &[f32]) -> VexResult<usize> {
        if vector.len() != self.dimension {
            return Err(VexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let byte_data: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                vector.len() * std::mem::size_of::<f32>(),
            )
        };

        // Seek to end and write
        let offset = self.vector_offset(self.count);
        self.file
            .set_len(offset as u64 + byte_data.len() as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&self.file)? };
        mmap[offset..offset + byte_data.len()].copy_from_slice(byte_data);

        self.count += 1;

        // Update count in header
        mmap[8..12].copy_from_slice(&(self.count as u32).to_le_bytes());
        mmap.flush()?;

        Ok(self.count - 1)
    }

    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> VexResult<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let vbs = self.vector_byte_size();
        let total_new_bytes = vectors.len() * vbs;
        let start_offset = self.vector_offset(self.count);

        self.file
            .set_len((start_offset + total_new_bytes) as u64)?;
        let mut mmap = unsafe { MmapMut::map_mut(&self.file)? };

        let mut indices = Vec::with_capacity(vectors.len());
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dimension {
                return Err(VexError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
            let byte_data: &[u8] = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, vbs)
            };
            let off = start_offset + i * vbs;
            mmap[off..off + vbs].copy_from_slice(byte_data);
            indices.push(self.count + i);
        }

        self.count += vectors.len();
        mmap[8..12].copy_from_slice(&(self.count as u32).to_le_bytes());
        mmap.flush()?;

        Ok(indices)
    }

    pub fn get(&self, index: usize) -> VexResult<Vec<f32>> {
        if index >= self.count {
            return Err(VexError::Internal(format!(
                "vector index {} out of bounds (count={})",
                index, self.count
            )));
        }

        let mmap = unsafe { Mmap::map(&self.file)? };
        let offset = self.vector_offset(index);
        let byte_len = self.vector_byte_size();
        let bytes = &mmap[offset..offset + byte_len];

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(floats)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_append() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let mut store = MmapVectorStorage::create(&path, 3).unwrap();
        assert_eq!(store.count(), 0);

        let idx = store.append(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(store.count(), 1);

        let idx = store.append(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_get() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let mut store = MmapVectorStorage::create(&path, 3).unwrap();
        store.append(&[1.0, 2.0, 3.0]).unwrap();
        store.append(&[4.0, 5.0, 6.0]).unwrap();

        let v0 = store.get(0).unwrap();
        assert_eq!(v0, vec![1.0, 2.0, 3.0]);

        let v1 = store.get(1).unwrap();
        assert_eq!(v1, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let mut store = MmapVectorStorage::create(&path, 3).unwrap();
        let result = store.append(&[1.0, 2.0]);
        assert!(matches!(result, Err(VexError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_persistence_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        // Write
        {
            let mut store = MmapVectorStorage::create(&path, 4).unwrap();
            store.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
            store.append(&[5.0, 6.0, 7.0, 8.0]).unwrap();
            store.append(&[9.0, 10.0, 11.0, 12.0]).unwrap();
        }

        // Reopen and verify
        {
            let store = MmapVectorStorage::open(&path).unwrap();
            assert_eq!(store.count(), 3);
            assert_eq!(store.dimension(), 4);
            assert_eq!(store.get(0).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(store.get(1).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
            assert_eq!(store.get(2).unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
        }
    }

    #[test]
    fn test_batch_append() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let mut store = MmapVectorStorage::create(&path, 2).unwrap();
        let indices = store
            .append_batch(&[
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
            ])
            .unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(store.count(), 3);

        assert_eq!(store.get(0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(store.get(2).unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_out_of_bounds() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let mut store = MmapVectorStorage::create(&path, 2).unwrap();
        store.append(&[1.0, 2.0]).unwrap();

        let result = store.get(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_batch_persistence() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vectors.vex");

        let dim = 128;
        let n = 1000;

        {
            let mut store = MmapVectorStorage::create(&path, dim).unwrap();
            let vectors: Vec<Vec<f32>> = (0..n)
                .map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect())
                .collect();
            store.append_batch(&vectors).unwrap();
        }

        {
            let store = MmapVectorStorage::open(&path).unwrap();
            assert_eq!(store.count(), n);
            // Spot-check first and last
            let first = store.get(0).unwrap();
            assert_eq!(first[0], 0.0);
            let last = store.get(n - 1).unwrap();
            assert_eq!(last[0], ((n - 1) * dim) as f32);
        }
    }
}
