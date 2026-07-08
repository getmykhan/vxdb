#![allow(clippy::useless_conversion)]

use std::collections::HashMap;
use std::path::Path;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use vxdb_core::collection::Database as CoreDatabase;
use vxdb_core::filter::Filter;
use vxdb_core::types::{CollectionConfig, DistanceMetricKind, IndexKind};

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let arr: PyResult<Vec<serde_json::Value>> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(arr?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Err(PyValueError::new_err("unsupported type for JSON conversion"))
    }
}

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let items: PyResult<Vec<PyObject>> = arr.iter().map(|v| json_to_py(py, v)).collect();
            Ok(PyList::new_bound(py, items?).to_object(py))
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.to_object(py))
        }
    }
}

fn parse_metric(s: &str) -> PyResult<DistanceMetricKind> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetricKind::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetricKind::Euclidean),
        "dot" | "dot_product" | "dotproduct" | "ip" => Ok(DistanceMetricKind::DotProduct),
        _ => Err(PyValueError::new_err(format!(
            "unknown metric: '{}'. Use 'cosine', 'euclidean', or 'dot'",
            s
        ))),
    }
}

fn parse_index_kind(s: &str) -> PyResult<IndexKind> {
    match s.to_lowercase().as_str() {
        "flat" => Ok(IndexKind::Flat),
        "hnsw" => Ok(IndexKind::Hnsw),
        _ => Err(PyValueError::new_err(format!(
            "unknown index: '{}'. Use 'flat' or 'hnsw'",
            s
        ))),
    }
}

fn vex_err(e: vxdb_core::VexError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn results_to_py(py: Python<'_>, results: Vec<vxdb_core::types::SearchResult>) -> PyResult<Vec<PyObject>> {
    results
        .into_iter()
        .map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            dict.set_item("document", r.document)?;
            Ok(dict.to_object(py))
        })
        .collect()
}

/// Convert the `vectors` argument into `Vec<Vec<f32>>`.
///
/// Fast path: if the object exposes the Python buffer protocol as a 2-D,
/// C-contiguous f32 buffer (numpy/torch/jax arrays, `array.array`), read its raw
/// memory directly — no per-element Python float boxing, no giant intermediate
/// list. This lets vxdb ingest numpy data efficiently while staying
/// dependency-free: numpy is never imported or required, we only read the
/// standard buffer interface it (and others) implement. Otherwise fall back to
/// extracting a `List[List[float]]`.
fn extract_vectors(vectors: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(buf) = PyBuffer::<f32>::get_bound(vectors) {
        // `dim > 0` guards `chunks(0)`, which panics. A zero-width buffer falls
        // through to the list path and yields a clean dimension/empty error.
        if buf.dimensions() == 2 && buf.is_c_contiguous() && buf.shape()[1] > 0 {
            let dim = buf.shape()[1];
            let flat = buf.to_vec(vectors.py())?;
            return Ok(flat.chunks(dim).map(<[f32]>::to_vec).collect());
        }
    }
    vectors.extract::<Vec<Vec<f32>>>()
}

#[pyclass]
struct Collection {
    name: String,
    db: std::sync::Arc<CoreDatabase>,
}

#[pymethods]
impl Collection {
    #[pyo3(signature = (ids, vectors, metadata=None, documents=None))]
    fn upsert(
        &self,
        ids: Vec<String>,
        vectors: &Bound<'_, PyAny>,
        metadata: Option<Bound<'_, PyList>>,
        documents: Option<Vec<String>>,
    ) -> PyResult<()> {
        let vectors = extract_vectors(vectors)?;
        let metas: Vec<HashMap<String, serde_json::Value>> = match metadata {
            Some(list) => {
                let mut result = Vec::new();
                for item in list.iter() {
                    let dict = item.downcast::<PyDict>()?;
                    let mut map = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        map.insert(key, py_to_json(&v)?);
                    }
                    result.push(map);
                }
                result
            }
            None => vec![HashMap::new(); ids.len()],
        };

        if let Some(docs) = documents {
            self.db
                .with_collection_mut(&self.name, |c| c.upsert_batch_with_docs(ids, vectors, metas, docs))
                .map_err(vex_err)?;
        } else {
            self.db
                .with_collection_mut(&self.name, |c| c.upsert_batch(ids, vectors, metas))
                .map_err(vex_err)?;
        }

        Ok(())
    }

    #[pyo3(signature = (vector, top_k = 10, filter = None, ef_search = None))]
    fn query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        top_k: usize,
        filter: Option<Bound<'_, PyDict>>,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        // Parse the filter while we still hold the GIL (it reads a Python dict).
        let parsed = match filter {
            Some(filter_dict) => {
                let json_val = py_to_json(&filter_dict.into_any())?;
                Some(Filter::parse(&json_val).map_err(vex_err)?)
            }
            None => None,
        };

        // Release the GIL for the search itself: it runs pure Rust over a shared
        // read lock, so independent queries on other threads execute in parallel.
        let results = py
            .allow_threads(|| match &parsed {
                Some(f) => self.db.with_collection(&self.name, |c| {
                    c.query_with_filter_ef(&vector, top_k, f, ef_search)
                }),
                None => self
                    .db
                    .with_collection(&self.name, |c| c.query_ef(&vector, top_k, ef_search)),
            })
            .map_err(vex_err)?;

        results_to_py(py, results)
    }

    #[pyo3(signature = (vector, query, top_k = 10, alpha = 0.5))]
    fn hybrid_query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        query: &str,
        top_k: usize,
        alpha: f32,
    ) -> PyResult<Vec<PyObject>> {
        let query = query.to_string();
        let results = py
            .allow_threads(|| {
                self.db
                    .with_collection(&self.name, |c| c.hybrid_query(&vector, &query, top_k, alpha))
            })
            .map_err(vex_err)?;

        results_to_py(py, results)
    }

    #[pyo3(signature = (query, top_k = 10))]
    fn keyword_search(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
    ) -> PyResult<Vec<PyObject>> {
        let query = query.to_string();
        let results = py
            .allow_threads(|| {
                self.db
                    .with_collection(&self.name, |c| c.keyword_search(&query, top_k))
            })
            .map_err(vex_err)?;

        results_to_py(py, results)
    }

    fn delete(&self, ids: Vec<String>) -> PyResult<Vec<bool>> {
        let mut deleted = Vec::new();
        for id in &ids {
            let d = self
                .db
                .with_collection_mut(&self.name, |c| c.delete(id))
                .map_err(vex_err)?;
            deleted.push(d);
        }
        Ok(deleted)
    }

    fn count(&self) -> PyResult<usize> {
        self.db
            .with_collection(&self.name, |c| Ok(c.count()))
            .map_err(vex_err)
    }

    fn __repr__(&self) -> String {
        format!("Collection(name='{}')", self.name)
    }
}

#[pyclass]
struct Database {
    inner: std::sync::Arc<CoreDatabase>,
}

#[pymethods]
impl Database {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<&str>) -> PyResult<Self> {
        let db = match path {
            Some(p) => CoreDatabase::open(Path::new(p)).map_err(vex_err)?,
            None => CoreDatabase::new(),
        };
        Ok(Self {
            inner: std::sync::Arc::new(db),
        })
    }

    #[pyo3(signature = (name, dimension, metric = "cosine", index = "flat"))]
    fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        metric: &str,
        index: &str,
    ) -> PyResult<Collection> {
        let config = CollectionConfig::new(name, dimension)
            .with_metric(parse_metric(metric)?)
            .with_index(parse_index_kind(index)?);

        self.inner.create_collection(config).map_err(vex_err)?;

        Ok(Collection {
            name: name.to_string(),
            db: self.inner.clone(),
        })
    }

    fn get_collection(&self, name: &str) -> PyResult<Collection> {
        self.inner
            .with_collection(name, |_| Ok(()))
            .map_err(vex_err)?;

        Ok(Collection {
            name: name.to_string(),
            db: self.inner.clone(),
        })
    }

    fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
    }

    fn delete_collection(&self, name: &str) -> PyResult<()> {
        self.inner.delete_collection(name).map_err(vex_err)
    }

    fn __repr__(&self) -> String {
        let count = self.inner.list_collections().len();
        format!("Database(collections={})", count)
    }
}

#[pymodule]
fn _vxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_class::<Collection>()?;
    Ok(())
}
