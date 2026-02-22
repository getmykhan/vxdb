use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use vxdb_core::collection::Database;
use vxdb_core::filter::Filter;
use vxdb_core::types::{CollectionConfig, DistanceMetricKind, IndexKind};

pub type AppState = Arc<Database>;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err_response(status: StatusCode, msg: impl Into<String>) -> impl IntoResponse {
    (status, Json(ErrorResponse { error: msg.into() }))
}

// --- Collection CRUD ---

#[derive(Deserialize)]
pub struct CreateCollectionRequest {
    name: String,
    dimension: usize,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_index")]
    index: String,
}

fn default_metric() -> String { "cosine".into() }
fn default_index() -> String { "flat".into() }

fn parse_metric(s: &str) -> Result<DistanceMetricKind, String> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetricKind::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetricKind::Euclidean),
        "dot" | "dot_product" | "dotproduct" | "ip" => Ok(DistanceMetricKind::DotProduct),
        _ => Err(format!("unknown metric: '{}'", s)),
    }
}

fn parse_index(s: &str) -> Result<IndexKind, String> {
    match s.to_lowercase().as_str() {
        "flat" => Ok(IndexKind::Flat),
        "hnsw" => Ok(IndexKind::Hnsw),
        _ => Err(format!("unknown index: '{}'", s)),
    }
}

pub async fn create_collection(
    State(db): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let metric = match parse_metric(&req.metric) {
        Ok(m) => m,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e).into_response(),
    };
    let index = match parse_index(&req.index) {
        Ok(i) => i,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e).into_response(),
    };

    let config = CollectionConfig::new(&req.name, req.dimension)
        .with_metric(metric)
        .with_index(index);

    match db.create_collection(config) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"name": req.name, "dimension": req.dimension})),
        )
            .into_response(),
        Err(e) => err_response(StatusCode::CONFLICT, e.to_string()).into_response(),
    }
}

pub async fn list_collections(State(db): State<AppState>) -> impl IntoResponse {
    let names = db.list_collections();
    Json(serde_json::json!({"collections": names}))
}

pub async fn delete_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match db.delete_collection(&name) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err_response(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    }
}

// --- Vector operations ---

#[derive(Deserialize)]
pub struct UpsertRequest {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    #[serde(default)]
    metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    #[serde(default)]
    documents: Option<Vec<String>>,
}

pub async fn upsert(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> impl IntoResponse {
    let metadata = req.metadata.unwrap_or_else(|| vec![HashMap::new(); req.ids.len()]);
    let count = req.ids.len();

    let result = if let Some(docs) = req.documents {
        db.with_collection_mut(&name, |c| {
            c.upsert_batch_with_docs(req.ids, req.vectors, metadata, docs)
        })
    } else {
        db.with_collection_mut(&name, |c| {
            c.upsert_batch(req.ids, req.vectors, metadata)
        })
    };

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({"upserted": count})),
        )
            .into_response(),
        Err(e) => err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
pub struct QueryRequest {
    vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

fn default_top_k() -> usize { 10 }

#[derive(Serialize)]
struct QueryResultItem {
    id: String,
    score: f32,
    metadata: HashMap<String, serde_json::Value>,
}

pub async fn query(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let results = if let Some(filter_val) = &req.filter {
        match Filter::parse(filter_val) {
            Ok(f) => db.with_collection(&name, |c| c.query_with_filter(&req.vector, req.top_k, &f)),
            Err(e) => return err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        db.with_collection(&name, |c| c.query(&req.vector, req.top_k))
    };

    match results {
        Ok(r) => {
            let items: Vec<QueryResultItem> = r
                .into_iter()
                .map(|sr| QueryResultItem {
                    id: sr.id,
                    score: sr.score,
                    metadata: sr.metadata,
                })
                .collect();
            Json(serde_json::json!({"results": items})).into_response()
        }
        Err(e) => err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
pub struct HybridQueryRequest {
    vector: Vec<f32>,
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_alpha")]
    alpha: f32,
}

fn default_alpha() -> f32 { 0.5 }

pub async fn hybrid_query(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<HybridQueryRequest>,
) -> impl IntoResponse {
    match db.with_collection(&name, |c| c.hybrid_query(&req.vector, &req.query, req.top_k, req.alpha)) {
        Ok(r) => {
            let items: Vec<QueryResultItem> = r
                .into_iter()
                .map(|sr| QueryResultItem {
                    id: sr.id,
                    score: sr.score,
                    metadata: sr.metadata,
                })
                .collect();
            Json(serde_json::json!({"results": items})).into_response()
        }
        Err(e) => err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
pub struct KeywordSearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

pub async fn keyword_search(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<KeywordSearchRequest>,
) -> impl IntoResponse {
    match db.with_collection(&name, |c| c.keyword_search(&req.query, req.top_k)) {
        Ok(r) => {
            let items: Vec<QueryResultItem> = r
                .into_iter()
                .map(|sr| QueryResultItem {
                    id: sr.id,
                    score: sr.score,
                    metadata: sr.metadata,
                })
                .collect();
            Json(serde_json::json!({"results": items})).into_response()
        }
        Err(e) => err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
pub struct DeleteVectorsRequest {
    ids: Vec<String>,
}

pub async fn delete_vectors(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteVectorsRequest>,
) -> impl IntoResponse {
    let mut deleted = Vec::new();
    for id in &req.ids {
        match db.with_collection_mut(&name, |c| c.delete(id)) {
            Ok(d) => deleted.push(d),
            Err(e) => return err_response(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    }
    Json(serde_json::json!({"deleted": deleted})).into_response()
}

pub async fn count(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match db.with_collection(&name, |c| Ok(c.count())) {
        Ok(n) => Json(serde_json::json!({"count": n})).into_response(),
        Err(e) => err_response(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    }
}
