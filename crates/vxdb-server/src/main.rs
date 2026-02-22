mod config;
mod routes;

use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use vxdb_core::collection::Database;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cfg = config::Config::parse();
    let db = Arc::new(Database::new());

    let app = build_router(db);

    let addr = cfg.addr();
    tracing::info!("vxdb server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub fn build_router(db: Arc<Database>) -> Router {
    Router::new()
        .route("/collections", post(routes::create_collection))
        .route("/collections", get(routes::list_collections))
        .route("/collections/{name}", delete(routes::delete_collection))
        .route("/collections/{name}/upsert", post(routes::upsert))
        .route("/collections/{name}/query", post(routes::query))
        .route("/collections/{name}/hybrid", post(routes::hybrid_query))
        .route("/collections/{name}/keyword", post(routes::keyword_search))
        .route("/collections/{name}/delete", post(routes::delete_vectors))
        .route("/collections/{name}/count", get(routes::count))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use axum::http::StatusCode;

    async fn body_json(body: Body) -> serde_json::Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn app() -> Router {
        build_router(Arc::new(Database::new()))
    }

    #[tokio::test]
    async fn test_create_and_list_collections() {
        let app = app();

        // Create
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"name":"docs","dimension":3}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // List
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let collections = json["collections"].as_array().unwrap();
        assert_eq!(collections.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_collection() {
        let app = app();

        // Create
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"docs","dimension":3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Delete
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/collections/docs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_upsert_query_flow() {
        let app = app();

        // Create collection
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"docs","dimension":3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Upsert
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/upsert")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"ids":["a","b","c"],"vectors":[[1,0,0],[0,1,0],[1,0.1,0]],"metadata":[{"color":"red"},{"color":"blue"},{"color":"red"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Count
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/docs/count")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp.into_body()).await;
        assert_eq!(json["count"], 3);

        // Query
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/query")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"vector":[1,0,0],"top_k":2}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["id"], "a");
    }

    #[tokio::test]
    async fn test_filtered_query() {
        let app = app();

        // Create + upsert
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"docs","dimension":3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/upsert")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"ids":["a","b","c"],"vectors":[[1,0,0],[0.9,0.1,0],[0.8,0.2,0]],"metadata":[{"color":"red"},{"color":"blue"},{"color":"red"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Filtered query
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/query")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"vector":[1,0,0],"top_k":10,"filter":{"color":{"$eq":"red"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r["metadata"]["color"] == "red"));
    }

    #[tokio::test]
    async fn test_delete_vectors() {
        let app = app();

        // Create + upsert
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"docs","dimension":3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/upsert")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"ids":["a","b"],"vectors":[[1,0,0],[0,1,0]]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Delete
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/delete")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"ids":["a"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify count
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/docs/count")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp.into_body()).await;
        assert_eq!(json["count"], 1);
    }

    #[tokio::test]
    async fn test_nonexistent_collection() {
        let app = app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/collections/nope/count")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_hybrid_search_flow() {
        let app = app();

        // Create collection
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"docs","dimension":3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Upsert with documents
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/upsert")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"ids":["vec_close","text_match","both"],"vectors":[[1,0,0],[0,1,0],[0.8,0.2,0]],"documents":["unrelated content about cooking","machine learning and artificial intelligence","machine learning for image processing"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Keyword search
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/keyword")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"machine learning","top_k":10}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().unwrap();
        assert!(results.len() >= 2);

        // Hybrid query
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/hybrid")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"vector":[1,0,0],"query":"machine learning","top_k":3,"alpha":0.5}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);

        // "both" should rank well (top 2) because it matches both vector and keyword
        let both_idx = results.iter().position(|r| r["id"] == "both").unwrap();
        assert!(both_idx <= 1, "expected 'both' in top 2, got position {}", both_idx);
    }
}
