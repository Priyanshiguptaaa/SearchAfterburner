use axum::{
    extract::Query,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use ranker_rs::scoring::{RerankRequest, RerankResponse, score_docs, PruneConfig};
use serde::Deserialize;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{info, error};
use rand::Rng;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let app = Router::new()
        .route("/rerank", post(handle_rerank))
        .route("/bench", get(handle_bench))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
        );

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8088")
        .await
        .expect("Failed to bind to address");

    info!("Reranker service starting on http://0.0.0.0:8088");
    info!("POST /rerank endpoint ready");
    info!("GET /bench endpoint ready");

    axum::serve(listener, app).await.expect("Server failed to start");
}

async fn handle_rerank(
    Json(payload): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, StatusCode> {
    info!("Received rerank request: {} query tokens, {} documents, topk={}", 
          payload.q_tokens.len(), payload.d_tokens.len(), payload.topk);
    info!("SIGIR 2025: Lossless token pruning enabled (q_max={}, d_max={})", 
          payload.prune.q_max, payload.prune.d_max);

    // Validate input
    if payload.q_tokens.is_empty() || payload.d_tokens.is_empty() {
        error!("Empty query tokens or document tokens");
        return Err(StatusCode::BAD_REQUEST);
    }

    if payload.q_tokens[0].is_empty() {
        error!("Empty query token vectors");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate all document tokens have same dimension
    let expected_dim = payload.q_tokens[0].len();
    for (i, doc_tokens) in payload.d_tokens.iter().enumerate() {
        for (j, token) in doc_tokens.iter().enumerate() {
            if token.len() != expected_dim {
                error!("Dimension mismatch: doc {} token {} has {} dims, expected {}", 
                       i, j, token.len(), expected_dim);
                return Err(StatusCode::BAD_REQUEST);
            }
        }
    }

    let start_time = std::time::Instant::now();

    // Perform reranking
    let (order, scores, perf) = score_docs(
        &payload.q_tokens,
        &payload.d_tokens,
        payload.topk,
        &payload.prune,
    );

    let total_time = start_time.elapsed().as_secs_f32() * 1000.0;
    info!("Reranking completed in {:.2}ms, p50: {:.2}ms, p95: {:.2}ms", 
          total_time, perf.per_doc_ms_p50, perf.per_doc_ms_p95);

    let response = RerankResponse {
        order,
        scores,
        perf,
    };

    Ok(Json(response))
}

#[derive(Deserialize)]
struct BenchParams {
    n_docs: Option<usize>,
    td: Option<usize>,
    d: Option<usize>,
    prune: Option<String>,
}

#[derive(serde::Serialize)]
struct BenchResponse {
    n_docs: usize,
    td: usize,
    d: usize,
    p50_ms: f32,
    p95_ms: f32,
    threads: usize,
    cpu_flags: String,
}

async fn handle_bench(Query(params): Query<BenchParams>) -> Result<Json<BenchResponse>, StatusCode> {
    let n_docs = params.n_docs.unwrap_or(100);
    let td = params.td.unwrap_or(64);
    let d = params.d.unwrap_or(128);
    let prune_setting = params.prune.unwrap_or_else(|| "16/64".to_string());
    
    info!("Running microbench: n_docs={}, td={}, d={}, prune={}", n_docs, td, d, prune_setting);
    
    // Parse prune setting
    let (q_max, d_max) = if prune_setting == "none" {
        (td, d)
    } else if prune_setting.contains("/") {
        let parts: Vec<&str> = prune_setting.split("/").collect();
        if parts.len() == 2 {
            (parts[0].parse().unwrap_or(td), parts[1].parse().unwrap_or(d))
        } else {
            (td, d)
        }
    } else {
        (td, d)
    };
    
    // Generate random unit-norm matrices
    let mut rng = rand::thread_rng();
    
    // Generate query tokens
    let mut q_tokens = Vec::new();
    for _ in 0..td {
        let mut token = vec![0.0; d];
        for i in 0..d {
            token[i] = rng.gen_range(-1.0..1.0);
        }
        // Normalize to unit length
        let norm: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for i in 0..d {
                token[i] /= norm;
            }
        }
        q_tokens.push(token);
    }
    
    // Generate document tokens
    let mut d_tokens = Vec::new();
    for _ in 0..n_docs {
        let mut doc = Vec::new();
        for _ in 0..td {
            let mut token = vec![0.0; d];
            for i in 0..d {
                token[i] = rng.gen_range(-1.0..1.0);
            }
            // Normalize to unit length
            let norm: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for i in 0..d {
                    token[i] /= norm;
                }
            }
            doc.push(token);
        }
        d_tokens.push(doc);
    }
    
    // Configure pruning
    let prune_config = PruneConfig {
        q_max,
        d_max,
        method: "idf_norm".to_string(),
    };
    
    // Run benchmark
    let start_time = std::time::Instant::now();
    let (_, _, perf) = score_docs(&q_tokens, &d_tokens, n_docs, &prune_config);
    let total_time = start_time.elapsed().as_secs_f32() * 1000.0;
    
    // Detect CPU flags (simplified)
    let cpu_flags = if std::arch::is_x86_feature_detected!("avx512f") {
        "AVX-512"
    } else if std::arch::is_x86_feature_detected!("avx2") {
        "AVX2"
    } else {
        "SSE"
    };
    
    let threads = rayon::current_num_threads();
    
    info!("Microbench completed: {:.2}ms total, p50: {:.2}ms, p95: {:.2}ms", 
          total_time, perf.per_doc_ms_p50, perf.per_doc_ms_p95);
    
    let response = BenchResponse {
        n_docs,
        td,
        d,
        p50_ms: perf.per_doc_ms_p50,
        p95_ms: perf.per_doc_ms_p95,
        threads,
        cpu_flags: cpu_flags.to_string(),
    };
    
    Ok(Json(response))
}
