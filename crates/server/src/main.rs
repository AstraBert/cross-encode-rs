use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use axum::{
    Router, extract::State, http::StatusCode, response::IntoResponse, response::Json, routing::post,
};
use clap::Parser;
use cross_encode_rs::{CrossEncoder, RerankResult};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};

/// `ort::Session::run` requires `&mut self`, so a single session can't serve
/// requests from multiple threads concurrently. Instead, default to one
/// worker (each with its own model + tokenizer) per available core, and
/// load-balance requests across them.
fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[derive(Debug, Parser)]
/// Serve a
/// cross-encoder model
struct Args {
    /// Path to the cross-encoder
    /// model to serve (as ONNX file)
    #[arg(long, short)]
    model: String,
    /// Path to the tokenizer
    /// to use along with the
    /// cross-encoder model
    #[arg(long, short)]
    tokenizer: String,
    /// Size of the worker
    /// channel buffer
    #[arg(long, short, default_value_t = 100)]
    buffer_size: usize,
    /// Number of parallel inference workers to spawn, each with its own
    /// model + tokenizer instance. Defaults to the number of available cores.
    #[arg(long, short, default_value_t = default_workers())]
    workers: usize,
    /// Intra-op threads for the ONNX session
    /// of each worker to allocate. When running
    /// several workers, keep this low to avoid
    /// oversubscribing the available cores.
    #[arg(long, default_value = None)]
    threads: Option<usize>,
    /// Address to bind the server to,
    /// defaults to 0.0.0.0:7432
    #[arg(long, default_value = None)]
    bind: Option<String>,
}

#[derive(Debug, Clone)]
enum RerankAPIError {
    ChannelError(String),
    InferenceError(String),
    WorkerUnavailableError,
}

impl IntoResponse for RerankAPIError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            Self::ChannelError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Error while trying to get a response from the worker: {}",
                    msg
                ),
            ),
            Self::InferenceError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error while running inference: {}", msg),
            ),
            Self::WorkerUnavailableError => (
                StatusCode::SERVICE_UNAVAILABLE,
                "The inference worker is unavailable".to_string(),
            ),
        };

        (status, message).into_response()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    return_documents: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RerankResponseItem {
    index: usize,
    score: f32,
    document: Option<String>,
}

#[derive(Debug, Clone)]
struct AppState {
    workers: Arc<Vec<mpsc::Sender<WorkerRequest>>>,
    next_worker: Arc<AtomicUsize>,
}

impl AppState {
    /// Picks the next worker's channel in round-robin order.
    fn next_worker(&self) -> &mpsc::Sender<WorkerRequest> {
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        &self.workers[idx]
    }
}

impl<'a> From<RerankResult<'a>> for RerankResponseItem {
    fn from(value: RerankResult) -> Self {
        Self {
            index: value.index,
            score: value.score,
            document: value.document.map(|d| d.to_string()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RerankResponse {
    items: Vec<RerankResponseItem>,
}

struct WorkerRequest {
    input: RerankRequest,
    reply: oneshot::Sender<(Option<RerankResponse>, Option<String>)>,
}

fn spawn_inference_worker(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    buffer_size: usize,
    intra_threads: Option<usize>,
) -> mpsc::Sender<WorkerRequest> {
    let (tx, mut rx) = mpsc::channel::<WorkerRequest>(buffer_size);

    std::thread::spawn(move || {
        let mut model = CrossEncoder::new(tokenizer_path, model_path, intra_threads);

        // blocking_recv because this is a plain OS thread, not an async task
        while let Some(req) = rx.blocking_recv() {
            let result = model.rerank(
                &req.input.query,
                req.input
                    .documents
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>()
                    .as_slice(),
                req.input.return_documents,
            );
            let response = match result {
                Ok(results) => {
                    let items = results
                        .iter()
                        .copied()
                        .map(|s| RerankResponseItem::from(s))
                        .collect::<Vec<RerankResponseItem>>();
                    (Some(RerankResponse { items }), None)
                }
                Err(e) => (None, Some(e.to_string())),
            };
            let _ = req.reply.send(response);
        }
    });

    tx
}

fn spawn_inference_workers(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    buffer_size: usize,
    intra_threads: Option<usize>,
    num_workers: usize,
) -> Vec<mpsc::Sender<WorkerRequest>> {
    (0..num_workers.max(1))
        .map(|_| {
            spawn_inference_worker(
                model_path.clone(),
                tokenizer_path.clone(),
                buffer_size,
                intra_threads,
            )
        })
        .collect()
}

#[tracing::instrument]
async fn rerank(
    State(state): State<AppState>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, RerankAPIError> {
    let (reply_tx, reply_rx) = oneshot::channel();

    let worker_req = WorkerRequest {
        input: request,
        reply: reply_tx,
    };

    let start = Instant::now();

    if state.next_worker().send(worker_req).await.is_err() {
        tracing::error!("Could not reach the worker channel");
        return Err(RerankAPIError::WorkerUnavailableError);
    }

    match reply_rx.await {
        Ok((response_opt, error_opt)) => {
            if let Some(response) = response_opt {
                let elapsed = start.elapsed().as_millis();
                let msg = format!(
                    "Reranked {} documents in {}ms",
                    response.items.len(),
                    elapsed
                );
                tracing::info!(msg);
                Ok(Json(response))
            } else {
                let msg = error_opt.unwrap_or("unknown error".to_string());
                tracing::error!(msg);
                Err(RerankAPIError::InferenceError(msg))
            }
        }
        Err(e) => {
            let msg = format!(
                "Error while trying to get a response from the worker: {}",
                e
            );
            tracing::error!(msg);
            Err(RerankAPIError::ChannelError(e.to_string()))
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt().json().init();

    let workers = spawn_inference_workers(
        PathBuf::from(args.model),
        PathBuf::from(args.tokenizer),
        args.buffer_size,
        args.threads,
        args.workers,
    );
    let msg = format!("Spawned {} inference worker(s)", workers.len());
    tracing::info!(msg);

    let state = AppState {
        workers: Arc::new(workers),
        next_worker: Arc::new(AtomicUsize::new(0)),
    };

    let app = Router::new()
        .route("/rerank", post(rerank))
        .with_state(state);

    let bind_address = args.bind.unwrap_or("0.0.0.0:7432".to_string());

    let listener = tokio::net::TcpListener::bind(&bind_address).await.unwrap();

    let msg = format!("Server running on {}", &bind_address);
    tracing::info!(msg);

    axum::serve(listener, app).await.unwrap();
}
