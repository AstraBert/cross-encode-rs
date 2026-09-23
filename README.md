# cross-encode-rs

Fast ONNX cross-encoder inference with [`ort`](https://ort.pyke.io/).

> [!NOTE]
>
> Currently the crate only works on Linux and MacOS.

## Layout

- `crates/cross-encode-rs` — library. `CrossEncoder::rerank(query, documents, with_documents)` scores documents against a query. Supports single-label (sigmoid) and two-label (softmax) heads. Optional `hf-hub` feature to pull models from Hugging Face.
- `crates/server` — Axum HTTP server wrapping the library (`POST /rerank`).
- `load-test` — Go load-test client for comparing inference performance across servers. See `load-test/README.md`.
- `scripts/generate-test-scores.py` — regenerates `crates/cross-encode-rs/testfiles/scores.jsonl`, the golden scores used by `tests/scores_parity.rs`.
- `scripts/serve-python.py` — Python reference server used by `load-test`.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
```

Tests run against the ONNX model + tokenizer in `crates/cross-encode-rs/testfiles/`, no downloads needed.

## Running the server

```bash
cargo run -p server -- \
  --model crates/cross-encode-rs/testfiles/model.onnx \
  --tokenizer crates/cross-encode-rs/testfiles/tokenizer.json
```

Flags (`cargo run -p server -- --help`):

- `--model` / `-m`, `--tokenizer` / `-t` — paths to the ONNX model and `tokenizer.json`.
- `--workers` / `-w` — number of inference workers, defaults to core count.
- `--threads` — intra-op threads per worker. May have no effect depending on the ONNX Runtime build; try `OMP_NUM_THREADS` instead if changing this doesn't move CPU usage.
- `--buffer-size` / `-b` — per-worker channel buffer size.
- `--bind` — bind address, defaults to `0.0.0.0:7432`.

### API

```
POST /rerank
{
  "query": "How many people live in Berlin?",
  "documents": ["...", "..."],
  "return_documents": true
}
```

```
{
  "items": [
    { "index": 0, "score": 0.999, "document": "..." },
    ...
  ]
}
```

### Docker

```bash
docker build . -t cross-encoder-server
```

Expects `model.onnx` and `tokenizer.json` at the workspace root at build time.

### Health checks

- `GET /livez` returns 200 while every inference worker thread is alive.
- `GET /readyz` returns 200 once every worker has loaded its model. It returns 503 after SIGTERM, while in-flight requests finish.

### Kubernetes

Manifests are in `deploy/k8s`. They run two replicas on two different nodes, behind a Service and a Traefik route that retries failed connections on the other pod.

Build the image, push it to a registry your nodes can pull from, and set that image in `deploy/k8s/kustomization.yaml`.

```bash
# Choose the nodes that run inference
kubectl label node <node> cross-encoder=enabled

kubectl apply -k deploy/k8s
```

Keep `--workers` equal to the CPU limit in `deployment.yaml`.

## Benchmarks

View benchmark results [here](https://astrabert.github.io/cross-encode-rs/). All results come from benchmark runs on a Mac M4 Max, with 48GB RAM and 14CPU (ARM architecture).

> [!NOTE]
>
> For a fair comparison, here `sentence-transformers` is used with the `onnx` backend, not `torch` (default).

### Static

Static benchmarks are run against `sentence-transformers` and `fastembed` on [mteb/scidocs-reranking](https://huggingface.co/datasets/mteb/scidocs-reranking).

For `cross-encode-rs`, you need to download `model.onnx` and `tokenizer.json` from [Xenova/ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) and place them under `xenova/`.

```bash
# from the repo root

# cross-encode-rs 
cargo build --release -p benchmarks
./target/release/benchmarks

# fastembed and sentence-transformers
# (needs uv)
./scripts/benchmark-python.py
```

### Server

The Axum-based server in [crates/server](crates/server/src/main.rs) is compared against FastAPI-based servers with [`sentence-transformers`](scripts/serve-python.py) and [`fastembed`](scripts/serve-fastembed.py). See more in the [dedicated README](./load-test/README.md).

The reported results are achieved by running (as server processes):

```bash
# Axum server
cargo build --release -p server
./target/release/server --model xenova/model.onnx \
    --tokenizer xenova/tokenizer.json \
    --threads 1 \
    --workers 14 \
    --buffer-size 1000000

# FastAPI + sentence-transformers
./scripts/serve-python.py  --model Xenova/ms-marco-MiniLM-L-6-v2 

# FastAPI + fastembed
./scripts/serve-fastembed.py
```
