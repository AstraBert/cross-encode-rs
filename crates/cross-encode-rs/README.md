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
