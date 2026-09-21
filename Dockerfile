# ── Build stage ──────────────────────────────────────────────────────────────
FROM rust:1.98-slim AS builder

# System deps needed by some crates
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy workspace-level Cargo files first for layer caching
COPY Cargo.toml Cargo.lock model.onnx tokenizer.json ./

# Copy both workspace members
COPY crates/ crates/

# Build only the rest-server binary in release mode
RUN cargo build --release -p server

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/server /usr/local/bin/cross-encoder-server
COPY --from=builder /workspace/model.onnx ./model.onnx
COPY --from=builder /workspace/tokenizer.json ./tokenizer.json

EXPOSE 7432

ENV OMP_NUM_THREADS=10

CMD ["cross-encoder-server", "--model", "model.onnx", "--tokenizer", "tokenizer.json", "--threads", "10"]
