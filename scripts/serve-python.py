#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "sentence-transformers>=6.1.0,<7",
# ]
# ///
"""FastAPI equivalent of crates/server, for comparing inference time against
the Rust/ONNX server. Exposes the same `/rerank` request/response shape and
serializes inference through a single worker thread, mirroring the Rust
server's single-threaded `CrossEncoder` worker (see spawn_inference_worker in
crates/server/src/main.rs) rather than letting FastAPI's default threadpool
run several inferences concurrently under the GIL.
"""

import argparse
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="cross-encoder/ms-marco-TinyBERT-L2-v2",
    help="HuggingFace model id or local path for sentence-transformers CrossEncoder",
)
parser.add_argument("--bind", default="0.0.0.0", help="Address to bind the server to")
parser.add_argument("--port", type=int, default=7433, help="Port to bind the server to")
args = parser.parse_args()

model = CrossEncoder(args.model, num_labels=1)
inference_lock = threading.Lock()

app = FastAPI()


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    return_documents: bool = False


class RerankResponseItem(BaseModel):
    index: int
    score: float
    document: str | None = None


class RerankResponse(BaseModel):
    items: list[RerankResponseItem]


@app.post("/rerank")
def rerank(request: RerankRequest) -> RerankResponse:
    start = time.monotonic()

    # Serialize inference to mirror the Rust server's single background
    # worker thread, so throughput comparisons aren't skewed by Python-side
    # concurrency the Rust server doesn't have.
    with inference_lock:
        scores = model.predict(
            [(request.query, doc) for doc in request.documents],
            activation_fn=torch.nn.Sigmoid(),
        )

    items = [
        RerankResponseItem(
            index=idx,
            score=float(score),
            document=doc if request.return_documents else None,
        )
        for idx, (score, doc) in enumerate(zip(scores, request.documents))
    ]

    elapsed_ms = (time.monotonic() - start) * 1000
    print(f"Reranked {len(items)} documents in {elapsed_ms:.2f}ms")

    return RerankResponse(items=items)


if __name__ == "__main__":
    uvicorn.run(app, host=args.bind, port=args.port)
