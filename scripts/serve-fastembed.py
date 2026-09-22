#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "fastembed>=0.8,<1",
# ]
# ///
"""FastAPI equivalent of crates/server, for comparing inference time against
the Rust/ONNX server. Exposes the same `/rerank` request/response shape.
"""

import argparse
import time

import uvicorn
from fastapi import FastAPI
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Xenova/ms-marco-MiniLM-L-6-v2",
    help="HuggingFace model id or local path for sentence-transformers CrossEncoder",
)
parser.add_argument("--bind", default="0.0.0.0", help="Address to bind the server to")
parser.add_argument("--port", type=int, default=7433, help="Port to bind the server to")
args = parser.parse_args()

model = TextCrossEncoder(args.model, cuda=False)

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

    scores = model.rerank_pairs(
        [(request.query, doc) for doc in request.documents],
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
