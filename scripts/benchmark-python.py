#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastembed>=0.7",
#   "sentence-transformers[onnx]>=5,<6",
# ]
# ///
"""Python equivalent of crates/benchmarks: reranks every entry of
data/data.jsonl.gz (same format: {query, positive[], negative[]}) with
fastembed and sentence-transformers (ONNX backend, not PyTorch), timing each
call and reporting min/p50/p90/p99/max.
"""

import argparse
import gzip
import json
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from fastembed.rerank.cross_encoder import TextCrossEncoder
from sentence_transformers import CrossEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/data.jsonl.gz", help="Path to the gzipped JSONL dataset")
parser.add_argument(
    "--st-model",
    default="Xenova/ms-marco-MiniLM-L-6-v2",
    help="sentence-transformers model id or path",
)
parser.add_argument(
   "--fastembed-model",
    default="Xenova/ms-marco-MiniLM-L-6-v2",
    help="fastembed reranker model name",
)
args = parser.parse_args()


T = TypeVar("T", float, int)


@dataclass
class Stats(Generic[T]):
    count: int
    min: T
    p50: T
    p90: T
    p99: T
    max: T

    @staticmethod
    def from_values(values: list[T]) -> "Stats[T]":
        values = sorted(values)

        def percentile(p: float) -> T:
            idx = round((len(values) - 1) * p)
            return values[idx]

        return Stats(
            count=len(values),
            min=values[0],
            p50=percentile(0.50),
            p90=percentile(0.90),
            p99=percentile(0.99),
            max=values[-1],
        )


def format_duration_stats(stats: Stats[float]) -> str:
    return "\n".join(
        f"  {label}: {value * 1000:.2f}ms"
        for label, value in [
            ("min", stats.min),
            ("p50", stats.p50),
            ("p90", stats.p90),
            ("p99", stats.p99),
            ("max", stats.max),
        ]
    )


def format_count_stats(stats: Stats[int]) -> str:
    return "\n".join(
        f"  {label}: {value}"
        for label, value in [
            ("min", stats.min),
            ("p50", stats.p50),
            ("p90", stats.p90),
            ("p99", stats.p99),
            ("max", stats.max),
        ]
    )


def load_entries(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_rerank_input(entry: dict) -> tuple[str, list[str]]:
    return entry["query"], [*entry["positive"], *entry["negative"]]


# Per-request latency (a single rerank call, which may score a different
# number of documents each time) can't be compared across entries on its
# own, since the dataset has a variable number of documents per query.
# Alongside it we report per-document latency (each duration divided by its
# document count) and the document-count distribution itself, so a slow p99
# can be told apart from an entry that simply had more documents.
def print_report(entry_durations: list[float], doc_counts: list[int]) -> None:
    per_doc_durations = [d / n for d, n in zip(entry_durations, doc_counts)]

    print(f"Documents per entry ({len(doc_counts)} entries)")
    print(format_count_stats(Stats.from_values(doc_counts)))

    print(f"\nPer-request latency ({len(entry_durations)} entries)")
    print(format_duration_stats(Stats.from_values(entry_durations)))

    print(f"\nPer-document latency ({len(per_doc_durations)} entries)")
    print(format_duration_stats(Stats.from_values(per_doc_durations)))


def benchmark_sentence_transformers(entries: list[dict], model_name: str) -> tuple[list[float], list[int]]:
    model = CrossEncoder(model_name, num_labels=1, backend="onnx")

    durations = []
    doc_counts = []
    for entry in tqdm(entries):
        query, documents = to_rerank_input(entry)
        start = time.monotonic()
        model.predict([(query, doc) for doc in documents])
        durations.append(time.monotonic() - start)
        doc_counts.append(len(documents))

    return durations, doc_counts


def benchmark_fastembed(entries: list[dict], model_name: str) -> tuple[list[float], list[int]]:
    model = TextCrossEncoder(model_name=model_name)

    durations = []
    doc_counts = []
    for entry in tqdm(entries):
        query, documents = to_rerank_input(entry)
        start = time.monotonic()
        list(model.rerank(query, documents))
        durations.append(time.monotonic() - start)
        doc_counts.append(len(documents))

    return durations, doc_counts


def main() -> None:
    entries = load_entries(args.data)

    print(f"sentence-transformers ({args.st_model}, onnx backend)")
    print_report(*benchmark_sentence_transformers(entries, args.st_model))
    print()

    print(f"fastembed ({args.fastembed_model})")
    print_report(*benchmark_fastembed(entries, args.fastembed_model))


if __name__ == "__main__":
    main()
