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
from typing import TYPE_CHECKING, Generic, TypeVar

from tqdm import tqdm

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from sentence_transformers import CrossEncoder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", default="data/data.jsonl.gz", help="Path to the gzipped JSONL dataset"
)
parser.add_argument(
    "--st-model",
    default="",
    help="sentence-transformers model id or path. If not provided, skips running sentence-transformers.",
)
parser.add_argument(
    "--remote-code",
    default=False,
    action="store_true",
    help="enable remote_code = True on sentence-transformers",
)
parser.add_argument(
    "--load-only",
    default=False,
    action="store_true",
    help="only load models without running the benchmark",
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


def load_st_model(model_name: str, remote_code: bool) -> tuple["CrossEncoder", float]:
    from sentence_transformers import CrossEncoder

    start = time.monotonic()
    model = CrossEncoder(model_name, num_labels=1, backend="onnx", model_kwargs={"provider": "CPUExecutionProvider"}, trust_remote_code=remote_code)
    return (model, (time.monotonic() - start) * 1000)


def benchmark_sentence_transformers(
    entries: list[dict], model: "CrossEncoder",
) -> tuple[list[float], list[int]]:
    durations = []
    doc_counts = []
    for entry in tqdm(entries):
        query, documents = to_rerank_input(entry)
        start = time.monotonic()
        model.predict([(query, doc) for doc in documents])
        durations.append(time.monotonic() - start)
        doc_counts.append(len(documents))

    return durations, doc_counts


def load_fastembed_model(model_name: str) -> tuple["TextCrossEncoder", float]:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    start = time.monotonic()
    model = TextCrossEncoder(model_name=model_name)
    return (model, (time.monotonic() - start) * 1000)


def time_fastembed_session(model_name: str) -> float:
    """Times only the ONNX session creation, to match `init_model()` on the
    Rust side. The constructor also resolves the model files and loads the
    tokenizer, so it runs lazily and untimed, and the session is built through
    the base class to skip `load_tokenizer`."""
    from fastembed.common.onnx_model import OnnxModel
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    inner = TextCrossEncoder(model_name=model_name, lazy_load=True).model
    start = time.monotonic()
    OnnxModel._load_onnx_model(
        inner,
        model_dir=inner._model_dir,
        model_file=inner.model_description.model_file,
        threads=inner.threads,
        providers=inner.providers,
        cuda=inner.cuda,
        device_id=inner.device_id,
        extra_session_options=inner._extra_session_options,
    )
    return (time.monotonic() - start) * 1000


def benchmark_fastembed(
    entries: list[dict], model: "TextCrossEncoder"
) -> tuple[list[float], list[int]]:
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
    if args.load_only:
        if args.st_model:
            _, t = load_st_model(args.st_model, args.remote_code)
            print(t)
        else:
            print(time_fastembed_session(args.fastembed_model))
        return

    entries = load_entries(args.data)

    if args.st_model:
        print(f"sentence-transformers ({args.st_model}, onnx backend)")
        model, _ = load_st_model(args.st_model, args.remote_code)
        print_report(*benchmark_sentence_transformers(entries, model))
        print()

    print(f"fastembed ({args.fastembed_model})")
    model, _ = load_fastembed_model(args.fastembed_model)
    print_report(*benchmark_fastembed(entries, model))


if __name__ == "__main__":
    main()
