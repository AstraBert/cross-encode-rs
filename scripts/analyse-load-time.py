#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars>=1.44,<2",
# ]
# ///


import sys
from typing import TypedDict, cast

import polars as pl


class Stats(TypedDict):
    mean: float
    p50: float
    p90: float
    p99: float


def load_times(file_name: str) -> Stats:
    df = pl.read_csv(file_name)
    times = df["load_time"]
    quantiles = times.quantile([0.5, 0.9, 0.99])
    avg = times.mean()
    return Stats(
        mean=cast(float, avg),
        p50=cast(float, quantiles[0]),
        p90=cast(float, quantiles[1]),
        p99=cast(float, quantiles[2]),
    )


def main() -> None:
    stats = load_times(sys.argv[1])
    print("STATS:")
    print(f"  p50: {stats['p50']}")
    print(f"  p90: {stats['p90']}")
    print(f"  p99: {stats['p99']}")
    print(f"  mean: {stats['mean']}")
