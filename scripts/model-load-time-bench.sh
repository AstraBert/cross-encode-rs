model_dir=$1
model_id=$2

hyperfine --warmup 10 \
    --runs 40 \
    "./target/release/benchmarks $model_dir true true" \
    "./scripts/benchmark-python.py --load-only --fastembed-model $model_id" \
    --export-json crates/benchmarks/results/load-time-${model_dir}.json
