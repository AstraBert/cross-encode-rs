model_dir=$1
model_id=$2

# warmup (rust)
for i in {0..10}
do
    ./target/release/benchmarks $model_dir true true > /dev/null
done

echo "load_time" > crates/benchmarks/results/load-time-rs-${model_dir}.csv

# benchmark runs (rust)
for i in {0..40}
do
    ./target/release/benchmarks $model_dir true true >> crates/benchmarks/results/load-time-rs-${model_dir}.csv
done

# warmup (fastembed)
for i in {0..10}
do
    ./scripts/benchmark-python.py --load-only --fastembed-model $model_id > /dev/null
done

echo "load_time" > crates/benchmarks/results/load-time-fastembed-${model_dir}.csv

# benchmark runs (fastembed)
for i in {0..40}
do
    ./scripts/benchmark-python.py --load-only --fastembed-model $model_id >> crates/benchmarks/results/load-time-fastembed-${model_dir}.csv
done
