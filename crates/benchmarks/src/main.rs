use std::{
    env,
    error::Error,
    fmt::Display,
    fs::File,
    io::Read,
    path::PathBuf,
    time::{Duration, Instant},
};

use cross_encode_rs::{CrossEncoder, errors::CrossEncoderError};
use flate2::read::GzDecoder;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};

const DATA_PATH: &str = "data/data.jsonl.gz";
const MODEL_BASE_NAME: &str = "model.onnx";
const TOKENIZER_BASE_NAME: &str = "tokenizer.json";

#[derive(Debug)]
enum BenchmarkError {
    Serde(String),
    IO(String),
    CrossEncoder(String),
    CliArgs(String),
}

impl Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serde(s) => write!(f, "SerDe error: {}", s),
            Self::IO(s) => write!(f, "IO error: {}", s),
            Self::CrossEncoder(s) => write!(f, "CrossEncoder error: {}", s),
            Self::CliArgs(s) => write!(f, "Incorrect CLI input: {}", s),
        }
    }
}

impl Error for BenchmarkError {}

impl From<serde_json::Error> for BenchmarkError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value.to_string())
    }
}

impl From<std::io::Error> for BenchmarkError {
    fn from(value: std::io::Error) -> Self {
        Self::IO(value.to_string())
    }
}

impl From<CrossEncoderError> for BenchmarkError {
    fn from(value: CrossEncoderError) -> Self {
        Self::CrossEncoder(value.to_string())
    }
}

impl From<&str> for BenchmarkError {
    fn from(value: &str) -> Self {
        Self::CliArgs(value.to_owned())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct DataEntry {
    query: String,
    positive: Vec<String>,
    negative: Vec<String>,
}

impl DataEntry {
    fn to_rerank_input(&self) -> (&str, Vec<&str>) {
        let mut v: Vec<&str> = self.positive.iter().map(|s| s.as_str()).collect();
        let mut n: Vec<&str> = self.negative.iter().map(|s| s.as_str()).collect();
        v.append(&mut n);
        (self.query.as_str(), v)
    }
}

fn read_gz_to_string(path: &str) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut contents = String::new();
    decoder.read_to_string(&mut contents)?;
    Ok(contents)
}

fn jsonl_content_to_data(content: &str) -> serde_json::Result<Vec<DataEntry>> {
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut entries: Vec<DataEntry> = Vec::with_capacity(lines.len());
    for l in lines {
        let entry: DataEntry = serde_json::from_str(l)?;
        entries.push(entry);
    }
    Ok(entries)
}

struct Stats<T> {
    count: usize,
    min: T,
    p50: T,
    p90: T,
    p99: T,
    max: T,
}

impl<T: Ord + Copy> Stats<T> {
    fn from_values(mut values: Vec<T>) -> Self {
        values.sort_unstable();

        let percentile = |p: f64| -> T {
            let idx = ((values.len() - 1) as f64 * p).round() as usize;
            values[idx]
        };

        Self {
            count: values.len(),
            min: values[0],
            p50: percentile(0.50),
            p90: percentile(0.90),
            p99: percentile(0.99),
            max: values[values.len() - 1],
        }
    }
}

/// Per-request latency (a single `rerank` call, which may score a different
/// number of documents each time) can't be compared across entries on its
/// own, since the dataset has a variable number of documents per query.
/// Alongside it we report per-document latency (each duration divided by
/// its document count) and the document-count distribution itself, so a
/// slow p99 can be told apart from an entry that simply had more documents.
fn print_report(
    entry_durations: Stats<Duration>,
    per_doc_durations: Stats<Duration>,
    doc_counts: Stats<usize>,
) {
    println!("Documents per entry ({} entries)", doc_counts.count);
    println!("  min: {}", doc_counts.min);
    println!("  p50: {}", doc_counts.p50);
    println!("  p90: {}", doc_counts.p90);
    println!("  p99: {}", doc_counts.p99);
    println!("  max: {}", doc_counts.max);

    println!("\nPer-request latency ({} entries)", entry_durations.count);
    println!("  min: {:?}", entry_durations.min);
    println!("  p50: {:?}", entry_durations.p50);
    println!("  p90: {:?}", entry_durations.p90);
    println!("  p99: {:?}", entry_durations.p99);
    println!("  max: {:?}", entry_durations.max);

    println!(
        "\nPer-document latency ({} entries)",
        per_doc_durations.count
    );
    println!("  min: {:?}", per_doc_durations.min);
    println!("  p50: {:?}", per_doc_durations.p50);
    println!("  p90: {:?}", per_doc_durations.p90);
    println!("  p99: {:?}", per_doc_durations.p99);
    println!("  max: {:?}", per_doc_durations.max);
}

fn main() -> Result<(), BenchmarkError> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        return Err("This commands accept exactly one positional argument: MODEL_DIRECTORY".into());
    }

    let model_dir = PathBuf::from(&args[1]);

    if !model_dir.exists() {
        return Err("The provided MODEL_DIRECTORY does not exist".into());
    }

    let mut cross_encoder = CrossEncoder::new(
        model_dir.join(TOKENIZER_BASE_NAME),
        model_dir.join(MODEL_BASE_NAME),
        None,
        None,
    );

    cross_encoder.initialize()?;

    let entries = jsonl_content_to_data(&read_gz_to_string(DATA_PATH)?)?;

    let mut entry_durations = Vec::with_capacity(entries.len());
    let mut per_doc_durations = Vec::with_capacity(entries.len());
    let mut doc_counts = Vec::with_capacity(entries.len());
    let bar = ProgressBar::new(entries.len() as u64);
    for entry in &entries {
        bar.inc(1);
        let (query, documents) = entry.to_rerank_input();

        let start = Instant::now();
        cross_encoder.rerank(query, &documents, false)?;
        let elapsed = start.elapsed();

        entry_durations.push(elapsed);
        per_doc_durations.push(elapsed / documents.len() as u32);
        doc_counts.push(documents.len());
    }

    print_report(
        Stats::from_values(entry_durations),
        Stats::from_values(per_doc_durations),
        Stats::from_values(doc_counts),
    );

    Ok(())
}
