//! Verifies that `CrossEncoder::rerank` scores match the reference scores
//! produced by `scripts/generate-test-scores.py` (sentence-transformers,
//! sigmoid-activated) for the same query/document pairs, within tolerance.

use std::fs;

use cross_encode_rs::CrossEncoder;
use serde::Deserialize;

const QUERY: &str = "How many people live in Berlin?";

const DOCUMENTS: &[&str] = &[
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
    "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
    "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
    "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
    "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
    "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
    "In 2015, the total labour force in Berlin was 1.85 million.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
    "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
];

// Absolute score tolerance between the Rust ONNX pipeline and the Python
// sentence-transformers reference. The two runtimes use different backends
// (ONNX Runtime vs. PyTorch), so exact equality isn't expected.
const SCORE_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct ReferenceScore {
    corpus_id: usize,
    score: f32,
    #[allow(dead_code)]
    text: String,
}

fn load_reference_scores() -> Vec<ReferenceScore> {
    let contents =
        fs::read_to_string("testfiles/scores.jsonl").expect("scores.jsonl should be readable");

    let mut scores: Vec<ReferenceScore> = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("each line should be valid JSON"))
        .collect();

    scores.sort_by_key(|s| s.corpus_id);
    scores
}

#[test]
fn rerank_scores_match_python_reference() {
    let reference = load_reference_scores();
    assert_eq!(reference.len(), DOCUMENTS.len());

    let mut ce = CrossEncoder::new(
        "testfiles/tokenizer.json".into(),
        "testfiles/model.onnx".into(),
        None,
        None,
        true,
    );
    let mut results = ce
        .rerank(QUERY, DOCUMENTS, false)
        .expect("rerank should succeed");
    results.sort_by_key(|r| r.index);

    for (result, reference) in results.iter().zip(reference.iter()) {
        assert_eq!(result.index, reference.corpus_id);
        let diff = (result.score - reference.score).abs();
        assert!(
            diff <= SCORE_TOLERANCE,
            "score mismatch for corpus_id {}: rust={} python={} diff={}",
            reference.corpus_id,
            result.score,
            reference.score,
            diff
        );
    }
}
