use std::path::PathBuf;

use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::{
    PaddingParams, TruncationParams, pad_encodings,
    pipeline::{Encoding, PipelineTokenizer},
};

use crate::{
    inference::run_inference,
    tokenizer::{encode_batch, load_tokenizer},
};

#[cfg(feature = "hf-hub")]
use crate::hf::download_from_hub;

pub mod errors;
#[cfg(feature = "hf-hub")]
pub mod hf;
pub mod inference;
pub mod tokenizer;

pub use errors::CrossEncoderError;

pub const DEFAULT_BATCH_SIZE: usize = 16;

/// Scores documents against a query with an ONNX cross-encoder. Model and
/// tokenizer are lazily loaded on first use.
pub struct CrossEncoder {
    pub tokenizer_path: PathBuf,
    pub model_path: PathBuf,
    pub intra_threads: Option<usize>,
    pub fallback_tokenizer_max_length: Option<usize>,
    pub use_type_ids: bool,
    model: Option<Session>,
    tokenizer: Option<PipelineTokenizer>,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

/// A single document's rerank outcome: its original index, relevance score
/// in `[0, 1]`, and, if requested, the document text.
#[derive(Debug, Clone, Copy)]
pub struct RerankResult<'a> {
    pub index: usize,
    pub score: f32,
    pub document: Option<&'a str>,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

impl CrossEncoder {
    /// Builds a `CrossEncoder` for local model/tokenizer files. Nothing is
    /// loaded yet; use [`CrossEncoder::initialize`] or call [`CrossEncoder::rerank`] directly.
    ///
    /// Set `use_type_ids` to true if the cross-encoder is BERT-based and requires
    /// also token type IDs for inference, `false` if it's RoBERTa-based and
    /// only requires token IDs and the attention mask.
    pub fn new(
        tokenizer_path: PathBuf,
        model_path: PathBuf,
        intra_threads: Option<usize>,
        fallback_tokenizer_max_length: Option<usize>,
        use_type_ids: bool,
    ) -> Self {
        Self {
            model_path,
            tokenizer_path,
            intra_threads,
            fallback_tokenizer_max_length,
            use_type_ids,
            tokenizer: None,
            model: None,
            truncation: None,
            padding: None,
        }
    }

    /// Downloads model/tokenizer from the Hugging Face Hub, then builds a
    /// `CrossEncoder` from the cached files.
    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(
        model_id: &str,
        force_download: bool,
        intra_threads: Option<usize>,
        fallback_tokenizer_max_length: Option<usize>,
        use_type_ids: bool,
    ) -> Result<Self, CrossEncoderError> {
        let (model_path, tokenizer_path) = download_from_hub(model_id, force_download).await?;

        Ok(Self {
            model_path,
            tokenizer_path,
            intra_threads,
            fallback_tokenizer_max_length,
            use_type_ids,
            model: None,
            tokenizer: None,
            truncation: None,
            padding: None,
        })
    }

    pub fn init_model(&mut self) -> Result<(), CrossEncoderError> {
        if self.model.is_some() {
            return Ok(());
        }
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads.unwrap_or(default_threads()))?
            .commit_from_file(&self.model_path)?;

        self.model = Some(model);
        Ok(())
    }

    pub fn init_tokenizer(&mut self) -> Result<(), CrossEncoderError> {
        if self.tokenizer.is_some() {
            return Ok(());
        }

        let (tok, pad, trn) =
            load_tokenizer(&self.tokenizer_path, self.fallback_tokenizer_max_length)?;

        self.tokenizer = Some(tok);
        self.truncation = Some(trn);
        self.padding = Some(pad);
        Ok(())
    }

    /// Eagerly loads the tokenizer and model, instead of on first [`CrossEncoder::rerank`] call.
    pub fn initialize(&mut self) -> Result<(), CrossEncoderError> {
        self.init_tokenizer()?;
        self.init_model()?;

        Ok(())
    }

    /// Scores each document against `query`, preserving input order.
    ///
    /// Documents are grouped by token length into batches of `batch_size`
    /// (default [`DEFAULT_BATCH_SIZE`]), so each batch is only padded to its
    /// own longest sequence rather than the longest in the whole request.
    pub fn rerank<'a>(
        &'a mut self,
        query: &str,
        documents: &[&'a str],
        with_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult<'a>>, CrossEncoderError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        self.init_model()?;
        self.init_tokenizer()?;
        if let Some(ref tokenizer) = self.tokenizer
            && let Some(ref mut model) = self.model
            && let Some(ref padding) = self.padding
            && let Some(ref truncation) = self.truncation
        {
            let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
            let encodings = encode_batch(tokenizer, truncation, query, documents)?;

            // Sort (original index, encoding) pairs by token length, then split
            // them so `order[i]` is the input position of `sorted[i]`.
            let mut indexed: Vec<(usize, Encoding)> = encodings.into_iter().enumerate().collect();
            indexed.sort_by_key(|(_, e)| e.len());
            let (order, mut sorted): (Vec<usize>, Vec<Encoding>) = indexed.into_iter().unzip();

            let mut scores = vec![0.0f32; documents.len()];
            for (idxs, batch) in order.chunks(batch_size).zip(sorted.chunks_mut(batch_size)) {
                pad_encodings(batch, padding)?;
                let batch_scores = run_inference(model, batch, self.use_type_ids)?;
                // Scatter each score back to its document's original position.
                for (&idx, score) in idxs.iter().zip(batch_scores) {
                    scores[idx] = score;
                }
            }

            let results = scores
                .into_iter()
                .enumerate()
                .map(|(idx, score)| RerankResult {
                    index: idx,
                    score,
                    document: with_documents.then(|| documents[idx]),
                })
                .collect();
            return Ok(results);
        }
        Err("Model and tokenizer where not correctly loaded".into())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Read,
        process::{Command, Stdio},
    };

    use super::*;

    fn test_encoder() -> CrossEncoder {
        CrossEncoder::new(
            "testfiles/tokenizer.json".into(),
            "testfiles/model.onnx".into(),
            None,
            None,
            true,
        )
    }

    #[test]
    fn rerank_orders_relevant_document_first() {
        let mut ce = test_encoder();
        let documents = [
            "rust is a systems programming language",
            "paris is in france",
        ];
        let results = ce
            .rerank("what is rust", &documents, true, None)
            .expect("rerank should succeed");

        assert_eq!(results.len(), documents.len());
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn rerank_without_documents_omits_document_field() {
        let mut ce = test_encoder();
        let documents = [
            "rust is a systems programming language",
            "paris is in france",
        ];
        let results = ce
            .rerank("what is rust", &documents, false, None)
            .expect("rerank should succeed");

        assert!(results.iter().all(|r| r.document.is_none()));
    }

    #[test]
    fn rerank_with_documents_returns_matching_document() {
        let mut ce = test_encoder();
        let documents = [
            "rust is a systems programming language",
            "paris is in france",
        ];
        let results = ce
            .rerank("what is rust", &documents, true, None)
            .expect("rerank should succeed");

        for (idx, result) in results.iter().enumerate() {
            assert_eq!(result.index, idx);
            assert_eq!(result.document, Some(documents[idx]));
        }
    }

    #[test]
    fn test_default_threads() {
        if std::env::consts::OS == "linux" {
            let cmd = Command::new("nproc")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("Should be able to spawn command");
            let output = cmd
                .wait_with_output()
                .expect("Command should exit successfully");
            let mut s = String::new();
            let mut serr = String::new();
            output
                .stdout
                .as_slice()
                .read_to_string(&mut s)
                .expect("Should read to string");
            output
                .stderr
                .as_slice()
                .read_to_string(&mut serr)
                .expect("Should read to string");
            let nproc: usize = s.trim().parse().expect("Should parse to a number");
            let threads = default_threads();
            assert_eq!(nproc, threads);
        } else if std::env::consts::OS == "macos" {
            let cmd = Command::new("getconf")
                .arg("_NPROCESSORS_ONLN")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("Should be able to spawn command");
            let output = cmd
                .wait_with_output()
                .expect("Command should exit successfully");
            let mut s = String::new();
            output
                .stdout
                .as_slice()
                .read_to_string(&mut s)
                .expect("Should read to string");
            let nproc: usize = s.trim().parse().expect("Should parse to a number");
            let threads = default_threads();
            assert_eq!(nproc, threads);
        } else {
            // skip
            return;
        }
    }

    #[test]
    fn rerank_batching_preserves_input_order() {
        let mut ce = test_encoder();
        // Lengths deliberately out of order so sorting reshuffles them.
        let documents = [
            "rust is a systems programming language focused on safety and speed",
            "paris",
            "the borrow checker enforces ownership rules at compile time in rust",
            "a cat",
            "cargo is the rust package manager",
        ];
        let single: Vec<f32> = ce
            .rerank("what is rust", &documents, false, Some(documents.len()))
            .expect("single-batch rerank should succeed")
            .iter()
            .map(|r| r.score)
            .collect();
        let batched = ce
            .rerank("what is rust", &documents, true, Some(2))
            .expect("batched rerank should succeed");

        assert_eq!(batched.len(), documents.len());
        for (idx, result) in batched.iter().enumerate() {
            assert_eq!(result.index, idx);
            assert_eq!(result.document, Some(documents[idx]));
            assert!((result.score - single[idx]).abs() < 1e-4);
        }
    }

    #[test]
    fn rerank_empty_documents_returns_empty() {
        let mut ce = test_encoder();
        let results = ce
            .rerank("what is rust", &[], false, None)
            .expect("empty rerank should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn rerank_reuses_loaded_model_and_tokenizer() {
        let mut ce = test_encoder();
        ce.rerank("first query", &["a document"], false, None)
            .expect("first rerank should succeed");
        ce.rerank("second query", &["another document"], false, None)
            .expect("second rerank should succeed");
    }

    #[test]
    fn rerank_scores_are_valid_probabilities() {
        let mut ce = test_encoder();
        let results = ce
            .rerank("what is rust", &["rust is a language"], false, None)
            .expect("rerank should succeed");

        for result in results {
            assert!((0.0..=1.0).contains(&result.score));
        }
    }

    #[test]
    fn rerank_truncates_documents_longer_than_max_position_embeddings() {
        let mut ce = test_encoder();
        // Long enough that, un-truncated, query + document tokenizes past
        // the model's 512-token position embedding table and ONNX Runtime
        // fails with a broadcast error instead of a document score.
        let long_document = "word ".repeat(3000);
        let docs = [long_document.as_str()];

        let result = ce.rerank("what is rust", &docs, false, None);

        assert!(result.is_ok());
    }

    #[test]
    fn rerank_fails_with_invalid_model_path() {
        let mut ce = CrossEncoder::new(
            "testfiles/tokenizer.json".into(),
            "testfiles/does-not-exist.onnx".into(),
            None,
            None,
            true,
        );
        let result = ce.rerank("query", &["doc"], false, None);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_fails_with_invalid_tokenizer_path() {
        let mut ce = CrossEncoder::new(
            "testfiles/does-not-exist.json".into(),
            "testfiles/model.onnx".into(),
            None,
            None,
            true,
        );
        let result = ce.rerank("query", &["doc"], false, None);
        assert!(result.is_err());
    }
}
