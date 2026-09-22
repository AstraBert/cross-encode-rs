use std::path::PathBuf;

use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

use crate::{
    errors::CrossEncoderError,
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

#[derive(Debug)]
pub struct CrossEncoder {
    pub tokenizer_path: PathBuf,
    pub model_path: PathBuf,
    pub intra_threads: Option<usize>,
    model: Option<Session>,
    tokenizer: Option<Tokenizer>,
}

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
    pub fn new(tokenizer_path: PathBuf, model_path: PathBuf, intra_threads: Option<usize>) -> Self {
        Self {
            model_path,
            tokenizer_path,
            intra_threads,
            tokenizer: None,
            model: None,
        }
    }

    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(
        model_id: &str,
        force_download: bool,
        intra_threads: Option<usize>,
    ) -> Result<Self, CrossEncoderError> {
        let (model_path, tokenizer_path) = download_from_hub(model_id, force_download).await?;

        Ok(Self {
            model_path,
            tokenizer_path,
            intra_threads,
            model: None,
            tokenizer: None,
        })
    }

    fn init_model(&mut self) -> Result<(), CrossEncoderError> {
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

    fn init_tokenizer(&mut self) -> Result<(), CrossEncoderError> {
        if self.tokenizer.is_some() {
            return Ok(());
        }

        self.tokenizer = Some(load_tokenizer(&self.tokenizer_path)?);
        Ok(())
    }

    pub fn rerank<'a>(
        &'a mut self,
        query: &str,
        documents: &[&'a str],
        with_documents: bool,
    ) -> Result<Vec<RerankResult<'a>>, CrossEncoderError> {
        self.init_model()?;
        self.init_tokenizer()?;
        if let Some(ref tokenizer) = self.tokenizer
            && let Some(ref mut model) = self.model
        {
            let encodings = encode_batch(tokenizer, query, documents)?;
            let scores = run_inference(model, encodings, documents.len())?;
            let mut results: Vec<RerankResult> = Vec::with_capacity(scores.len());
            for (idx, score) in scores.iter().enumerate() {
                results.push(RerankResult {
                    index: idx,
                    score: *score,
                    document: {
                        if with_documents {
                            Some(documents[idx])
                        } else {
                            None
                        }
                    },
                })
            }
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
            .rerank("what is rust", &documents, true)
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
            .rerank("what is rust", &documents, false)
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
            .rerank("what is rust", &documents, true)
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
    fn rerank_reuses_loaded_model_and_tokenizer() {
        let mut ce = test_encoder();
        ce.rerank("first query", &["a document"], false)
            .expect("first rerank should succeed");
        ce.rerank("second query", &["another document"], false)
            .expect("second rerank should succeed");
    }

    #[test]
    fn rerank_scores_are_valid_probabilities() {
        let mut ce = test_encoder();
        let results = ce
            .rerank("what is rust", &["rust is a language"], false)
            .expect("rerank should succeed");

        for result in results {
            assert!((0.0..=1.0).contains(&result.score));
        }
    }

    #[test]
    fn rerank_fails_with_invalid_model_path() {
        let mut ce = CrossEncoder::new(
            "testfiles/tokenizer.json".into(),
            "testfiles/does-not-exist.onnx".into(),
            None,
        );
        let result = ce.rerank("query", &["doc"], false);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_fails_with_invalid_tokenizer_path() {
        let mut ce = CrossEncoder::new(
            "testfiles/does-not-exist.json".into(),
            "testfiles/model.onnx".into(),
            None,
        );
        let result = ce.rerank("query", &["doc"], false);
        assert!(result.is_err());
    }
}
