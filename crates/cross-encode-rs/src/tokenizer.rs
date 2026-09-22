//! Tokenization utilities for the `statembed` library.
//!
//! This module provides helpers for loading `tokie` from JSON files,
//! encoding text, and extracting vocabulary statistics such as median token length.

use std::path::PathBuf;

use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::errors::CrossEncoderError;

const DEFAULT_MAX_LENGTH: usize = 512;

/// Loads a `Tokenizer` from a JSON file on disk.
///
/// Returns also the ID of the `unk_token` if one is defined in the model config.
///
/// # Arguments
/// * `path` - Path to the `tokenizer.json` file.
/// * `fallback_truncation_length` - truncation length for the tokenizer, in case
///   one isn't already configured
pub fn load_tokenizer(
    path: impl Into<PathBuf>,
    fallback_truncation_length: Option<usize>,
) -> Result<Tokenizer, CrossEncoderError> {
    let p = path.into();

    let mut tokenizer = Tokenizer::from_file(&p)?;
    // `run_inference` stacks encodings into a single [batch, seq_len] tensor, so every
    // encoding in a batch must share the same length regardless of the tokenizer.json config.
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    if tokenizer.get_truncation().is_none() {
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: fallback_truncation_length.unwrap_or(DEFAULT_MAX_LENGTH),
                ..Default::default()
            }))
            .map_err(|e| CrossEncoderError::TokenizerError(e.to_string()))?;
    }

    Ok(tokenizer)
}

pub fn encode_batch(
    tk: &Tokenizer,
    query: &str,
    documents: &[&str],
) -> Result<Vec<Encoding>, CrossEncoderError> {
    let encodings: Vec<Encoding> = tk.encode_batch(
        documents
            .iter()
            .map(|d| (query, *d))
            .collect::<Vec<(&str, &str)>>(),
        true,
    )?;
    Ok(encodings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tokenizer() {
        let _ = load_tokenizer("testfiles/tokenizer.json", None).expect("Should not fail");
    }

    #[test]
    fn encode_batch_truncates_to_default_max_length() {
        let tk = load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let long_document = "word ".repeat(3000);

        let encodings =
            encode_batch(&tk, "what is rust", &[long_document.as_str()]).expect("should encode");

        assert_eq!(encodings[0].len(), DEFAULT_MAX_LENGTH);
    }
}
