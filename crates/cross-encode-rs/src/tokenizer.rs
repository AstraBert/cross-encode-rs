//! Tokenization utilities for the `statembed` library.
//!
//! This module provides helpers for loading `tokie` from JSON files,
//! encoding text, and extracting vocabulary statistics such as median token length.

use std::path::PathBuf;

use tokenizers::{Encoding, Tokenizer};

use crate::errors::CrossEncoderError;

/// Loads a `Tokenizer` from a JSON file on disk.
///
/// Returns also the ID of the `unk_token` if one is defined in the model config.
///
/// # Arguments
/// * `path` - Path to the `tokenizer.json` file.
pub fn load_tokenizer(path: impl Into<PathBuf>) -> Result<Tokenizer, CrossEncoderError> {
    let p = path.into();

    let tokenizer = Tokenizer::from_file(&p)?;

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
        let _ = load_tokenizer("testfiles/tokenizer.json").expect("Should not fail");
    }
}
