//! Tokenizer loading and query/document batch encoding.

use std::path::PathBuf;

use tokenizers::{
    PaddingParams, PaddingStrategy, TruncationParams,
    convert::canonicalize_file,
    from_json, from_json_file,
    pipeline::{EncodeHandle, EncodeOptions, Encoding, PipelineTokenizer},
};

use crate::errors::CrossEncoderError;

const DEFAULT_MAX_LENGTH: usize = 512;

/// Loads a tokenizer from a `tokenizer.json` file, forcing batch-longest
/// padding and, unless the file already configures its own, truncation at
/// `fallback_truncation_length` (default 512).
pub fn load_tokenizer(
    path: impl Into<PathBuf>,
    fallback_truncation_length: Option<usize>,
) -> Result<(PipelineTokenizer, PaddingParams, TruncationParams), CrossEncoderError> {
    let p = path.into();

    let res = from_json_file(&p);

    let tokenizer = match res {
        Ok(t) => t,
        Err(e) => {
            if e.to_string()
                .contains("tokenizer version '1.0' is not `2.0`")
            {
                let converted = canonicalize_file(&p)?;

                from_json(&converted)?
            } else {
                return Err(e.into());
            }
        }
    };

    let padding_params = tokenizer
        .get_padding()
        .unwrap_or(&PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        })
        .to_owned();
    let truncation_params = tokenizer
        .get_truncation()
        .unwrap_or(&TruncationParams {
            max_length: fallback_truncation_length.unwrap_or(DEFAULT_MAX_LENGTH),
            ..Default::default()
        })
        .to_owned();

    Ok((tokenizer, padding_params, truncation_params))
}

/// Encodes each `(query, document)` pair for cross-encoder input, truncated
/// but unpadded: callers pad each inference batch with `pad_encodings`.
pub fn encode_batch(
    tk: &PipelineTokenizer,
    truncation: &TruncationParams,
    query: &str,
    documents: &[&str],
) -> Result<Vec<Encoding>, CrossEncoderError> {
    let handle: EncodeHandle = tk.encode(
        documents
            .iter()
            .map(|d| (query, *d))
            .collect::<Vec<(&str, &str)>>()
            .as_slice(),
        &EncodeOptions {
            add_special_tokens: true,
            encode_special_tokens: true,
            padding: tokenizers::pipeline::Override::Off,
            truncation: tokenizers::pipeline::Override::With(truncation.to_owned()),
        },
    );
    let encodings: Vec<Encoding> = handle.wait()?;
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
        let (tk, _, tr) =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let long_document = "word ".repeat(3000);

        let encodings = encode_batch(&tk, &tr, "what is rust", &[long_document.as_str()])
            .expect("should encode");

        assert_eq!(encodings[0].len(), DEFAULT_MAX_LENGTH);
    }
}
