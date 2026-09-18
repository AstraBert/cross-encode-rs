use std::{error::Error, fmt::Display, io};

use hf_hub::HFError;

#[derive(Debug)]
pub enum CrossEncoderError {
    HuggingFaceLoadError(String),
    IOError(String),
    TokenizerError(String),
    InferenceError(String),
}

impl Display for CrossEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossEncoderError::HuggingFaceLoadError(s) => write!(
                f,
                "Error while loading the model from HuggingFace (or from cache): {}",
                s
            ),
            CrossEncoderError::IOError(s) => write!(f, "IO error: {}", s),
            CrossEncoderError::TokenizerError(s) => {
                write!(f, "Error loading the tokenizer: {}", s)
            }
            CrossEncoderError::InferenceError(s) => {
                write!(f, "Error while running inference: {}", s)
            }
        }
    }
}

impl Error for CrossEncoderError {}

impl From<HFError> for CrossEncoderError {
    fn from(value: HFError) -> Self {
        CrossEncoderError::HuggingFaceLoadError(value.to_string())
    }
}

impl From<io::Error> for CrossEncoderError {
    fn from(value: io::Error) -> Self {
        CrossEncoderError::IOError(value.to_string())
    }
}

impl From<tokenizers::Error> for CrossEncoderError {
    fn from(value: tokenizers::Error) -> Self {
        CrossEncoderError::TokenizerError(value.to_string())
    }
}

impl From<ort::Error> for CrossEncoderError {
    fn from(value: ort::Error) -> Self {
        CrossEncoderError::InferenceError(value.to_string())
    }
}
