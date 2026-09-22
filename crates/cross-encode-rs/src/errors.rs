use std::{error::Error, fmt::Display, io};

#[cfg(feature = "hf-hub")]
use hf_hub::HFError;
use ort::session::builder::SessionBuilder;

/// Errors returned by this crate.
#[derive(Debug)]
pub enum CrossEncoderError {
    HuggingFaceLoadError(String),
    IOError(String),
    TokenizerError(String),
    InferenceError(String),
    OnnxLoadingError(String),
    GenericFailure(String),
}

impl Display for CrossEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFaceLoadError(s) => write!(
                f,
                "Error while loading the model from HuggingFace (or from cache): {}",
                s
            ),
            Self::IOError(s) => write!(f, "IO error: {}", s),
            Self::TokenizerError(s) => {
                write!(f, "Error loading the tokenizer: {}", s)
            }
            Self::InferenceError(s) => {
                write!(f, "Error while running inference: {}", s)
            }
            Self::OnnxLoadingError(s) => {
                write!(f, "Error while loading ONNX model: {}", s)
            }
            Self::GenericFailure(s) => {
                write!(f, "{}", s)
            }
        }
    }
}

impl Error for CrossEncoderError {}

impl From<&str> for CrossEncoderError {
    fn from(value: &str) -> Self {
        Self::GenericFailure(value.to_string())
    }
}

#[cfg(feature = "hf-hub")]
impl From<HFError> for CrossEncoderError {
    fn from(value: HFError) -> Self {
        Self::HuggingFaceLoadError(value.to_string())
    }
}

impl From<io::Error> for CrossEncoderError {
    fn from(value: io::Error) -> Self {
        Self::IOError(value.to_string())
    }
}

impl From<tokenizers::Error> for CrossEncoderError {
    fn from(value: tokenizers::Error) -> Self {
        Self::TokenizerError(value.to_string())
    }
}

impl From<ort::Error> for CrossEncoderError {
    fn from(value: ort::Error) -> Self {
        Self::InferenceError(value.to_string())
    }
}

impl From<ort::Error<SessionBuilder>> for CrossEncoderError {
    fn from(value: ort::Error<SessionBuilder>) -> Self {
        Self::OnnxLoadingError(value.to_string())
    }
}
