use std::{path::PathBuf, sync::OnceLock};

use hf_hub::{HFClient, RepoTypeModel};

use crate::errors::CrossEncoderError;

/// Files pulled from a Hugging Face model repo, model then tokenizer.
pub const DOWNLOAD_FILES: &[&str] = &["onnx/model.onnx", "tokenizer.json"];
pub static HF_CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Local cache directory for downloaded models, `~/.cross-encode-rs`.
pub fn hf_cache_dir() -> &'static PathBuf {
    HF_CACHE_DIR.get_or_init(|| {
        dirs::home_dir()
            .expect("No home dir could be found for the current environment")
            .join(".cross-encode-rs")
    })
}

/// Downloads a model's ONNX weights and tokenizer from the Hugging Face Hub
/// into the local cache, skipping files already cached unless
/// `force_download` is set. Returns `(model_path, tokenizer_path)`.
pub async fn download_from_hub(
    model_id: &str,
    force_download: bool,
) -> Result<(PathBuf, PathBuf), CrossEncoderError> {
    let client = HFClient::new()?;
    let split = model_id.split_once("/");

    if let Some((owner, name)) = split {
        let repo = client.repository(RepoTypeModel, owner, name);
        let base_path = hf_cache_dir().join(model_id.replace("/", "--"));
        for f in DOWNLOAD_FILES {
            // skip downloading if already there, unless we want to forcibly re-download
            if base_path.join(f).exists() && !force_download {
                continue;
            }
            repo.download_file()
                .filename(f.to_string())
                .local_dir(&base_path)
                .send()
                .await?;
        }
        return Ok((
            base_path.join(DOWNLOAD_FILES[0]),
            base_path.join(DOWNLOAD_FILES[1]),
        ));
    }

    Err(CrossEncoderError::HuggingFaceLoadError(
        "Model ID should be reported as owner/repo_name".to_string(),
    ))
}
