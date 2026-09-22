use ndarray::Ix2;
use ort::{session::Session, value::TensorRef};
use tokenizers::Encoding;

use crate::errors::CrossEncoderError;

/// Runs the ONNX model over a batch of encodings and returns relevance
/// scores in `[0, 1]` (sigmoid for a single-label head, softmax for two).
pub fn run_inference(
    session: &mut Session,
    encodings: Vec<Encoding>,
    k: usize,
) -> Result<Vec<f32>, CrossEncoderError> {
    let padding_dim = encodings[0].len();
    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
        .collect();
    let mask: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
        .collect();
    let type_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
        .collect();

    // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
    let a_ids = TensorRef::from_array_view(([k, padding_dim], &*ids))?;
    let a_mask = TensorRef::from_array_view(([k, padding_dim], &*mask))?;
    let a_type_ids = TensorRef::from_array_view(([k, padding_dim], &*type_ids))?;

    let outputs = session.run(ort::inputs![a_ids, a_mask, a_type_ids])?;

    let logits = outputs[0]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix2>()
        .unwrap();

    let (_, num_labels) = logits.dim();
    let scores: Vec<f32> = if num_labels == 1 {
        let s = logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        s.iter().copied().collect()
    } else if num_labels == 2 {
        let mut s = Vec::with_capacity(logits.dim().0);
        for row in logits.rows() {
            let max = row.iter().cloned().fold(f32::MIN, f32::max); // stability trick
            let exp0 = (row[0] - max).exp();
            let exp1 = (row[1] - max).exp();
            let sum = exp0 + exp1;
            s.push(exp1 / sum); // probability of the "relevant" class
        }
        s
    } else {
        return Err(CrossEncoderError::InferenceError(format!(
            "Unsupported num_labels: {:?}",
            num_labels
        )));
    };

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use ort::session::{Session, builder::GraphOptimizationLevel};

    use crate::tokenizer::{encode_batch, load_tokenizer};

    use super::*;

    fn test_session() -> Session {
        Session::builder()
            .expect("builder should not fail")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("optimization level should be valid")
            .commit_from_file("testfiles/model.onnx")
            .expect("model file should exist and load")
    }

    #[test]
    fn run_inference_returns_one_score_per_document() {
        let tokenizer =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let documents = [
            "rust is a language",
            "paris is in france",
            "another document",
        ];
        let encodings =
            encode_batch(&tokenizer, "what is rust", &documents).expect("encoding should succeed");
        let mut session = test_session();

        let scores = run_inference(&mut session, encodings, documents.len())
            .expect("inference should succeed");

        assert_eq!(scores.len(), documents.len());
        for score in scores {
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn run_inference_ranks_relevant_document_higher() {
        let tokenizer =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let documents = [
            "rust is a systems programming language",
            "paris is in france",
        ];
        let encodings =
            encode_batch(&tokenizer, "what is rust", &documents).expect("encoding should succeed");
        let mut session = test_session();

        let scores = run_inference(&mut session, encodings, documents.len())
            .expect("inference should succeed");

        assert!(scores[0] > scores[1]);
    }
}
