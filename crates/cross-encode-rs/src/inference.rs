use ndarray::Ix2;
use ort::{session::Session, value::TensorRef};
use tokenizers::pipeline::Encoding;

use crate::errors::CrossEncoderError;

/// Runs the ONNX model over a batch of encodings and returns relevance
/// scores in `[0, 1]` (sigmoid for a single-label head, softmax for two).
/// All encodings must already be padded to the same length.
pub fn run_inference(
    session: &mut Session,
    encodings: &[Encoding],
    use_type_ids: bool,
) -> Result<Vec<f32>, CrossEncoderError> {
    let k = encodings.len();
    let padding_dim = encodings[0].len();
    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.ids().iter().map(|i| i.id() as i64))
        .collect();
    // tokenizers v1 returns `None` for an all-ones attention mask (unpadded
    // encoding) and for all-zero type ids, so expand those to full length.
    let mask: Vec<i64> = encodings
        .iter()
        .flat_map(|e| match e.attention_mask() {
            Some(m) => m.iter().map(|&b| b as i64).collect::<Vec<_>>(),
            None => vec![1; e.len()],
        })
        .collect();
    let type_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| match e.type_ids() {
            Some(t) => t.iter().map(|&b| b as i64).collect::<Vec<_>>(),
            None => vec![0; e.len()],
        })
        .collect();

    // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
    let a_ids = TensorRef::from_array_view(([k, padding_dim], &*ids))?;
    let a_mask = TensorRef::from_array_view(([k, padding_dim], &*mask))?;

    let outputs = if use_type_ids {
        let a_type_ids = TensorRef::from_array_view(([k, padding_dim], &*type_ids))?;

        session.run(ort::inputs![a_ids, a_mask, a_type_ids])?
    } else {
        session.run(ort::inputs![a_ids, a_mask])?
    };

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

    use tokenizers::pad_encodings;

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
        let (tokenizer, padding_params, truncation_params) =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let documents = [
            "rust is a language",
            "paris is in france",
            "another document",
        ];
        let mut encodings = encode_batch(
            &tokenizer,
            &truncation_params,
            "what is rust",
            &documents,
        )
        .expect("encoding should succeed");
        pad_encodings(&mut encodings, &padding_params).expect("padding should succeed");
        let mut session = test_session();

        let scores = run_inference(&mut session, &encodings, true)
            .expect("inference should succeed");

        assert_eq!(scores.len(), documents.len());
        for score in scores {
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn run_inference_ranks_relevant_document_higher() {
        let (tokenizer, padding, truncation) =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let documents = [
            "rust is a systems programming language",
            "paris is in france",
        ];
        let mut encodings = encode_batch(
            &tokenizer,
            &truncation,
            "what is rust",
            &documents,
        )
        .expect("encoding should succeed");
        pad_encodings(&mut encodings, &padding).expect("padding should succeed");
        let mut session = test_session();

        let scores = run_inference(&mut session, &encodings, true)
            .expect("inference should succeed");

        assert!(scores[0] > scores[1]);
    }
}
