use ndarray::{Axis, Ix2};
use ort::{session::Session, value::TensorRef};
use tokenizers::Encoding;

use crate::errors::CrossEncoderError;

pub fn run_raw_inference(
    session: &mut Session,
    encodings: Vec<Encoding>,
    k: usize,
) -> Result<(), CrossEncoderError> {
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
    if num_labels == 1 {
        for val in logits.axis_iter(Axis(0)) {}
    } else if num_labels == 2 {
    } else {
        return Err(CrossEncoderError::InferenceError(format!(
            "Unsupported num_labels: {:?}",
            num_labels
        )));
    }

    Ok(())
}
