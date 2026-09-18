use std::time::Instant;

use cross_encode_rs::tokenizer::{encode_batch, load_tokenizer};

fn main() -> Result<(), cross_encode_rs::errors::CrossEncoderError> {
    let docs = vec!["this is a document"; 1_000];
    let start_time = Instant::now();
    let tk = load_tokenizer(
        "/home/clelia/.cross-encode-rs/cross-encoder--ms-marco-MiniLM-L6-v2/tokenizer.json",
    )?;
    let _ = encode_batch(&tk, "is this a query?", &docs);
    let elapsed = start_time.elapsed().as_millis();

    println!("{:?}ms", elapsed);

    Ok(())
}
