cp ~/.cross-encode-rs/cross-encoder--ms-marco-MiniLM-L6-v2/onnx/model.onnx .
cp ~/.cross-encode-rs/cross-encoder--ms-marco-MiniLM-L6-v2/tokenizer.json .

kubench image . --name cross-encoder-server --tag latest --registry clelias-dell-pro.olm-gecko.ts.net:5000

rm -rf model.onnx
rm -rf tokenizer.json
