# load-test

Go load-test client for `crates/server`, plus a Python reference server for comparing inference performance.

## Client (`main.go`)

Fires `N` concurrent `POST /rerank` requests at a target endpoint (capped at 1000 in-flight), prints success rate, average response time, and requests/sec.

```bash
go run main.go <num_requests> <endpoint>
# e.g.
go run main.go 10000 http://127.0.0.1:7432/rerank
```

### Failures

Printed as an aggregated summary at the end:

```
Failure reasons:
  49323  http 502:
```

- `connection error: ...` — request never reached the server (refused, reset, timeout, fd exhaustion). Won't show up in server logs.
- `http <status>: <body>` — reached the server, got an error response back. The body is the actual error message from `crates/server`.

### "too many open files"

At concurrency 1000, raise the fd limit on both client and server shells:

```bash
ulimit -n 65536
```

If `ulimit -Hn` is also too low: `sudo launchctl limit maxfiles 65536 200000`, then open a new terminal.

## `send_requests.sh`

Runs the client at 1000 and 10,000 requests:

```bash
ENDPOINT=http://127.0.0.1:7432/rerank ./send_requests.sh
```

## Python reference server (`../scripts/serve-python.py`)

Same request/response shape as `crates/server`, same model (`cross-encoder/ms-marco-TinyBERT-L2-v2`, sigmoid-activated), ONNX Runtime backend — isolates the serving layer instead of comparing PyTorch vs ONNX Runtime.

```bash
./scripts/serve-python.py --port 7433
```

## Results

`results/cross-encode-rs.txt` and `results/python.txt` hold 1000/10,000-request runs.
