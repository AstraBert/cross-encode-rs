# load-test

Go load-test client for `crates/server`, plus Python reference servers for comparing inference performance.

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

Runs a warmup (100 requests), then `<turns>` repeats at 1000 and 10,000 requests each, appending to `results/<name>/<i>.txt`:

```bash
ENDPOINT=http://127.0.0.1:7432/rerank ./send_requests.sh 3 cross-encode-rs
```

## Python reference servers

Same request/response shape as `crates/server`, same model, so the comparison isolates the serving layer:

- `../scripts/serve-python.py` — sentence-transformers, ONNX Runtime backend.
- `../scripts/serve-fastembed.py` — fastembed.

```bash
./scripts/serve-python.py --port 7433
# or
./scripts/serve-fastembed.py --port 7433
```

## Results

`results/<name>/<i>.txt` holds repeated `send_requests.sh` runs per system (`cross-encode-rs`, `fastembed`, `sentence-transformers`). Render them into an HTML report with `../scripts/render-benchmark-charts.py`.
