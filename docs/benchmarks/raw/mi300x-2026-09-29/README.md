# Raw evidence — MI300X run 3, 2026-09-29

Verbatim capture, no interpretation. One run, one droplet:

| Directory | Report | Read-out | Runsheet |
|---|---|---|---|
| `run3/` | `docs/benchmarks/mi300x-run3.md` | `bench/measured_mi300x_run3.py` | `docs/benchmarks/runsheets/mi300x-run-3.md` |

## Run 3

| Path | What it is |
|---|---|
| `run3/engine-solo.log` | the single engine at 0.90, block A's control: startup log (checkpoint A) and every 10-second `loggers.py` line until it was stopped |
| `run3/engine-8000.log`, `run3/engine-8001.log` | the pair at 0.45 each, started serially, 8001 only once 8000 answered `/health`; up from block A to the harvest |
| `run3/results/solo/` | `fleet-bill` against the solo engine: `fleet-c001-h00` (the droplet check) and `fleet-c064-h00` |
| `run3/results/pair-8000/`, `run3/results/pair-8001/` | `fleet-bill-half` against each engine of the pair, the two harness calls started together |
| `run3/results/<arm>-n<N>/` | one router-arm level, `<arm>` `prefix` or `round_robin`, N 031 … 511 |
| `run3/results/*/<level>.json` | the harness's per-level summary, in `vllm bench serve`'s key names plus `harness_*` |
| `run3/results/*/<level>-metrics.json` | Prometheus counter increments over the level, summed over every `--metrics-endpoint` |
| `run3/results/*/<level>-requests.jsonl` | one record per measured request, warmup excluded: TTFT, every ITL, `policy` and `upstream` from the router's headers |
| `run3/results/*/run.json` | each harness call: argv, startup-log facts, the pool gate, the seat verdict |
| `run3/solo-harness.txt`, `run3/pair-800x-harness.txt`, `run3/arm-<arm>-n<N>.txt` | what each harness call printed |
| `run3/router-<arm>-n<N>.log` | `docker logs` of the router container that served that level, taken after it |
| `run3/arm-progress` | one `ARMDONE <arm> <N> <exit code>` per level, in the order they ran |
| `run3/metrics-after-8000.log`, `run3/metrics-after-8001.log` | each engine's full `/metrics` after the last level |
| `run3/host-under-load.txt` | CPU model, `nproc`, kernel, amdgpu version, and three `rocm-smi` samples taken while block A's first levels ran |
| `run3/rocm-smi-end.txt` | VRAM and KFD processes at the harvest, both engines still up |
| `run3/vllm-version.txt`, `run3/pull.txt`, `run3/hf-download.txt` | `vllm --version`, the image digest, the weights' snapshot path |
| `run3/s1.sh`, `run3/s2.sh`, `run3/s3.sh`, `run3/hostrec.sh`, `run3/arm.sh` | the scripts that ran each stage; the runsheet's commands, put into files so a stage survives an SSH drop inside `tmux` |
| `run3/summary.py` | the table printed on the droplet while block B ran; `bench/measured_mi300x_run3.py` supersedes it |
| `run3/stage`, `run3/*.pid` | the scripts' handshakes: the last stage marker and the engines' PIDs |

## What differs from the runsheet's commands

- The engines were started from `s1.sh` / `s2.sh` in the background, with output
  redirected to the log, not in the foreground through `tee`. The container ran
  in `tmux` as the sheet says, and the host's commands in a second window.
- `fleet-bill` ran with `--accelerator mi300x-run2` before the droplet check had
  been read. The flag moves only the predicted columns and the prefill-floor
  flag, never a measured figure; the check then chose `mi300x-run2`.
- `arm.sh` adds `docker logs router` after each level and an `ARMDONE` line.
- The two block-A pair calls overlapped for 41 of 49 s: engine 8001's call
  finished first, and 8000 ran its last 8 s alone.
