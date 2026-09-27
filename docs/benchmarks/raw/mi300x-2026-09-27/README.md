# Raw evidence — MI300X run 1, 2026-09-27

Verbatim capture, no interpretation. The analysis lives in
`docs/benchmarks/mi300x-run1.md`; the arithmetic that reads these files is
`bench/measured_mi300x_run1.py`; the before-the-fact predictions are
`docs/benchmarks/runsheets/mi300x-run-1.md`.

| Path | What it is |
|---|---|
| `run1/sweep.log` | everything the sweep printed, both launches: the server's startup log (checkpoint A), every benchmark's console output, and the engine's 10-second `loggers.py` lines with Running / Waiting / KV usage. The second launch starts at the line `===== RESUME … =====` |
| `run1/metrics-after.log` | the after-bench hook: `num_preemptions_total`, `num_requests_waiting`, `num_requests_running` from `/metrics`, once after each of the 28 repeats, `--` between reads |
| `run1/results/mi300x-run-1/<row>/run=N.json` | one per repeat, written by `vllm bench serve` |
| `run1/results/mi300x-run-1/<row>/summary.json` | the sweep's per-row aggregate, **rewritten by the second launch from `run=0` and `run=1` only** — see below |
| `run1/results/mi300x-run-1/summary.csv` | the sweep's table, **24 of the 28 repeats** — see below |
| `run1/mi300x-run-1-serve.json`, `run1/mi300x-run-1-bench.json` | the parameter files exactly as copied to the droplet |
| `run1/sweep.sh` | the first launch, the runsheet's §3 line verbatim |
| `run1/sweep-resume.sh` | the second launch: the same line with `--num-runs 2 --resume` and `tee -a` |

## Two launches, and why the repeat counts differ

The first launch ran three repeats per row. At 10:19 UTC the pace
(~0.39 s per prompt once saturated) put the end of three repeats past the
runsheet's two-hour mark, so its drop order step 1 was taken early: Ctrl-C
seconds after `c128/run=1.json` was written (its `date` is 10:22:34) and its hook
read, then the second launch at 10:23:12 with `--num-runs 2 --resume`. It found
the ten repeats it asked for among c001–c128 — `run=0` and `run=1` of each — by
their `run=N.json` files, skipped them, and ran everything after.

| Rows | Repeats | Launch |
|---|---|---|
| c001, c008, c032, c064 | 3 | 1 |
| c128 | 2 | 1 |
| c192 … c288, c001-in2000, c001-in8000 | 2 | 2 |

`summary.csv` and every `summary.json` were written by the second launch, which
asked for two repeats and so folded `run=0` and `run=1` of each row: both lack
`run=2` of c001, c008, c032 and c064. The `run=N.json` files are the
measurement; `bench/measured_mi300x_run1.py` reads them directly and prints
every repeat.

`tee -a` in the second launch is not in the runsheet: its line uses `tee`, and
rerunning it after an interruption would have truncated `sweep.log`, taking
checkpoint A's startup log with it.

## Provenance, and what is missing

- AMD Developer Cloud GPU droplet `7.14-gpu-mi300x1-192gb-devcloud-atl1`, ATL1,
  on-demand at $1.990/h, image *Quick Start → ROCm Software*, version **7.14**
  selected (the dropdown defaulted to 10.0), Ubuntu 24.04. From the operator's
  screenshot of the create screen that day, not committed: the *Quick Start* tab
  offers *ROCm Software* above a list of *Quick Start Packages*, no item is named
  *Vanilla ROCm*, and the *vLLM* package reads "vLLM 0.27.1 … on an AMD ROCm 7.14
  host". Destroyed after the
  harvest; archive checksum matched on both ends before it was deleted.
- `rocm-smi`: one *AMD Instinct MI300X VF*, gfx942, 205 822 885 888 B of VRAM
  (191.69 GiB), 0.30 GB in use at idle with no KFD process. Docker 29.7.2 and
  tmux were preinstalled.
- **The plain ROCm image was not plain.** A container named `rocm` (Jupyter Lab
  on `0.0.0.0:8888`, with `/dev/kfd` and `/dev/dri`) was running at boot, and
  `rocm/device-metrics-exporter:v1.5.0` existed in state *Created*. Neither held
  the GPU; `rocm` was stopped with `docker stop` before the run.
- Image `vllm/vllm-openai-rocm:v0.27.1` (`0.27.1+rocm723`), pulled in ~2 min;
  `Qwen/Qwen3-8B` downloaded in 12 s, unauthenticated.
- **No in-run gauge sampler.** The engine's own 10-second log lines stand in for
  one: Running, Waiting and KV usage are a snapshot every 10 s, so their maximum
  is a floor on the peak; the two throughputs are 10-second averages.
  `metrics-after.log` reads after each repeat has drained, so its
  `num_requests_*` lines are all zero by construction; only the preemption
  counter in it carries information.
- **`block_size` is not in the log** on this build: vLLM prints it only when it
  differs from the default 16, and 16 is what `ROCM_ATTN` takes.
