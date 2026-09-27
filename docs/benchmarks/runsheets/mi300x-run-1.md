# Runsheet — MI300X run 1: calibrating the card

Runs 1–3 (2026-08-18 … 08-30) were taken on an L40S, and every coefficient in
`docs/SLO.md` is that card's. This run is the first on the card the document was
derived for, and it does for the MI300X what run 1 did for the L40S and nothing
more: **read the KV pool from the startup log, fit `eff_mem` from the decode
step, fit `mfu` from an uncontended prefill, and find where the seat curve
ends.** Per `docs/SLO.md` §9 it is reported against the **uncalibrated 0.70 /
0.45**, and the fitted figures become a hypothesis for MI300X run 2.

**Status: reviewed 2026-09-07**, section by section, in two passes — the
numbers against `bench/predictions.py` tables 9–10 (all reproduced), and every
external claim against the `v0.27.1` sweep source, Docker Hub, DigitalOcean's
billing rules and AMD's getting-started guide. Twelve edits came out of it, one
to the launch line (`--server-ready-timeout`) and one to a formula (the `eff_mem`
read). **Re-checked 2026-09-27** against the `v0.27.1` source: `max_num_seqs`
defaults to **1024**, not 256, on any card of 70 GiB or more
(`vllm/engine/arg_utils.py`, `get_batch_defaults`), so it is now pinned in the
serve params; `hf` and `curl` are on the ROCm image (`docker/Dockerfile.rocm`);
the attention backend should be `ROCM_ATTN` with block size 16
(`vllm/platforms/rocm.py`), which the log confirms. Still open until the day:
the console price, whether Vanilla ROCm ships Docker and tmux, and how long a
cold launch takes on ROCm.

Written 2026-09-04, before the credits were activated and therefore before any
card exists — which is the point. **Every predicted figure below comes from
`bench/predictions.py` tables 9 and 10**, added the same day; re-run it if the
module has changed since. Where this sheet disagrees with a measured number,
`docs/benchmarks/` wins.

**The instrument is `vllm bench sweep serve`**, not the hand ladder of runs 1–2
and not `bench/harness.py`: one server launch, twelve benchmark rows, three
repeats each, machine-shaped JSON. The decision and its limits are in
`docs/instrument-vllm-bench-sweep.md`; the two rules that outrank this sheet on
the day are
`vllm bench sweep serve --help=all` and the server's startup log.

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X at **$1.99/h**
(Phoronix's review of the service, 2025; **confirmed on the console 2026-09-27**, on-demand; the console price at droplet creation
wins and is written here when read). The $100 of credits are **~50 h** behind a
**short, fixed expiry** — a date on the calendar, not a budget that can be
spread — and that is the whole reason this sheet, the harness and the drop
order are final before a droplet exists: the hours go into the run, not into
debugging. The credit is DigitalOcean account credit applied to the next
invoice and, per the console, **not to Add-Ons**; a GPU droplet is not an
Add-On, so the run is covered, but the account needs a payment method for the
invoice the credit then settles. Session budget **≤ 2.5 h ≈ $5**, hard stop;
expected clock ≈ 1.5 h (§5). Two facts that set the shape of the day, both
checked 2026-09-04 and neither in the repository before:

- **The service is DigitalOcean GPU Droplets under an AMD front door.** The same
  card is $2.59/h on DigitalOcean's own price list — a different contract, not
  a different card — so `MI300X_HOURLY = 2.00` in `bench/predictions.py` stands
  for the credits and would be wrong for a paid droplet.
- **`vllm/vllm-openai-rocm:v0.27.1` exists** on Docker Hub (11.3 GB, amd64,
  built 2026-08-11), the same tag runs 1–3 used. AMD's own `rocm/vllm` images
  are deprecated in favour of it and carry other vLLM versions. Same tag, same
  `bench sweep` code, same defaults — the instrument note holds without a
  re-check. What differs is the backend under it, and §2 reads that off the log.

---

## What the L40S runs established, and which of it transfers

Nothing numerical transfers; three shapes do, and each is a prediction here.

| From the L40S | Status on the MI300X |
|---|---|
| `eff_mem` 0.83, `mfu` 0.439 | **Do not transfer** (`docs/SLO.md` §9). This run fits both from scratch against 0.70 / 0.45. |
| The derived pool is **7.6 % above** the logged one — non-torch memory, activation peak, graph pool | A hypothesis: table 9 applies 7 % and predicts ~986 000 tokens against 1 060 655 derived. The log decides. |
| TPOT p50 is up to **2.06 ×** the median ITL at the top of the curve — prefill interference | Engine behaviour, so it may transfer in shape; its size should shrink where prefill is 3.6 × faster (47 ms against 171 ms per 2 000 tokens). Read, not predicted. |
| Latency ends the curve at 23 seats, half the pool | **Inverted here**: the latency limit is 286 and the corrected pool seats ~234, so capacity ends the curve with every level inside 50 ms. This is the sentence the run exists to test. |
| `max_num_seqs` 256 is never reached | It is the **third limit** (§10) and it sits *between* the pool and the latency limit: c = 288 should queue, not preempt. |

---

## 0 · Before the card (no credits spent)

- [x] `bench/sweep/dry-run.sh mi300x-run-1-serve.json mi300x-run-1-bench.json`
      — 36 benchmark commands, one server launch, exit 0 (2026-09-04).
- [x] `python3 bench/predictions.py` — tables 9 and 10 open beside the terminal.
- [x] **Activation is not a lever.** This sheet was written expecting to start
      the window itself, on the morning of the run. The credits arrived already
      active, so the expiry is a date the run has to land *before* rather than
      one it chooses — the operator tracks the date. What it changes here is
      nothing, because §0 already said it: the sheet and the harness are final
      before the droplet.
- [x] Read the hourly price off the console and write it into the header above
      — **$1.99/h**, on-demand, 1 × MI300X (2026-09-27).
- [x] SSH key registered with the service before the droplet is created.
- [ ] **Image: Vanilla ROCm**, not a Quick Start image. The Quick Start images
      boot with a container named `rocm` already running AMD's own vLLM build
      — a second vLLM on the card is VRAM noise and a version confusion, and
      the run is pinned to `v0.27.1` in its own container (verified
      2026-09-07 against AMD's getting-started guide).

---

## 1 · Droplet up (~20 min, mostly download)

A DigitalOcean GPU droplet is a VM with a public address and direct SSH — none
of run 2's proxy trouble; `scp` works from the first minute. Two decisions that
cannot be changed afterwards:

1. **One card, not eight.** The 8 × MI300X droplet is $15.92/h and answers a
   question this run does not ask. The open question about replicas
   duplicating the weights read needs two cards and is its own decision.
2. **Destroy, never stop.** A stopped droplet is billed; only a destroyed one is
   not. Everything the run produces therefore lives under `/workspace` on the
   droplet's disk *and* is copied off before §4 destroys it.

On the droplet, in this order, each a gate for the next:

```
rocm-smi --showproductname --showmeminfo vram      # one MI300X, 192 GB visible
docker --version                                    # gate: Docker on Vanilla ROCm is not documented
which tmux || apt-get install -y tmux               # the sweep outlives an SSH drop only inside tmux
mkdir -p /workspace/hf /workspace/run1
docker pull vllm/vllm-openai-rocm:v0.27.1           # 11.3 GB
```

From the laptop, while the image pulls:

```
scp bench/sweep/mi300x-run-1-*.json root@<ip>:/workspace/run1/
```

Then the container. **vLLM is not PID 1**: the sweep launches the server itself
(mechanic 1), so the container runs a shell and everything happens inside it.
The device and IPC flags are vLLM's own ROCm instructions for `v0.27.1`,
and the whole thing runs inside `tmux new -s run1` (reattach with
`tmux attach -t run1`):

```
docker run -it --rm --name run1 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v /workspace:/workspace -v /workspace/hf:/root/.cache/huggingface \
  --entrypoint bash vllm/vllm-openai-rocm:v0.27.1
```

Inside it, before anything else — **the help wins over the instrument note**:

```
vllm --version                                      # 0.27.1 + the ROCm suffix
vllm bench sweep serve --help=all | grep -E 'show-stdout|after-bench|link-vars|num-runs|resume'
hf download Qwen/Qwen3-8B                           # 16.4 GB, once; the cache is on the host
which curl                                          # the after-bench hook needs it; Dockerfile.rocm installs it
```

**No published HTTP port and no API key**: the server binds `127.0.0.1` and the
load runs in the same container, as in every earlier run. The droplet exposes
SSH and nothing else.

---

## 2 · Checkpoint A — the startup log against the arithmetic

The sweep launches the server and, once `/health` answers, fires the first
benchmark row; the first row is c = 1 with 20 prompts and costs nothing, so
**checkpoint A is read from the log while row 1 runs**, and a failed checkpoint
means Ctrl-C (the sweep SIGKILLs the server's process group), fix, relaunch
without `--resume`. The log exists only because of `--show-stdout | tee`
(mechanic 7) — check that the file is growing before reading anything else.

```
grep -E 'KV cache size|kv cache memory in use|Maximum concurrency|max_num_batched_tokens|block_size|backend|prefix_caching|Graph capturing|non-torch|activation|dtype' /workspace/run1/sweep.log
```

| Log line | Predicted | Source | On a miss |
|---|---|---|---|
| `GPU KV cache size` | **1 060 655** tokens derived; **~986 000** with the L40S's 7 % shortfall | table 9 | Either figure within 3 % is a pass and the *other* one is the finding. Outside both: read the `non-torch` / `activation` lines and account for it before the sweep continues. **Why 3 % and not §9's 5 %:** at ±5 % the two bands overlap (1.008–1.036 M) and one log value would confirm both; at ±3 % they do not touch. The price is a gap — 1.016–1.029 M confirms *neither*, and with the 4.2 % launch-to-launch spread run 2 saw, one launch can land there. That is a recorded outcome, not a failure |
| `Maximum concurrency … at max_model_len 9000` | **117.9 ×** derived, **~109.6 ×** corrected | table 9 | Same rule; it is the pool divided by 9 000 |
| dtype | BF16 from `config.json` | `docs/SLO.md` §3 | Anything else invalidates every byte count |
| `max_num_batched_tokens` | **2 048**, set explicitly | runs 1–3 geometry | The flag is in the serve command; a different value means the override did not apply |
| attention backend | **`ROCM_ATTN`** by the source's priority order | `vllm/platforms/rocm.py` | Not a gate. Whatever it is, it is the backend `eff_mem` gets fitted against, and the write-up names it the way run 1 named FlashAttention 2 |
| `block_size` | **16** | `CacheConfig`, `ROCM_ATTN` | Not a gate: nothing in tables 9–10 depends on it. Record it |
| `max_num_seqs` | **256**, set explicitly — the default on a card this size is 1024 | serve params | Not a log fact on this build (run 3 §3); read it from the command, never from the log |

---

## 3 · The sweep — twelve rows, one server (~45–85 min)

```
cd /workspace/run1
vllm bench sweep serve \
  --serve-cmd "vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
    --gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching \
    --max-num-batched-tokens 2048 --kv-cache-dtype auto" \
  --bench-cmd "vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions --model Qwen/Qwen3-8B \
    --dataset-name random --random-input-len 4000 --random-output-len 200 \
    --random-range-ratio 0 --ignore-eos --request-rate inf \
    --max-concurrency 8 --num-prompts 48 --metric-percentiles 50,90,99" \
  --serve-params /workspace/run1/mi300x-run-1-serve.json \
  --bench-params /workspace/run1/mi300x-run-1-bench.json \
  --after-bench-cmd "bash -c 'curl -s 127.0.0.1:8000/metrics | grep -E \"^vllm:(num_preemptions_total|num_requests_waiting|num_requests_running)\" >> /workspace/run1/metrics-after.log; echo -- >> /workspace/run1/metrics-after.log'" \
  --num-runs 3 --server-ready-timeout 900 \
  -o /workspace/run1/results -e mi300x-run-1 --show-stdout \
  2>&1 | tee /workspace/run1/sweep.log
```

The two JSON files are `bench/sweep/mi300x-run-1-*.json`, copied over by `scp`
before the container starts; `dry-run.sh` printed exactly these commands,
with `--max-num-seqs 256` appended to the server line from the serve params
(`--server-ready-timeout` is a sweep-level flag and does not appear in them).
Four things about the line that are not cosmetic:

- **`--server-ready-timeout 900`** — the default is **300 s**, after which the
  sweep raises `TimeoutError` and kills the server. The L40S's cold launch took
  ~70 s (`torch.compile` 35 s, profile and graph capture 15 s, weights ~16 s);
  a cold ROCm compile is an unknown, and it is the first thing that runs on the
  clock. Fifteen minutes costs nothing unless it is needed (source, `serve.py`,
  2026-09-07).

- **`--after-bench-cmd` replaces the cache reset** (mechanic 4), and that is
  wanted: prefix caching is off, there is nothing to reset, and the hook is the
  only place a per-level `/metrics` read fits. It runs after **every repeat**,
  so the log gets 36 reads, three per row. `num_preemptions_total` is
  cumulative, so the *difference* between consecutive reads is one repeat's
  preemptions; the first read is against zero. **The hook runs under
  `subprocess.run(check=True)`** — a non-zero exit aborts the sweep — so its
  last command must be one that cannot fail; here that is the `echo`, and
  `grep` with no match is harmless only because it is not last (`server.py`,
  2026-09-07).
- **`--show-stdout | tee`** is checkpoint A's only path to the log (mechanic 7).
- **`--percentile-metrics`, `--save-result`, `--result-dir`, `--result-filename`
  are absent** from `--bench-cmd` because the sweep appends them; a second copy
  would be argparse last-wins, harmless and confusing.

### Predictions, row by row (table 9, uncalibrated 0.70 / 0.45)

Every row sends 4 000 in, 200 out, `num_prompts = max(20, 4 × c)`. A decode step
at concurrency `c` reads the weights once and `c × 4 000` tokens of KV; the floor
is those bytes over 5.3 TB/s × 0.70. **Read the median ITL against the floor**:
the floor is bytes / (5.3 TB/s × 0.70) and the ITL is bytes / (5.3 TB/s ×
`eff_mem`), so **`eff_mem` = 0.70 × floor / ITL** — an ITL at the floor means
0.70, an ITL below it means more. Run 1's fit was the same read over twelve
levels.

| Row | c | bytes/step | TPOT floor | Inside 50 ms | Seat @ 4 200 | vs `max_num_seqs` 256 | What the row is for |
|---|---|---|---|---|---|---|---|
| c001 | 1 | 16.99 GB | 4.58 ms | yes | fits | runs | the weights read alone: `eff_mem` at batch 1, where the L40S gave its best point (0.867) |
| c008 | 8 | 21.12 GB | 5.69 ms | yes | fits | runs | |
| c032 | 32 | 35.27 GB | 9.51 ms | yes | fits | runs | the L40S's whole range ends here — 32 was its second-to-last level |
| c064 | 64 | 54.15 GB | 14.60 ms | yes | fits | runs | `docs/SLO.md` §5's batching row |
| c128 | 128 | 91.90 GB | 24.77 ms | yes | fits | runs | half the pool; the fit's anchor if the top rows misbehave |
| c192 | 192 | 129.65 GB | 34.95 ms | yes | fits | runs | |
| c224 | 224 | 148.52 GB | 40.03 ms | yes | fits | runs | last row inside the *corrected* pool |
| c240 | 240 | 157.96 GB | 42.58 ms | yes | **fits?** | runs | inside the clean arithmetic (252), outside the corrected pool (~234): **the row that says which pool figure was right** |
| c256 | 256 | 167.39 GB | 45.12 ms | yes | **no seat** | runs | at the default cap and past the pool: **preemptions expected**, `num_requests_running` peaks below 256 |
| c288 | 288 | 186.27 GB | 50.21 ms | **no** | no seat | **queues** | past the cap: `num_requests_waiting` > 0 while running ≤ 256 — the third limit, seen |
| c001-in2000 | 1 | — | — | — | — | — | prefill: TTFT floor **47.3 ms** |
| c001-in8000 | 1 | — | — | — | — | — | prefill: TTFT floor **189.1 ms**; with c001's 4 000-token **94.5 ms** the three must double in step (table 10) |

The floor at c = 288 sits over 50 ms **only at 0.70**; at anything like the
L40S's 0.83 it does not, and then no row in this sweep crosses the SLO at all —
which is the prediction's stronger form, not a failure of the sweep.

### The four reads

1. **`eff_mem`.** Median ITL at every row from c001 to c224, each divided into
   its floor at 100 %; the L40S's twelve implied values sat within ±0.02 of each
   other with the best at batch 1. Rows past the pool are excluded from the fit
   — a step with preemptions in it is not a decode step.
2. **`mfu`.** Median TTFT at c001, c001-in2000, c001-in8000 against 47.3 / 94.5
   / 189.1 ms. One request alone on the card is a prefill and nothing else; the
   ratio to floor is the utilisation, and run 1 took 0.439 from exactly this
   point. The three must scale 2.00 × / 2.00 × or something other than prefill
   is in the TTFT.
3. **Where the curve ends.** The `metrics-after.log` deltas at c240, c256, c288,
   and each row's `num_requests_running` peak — the sweep has no in-run gauge
   sampler (`bench/harness.py --sample-gauges` does; the sweep does not), so the
   *after* read is a floor on the peak, not the peak. Record it as such.
4. **Interference.** TPOT p50 and p99 against median ITL per row. Prefill is
   3.6 × faster here, so the L40S's 2.06 × should shrink; how much is the number
   that decides whether the interactive class fits this card at 4 000 tokens,
   where it did not fit the L40S (`docs/benchmarks/l40s-baseline.md` §6).

### Stop conditions

- Checkpoint A misses both pool figures by more than 3 % → stop, account for it,
  relaunch.
- Any row reports failed requests → note the row, let the sweep continue; a
  failed request in a closed loop is a rejected length or a crash, not load.
- The server dies mid-sweep → the sweep raises (the benchmark's
  `CalledProcessError`; nothing in it names `--resume`). Relaunch the identical
  line with `--resume`, and note that the rows after the restart are a second
  server launch (run 2 §6: launch-to-launch spread ~2 %).
- The clock passes 2 h with rows left → §5's drop order, then §4 regardless.

---

## 4 · Close — harvest, then destroy, then write (~10 min on the clock)

```
ls /workspace/run1/results/mi300x-run-1/*/            # 12 rows × (run=0..2.json + summary.json) = 48 files
wc -l /workspace/run1/results/mi300x-run-1/summary.csv   # 37 = 36 runs + header (pandas to_csv)
tar czf /workspace/run1.tgz -C /workspace run1
```

From the laptop: `scp root@<ip>:/workspace/run1.tgz docs/benchmarks/raw/mi300x-<date>/`,
then **unpack, count, and delete the archive** — `.gitignore` re-admits
`raw/**/*.log` and `raw/**/results/` but says nothing about `*.tgz`, so a
forgotten archive is committed whole. The count is 48 JSON — 36 `run=N.json`
and 12 `summary.json` — plus `summary.csv`, `sweep.log`, `metrics-after.log`,
and it happens before destroying anything — the 2026-08-26 loss was caught by
a count. If `summary.csv` has fewer rows than there are `run=N.json` (a
`--resume` may or may not fold skipped runs into it), rebuild it from the
JSON off-card; the JSON is the measurement. Then destroy the droplet from the console and confirm the billing
line has stopped.

Off the clock: `docs/benchmarks/mi300x-run1.md`, every row carrying the three §9
columns — coefficient used (0.70 / 0.45), fitted-to-this-run (no), accelerator
(MI300X) — and `bench/roofline.py` gains an `MI300X_RUN1` instance the way
`L40S_RUN1` exists, **without** editing `MI300X`.

---

## 5 · Budget, drop order, and what "done" means

**Expected clock:** droplet and image pull 10 min · weights 5 min (DO's network,
not RunPod's — verify) · container and `--help=all` 3 min · server launch 2 min ·
the sweep **42 min at the floors and ~85 min at twice the floors**, which is
where the L40S ran — three repeats of: c001–c064 ≈ 4 min, c128–c224 ≈ 14 min,
c240–c288 ≈ 21 min, the two length rows 2 min (summed from tables 9–10 with
`num_prompts = max(20, 4 × c)` and the batch capped at the corrected pool) ·
harvest and destroy 10 min ≈ **80–120 min**. Hard stop 2.5 h. The top three
rows are half the sweep: 4 × 288 prompts of 4 000 tokens is 4.6 M tokens of
prefill per repeat, and that is the price of placing the shelf.

**Drop order, decided now:**

1. `--num-runs 3 → 2`, at the 90-minute mark or after a crash: Ctrl-C, then
   the identical line with `--resume --num-runs 2`. Rows already complete are
   skipped whole and a half-finished row keeps its `run=N.json` files, so the
   repeats are the cheapest thing to lose — the sweep cannot vary them per row,
   but it can be told to want fewer from here on.
2. c008 and c224 — interpolable from their neighbours.
3. c192 — the fit holds on c001–c128 plus c240.

The sweep cannot skip a row on the fly. Drops 2 and 3 are: Ctrl-C, delete the
row's key from the droplet's copy of `mi300x-run-1-bench.json`, relaunch with
`--resume` — finished rows are skipped on their files, the deleted row is never
asked for.

**Never dropped:** checkpoint A · c001 and both length rows (the `mfu` read) ·
c128 · c240, c256, c288 (the three rows that place the shelf and the cap) · the
harvest count.

**Done for MI300X run 1** — four sentences, a number in each:

- the logged pool against 1 060 655 and ~986 000, and whether the L40S's 7 %
  shortfall is a property of vLLM or of that card;
- `eff_mem` on the MI300X from the median ITL, and whether batch 1 is again the
  best point;
- `mfu` from three uncontended prefills, and whether TTFT doubled with the prompt;
- which limit ended the curve — capacity at ~234–252, the cap at 256, or latency
  — against the sheet's prediction that latency is never reached.

---

## Not in this run, and where each goes

- **FP8 KV** — MI300X run 2, as one more `--serve-params` row (one model load,
  mechanic 1), so it lands against a measured BF16 pool as it did on the L40S.
- **Prefix caching and `h`** — `bench/harness.py`, never the sweep (mechanic 2).
  Table 8 has no MI300X row for the reason `bench/predictions.py` gives.
- **The histogram SLI rule and the promote branch** (written, waiting for a
  card) — need the stack on Kubernetes on a GPU node, which a droplet is not.
- **Replicas duplicating the weights read**, open since the first derivation —
  needs the 8-card droplet, which is a separate decision, not this sheet.

---

## If it goes sideways

| Symptom | First move |
|---|---|
| `docker pull` slow or refused | The image is public; check the droplet's outbound path, not credentials. 11.3 GB at 100 MB/s is 2 min |
| `TimeoutError` from the readiness wait | At 900 s this should not fire. If it does, `sweep.log` shows where the launch stood — download, `torch.compile`, profiling, graph capture — and that is read before any flag is touched |
| `vllm serve` exits at once inside the sweep | `--show-stdout` puts the traceback in `sweep.log`. Read it before touching a flag; a ROCm backend error names its env var in the message |
| OOM at launch | `--gpu-memory-utilization 0.85`, note it, and every pool figure in this sheet is no longer comparable — checkpoint A is re-derived at 0.85 before the sweep continues |
| Checkpoint A pool far below both figures | `grep -E 'non-torch|activation|Graph' sweep.log` — the three terms run 1 found; account, then decide |
| Rows past c224 stall for minutes | Expected shape at c256/c288 is slow, not stalled: 4 × 256 prompts of 4 000 tokens is 4 M tokens of prefill. Watch `metrics-after.log` grow; stall > 10 min → Ctrl-C, `--resume` |
| Sweep refuses to start: `Cannot overwrite existing experiment_dir` | A real run already used `-e mi300x-run-1`; `--resume` to continue it, or a new `-e` for a fresh one — never delete the directory |
