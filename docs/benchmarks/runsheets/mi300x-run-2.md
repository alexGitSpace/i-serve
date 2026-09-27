# Runsheet — MI300X run 2: the card configured for itself

MI300X run 1 (`docs/benchmarks/mi300x-run1.md`) fitted `eff_mem` 0.46 and `mfu`
0.166, and found latency ending the curve between 8 and 32 seats through prefill
interference: a slow prefill stretched every other seat's decode. At that
configuration the card costs 19 % more per output token than the L40S at the
plateau and ~15 % more at the TPOT p99 crossing. The configuration was chosen for
parity with the L40S runs, not for this card. This run asks three things, in
this order:

1. **Do run 1's coefficients hold on a run that did not fit them?** `docs/SLO.md`
   §9 treats a fitted coefficient as a hypothesis until a later run faces it.
2. **Which lever moves `mfu`?** Run 1 §4 and §8 named candidates it could not
   separate. Two are one flag each: the chunking of prefill, which is the token
   budget, and the attention kernel.
3. **Is there a configuration at which the card is cheaper than the L40S?**

One lever at a time: five server configurations in one sweep, each compared
against `b2048`, this run's own replay of run 1.

**Status: written 2026-09-27; reviewed the same day in two passes.** The internal
pass checked the numbers against table 12 and run 1's read-out, and the procedure
against `docs/adding-a-run.md`. The external pass checked vLLM's behaviour
against the `v0.27.1` source, and the image and the platform against their own
documentation. Their corrections are in this text; *External claims* lists what
the external pass settled. Every predicted figure comes from `bench/predictions.py`
table 12; re-run it if the module has changed since. Where this sheet disagrees
with a measured number, `docs/benchmarks/` wins.

**Why this is not the FP8 KV run.** Run 1's sheet named run 2 as the FP8 KV run.
Run 1 then showed that KV use never passed 37 %. FP8 halves the bytes per token
and so buys room this card did not use at this geometry. It stays under *Not in
this run*.

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X, **$1.99/h**
on-demand as read off the console on 2026-09-27. The console price at droplet
creation wins, and is written here when read. Expected clock **~2.6 h ≈ $5.1**
(§5); hard stop **3.5 h ≈ $7**. The credits are dated and the date is the
constraint, not the balance: run 1 spent ~$3.3 of $100.

---

## What run 1 established, and what this run spends

| From run 1 | Here |
|---|---|
| `eff_mem` 0.46 (0.41–0.49, five rows at batch 1–32), `mfu` 0.166 at 4 000 tokens | Faced by `b2048`, which is run 1's server line. c016 and c024 are batches run 1 never measured. Of the three prefill lengths only 4 000 is faced: run 1 already measured 2 000 and 8 000 off the floor (116.4 and 642.6 ms), and they are kept for the doubling ratio |
| TPOT ≈ median ITL + (c − 1) × TTFT(c = 1) / 200, written after the run from two rows (§6) | Predicts TPOT p50 at c016 and c024 before they are measured. It has no budget term, so with each row's own measured TTFT at c001 in it, it predicts **the same TPOT p50 at every budget** |
| The token-budget ceiling, 2 048 × 200 / 4 200 = 97.5, logged as 98–99 (§5) | Its falsifiable form, in full: at a budget of 4 096, **~195 running** (196–197 if the one or two partly prefilled requests run 1 saw recur), **output within 5 % of 531.7 tok/s**, and **the median ITL at c256 roughly doubling** from run 1's 170 ms |
| Output tok/s flat from c032 up: 499 at c032, **531.7 at c256** (§5) | Every plateau here is read at c256, run 1's c256 with the same prompt count |
| TTFT ×2.21, then ×2.50 per doubling of the prompt; attention FLOPs alone explain ×2.08 and ×2.16 (§4) | At a budget of 8 192 every prompt is one engine step. Chunking made the ×2.50 if `b8192`'s second ratio falls below **×2.33**, the midpoint to ×2.16 |
| Pool 1 123 065 tokens on launch 1 and 1 138 444 on launch 2, 1.4 % apart; the CUDA graphs sit outside the pool (§2) | Checkpoint A is read at **every** launch: a larger budget raises the peak activation the pool pays for |

---

## The five serve rows

A **serve row** is one `--serve-params` entry: one server launch, kept up across
all eight benchmark rows (mechanic 1 of `docs/instrument-vllm-bench-sweep.md`).
The sweep runs them in file order, which is also the order of importance.

| Serve row | Budget | Attention | Question | Compared with |
|---|---|---|---|---|
| `b2048` | 2 048 | `ROCM_ATTN` | Run 1 replayed: do 0.46 and 0.166 hold? | run 1, table 12 |
| `b8192` | 8 192 (this build's default) | `ROCM_ATTN` | Chunking: is TTFT's ×2.50 the chunks? And does TPOT p50 stay put when the budget moves? | `b2048` |
| `triton` | 2 048 | `TRITON_ATTN` | The attention kernel, at every row | `b2048` |
| `aiter-fa` | 2 048 | `ROCM_AITER_FA` | The attention kernel, **at c001 only** (below) | `b2048` |
| `b4096` | 4 096 | `ROCM_ATTN` | Run 1's falsifiable form | `b2048`, table 12 |

Every row pins `max_num_seqs` 256, as run 1 did. What each backend runs for
prefill and for decode is in `docs/GLOSSARY.md`, *Attention backend*: both kernel
rows replace the decode kernel as well as the prefill one, so a decode step that
moves under them is attributed to the kernel, not to `eff_mem`.

**`aiter-fa` is read at c001 only.** On this build Qwen3-8B runs on the V2 model
runner, which orders a step's requests decode first and then by ascending token
count (`vllm/v1/worker/gpu/model_runner.py`, `sort_batch_req_ids`).
`ROCM_AITER_FA` splits the batch assuming the older order, in which continued
prompts come before new ones (`vllm/v1/attention/backends/utils.py`,
`split_decodes_prefills_and_extends`). When a new prompt's first chunk is
shorter than another prompt's continuing chunk in the same step, the continuing
one is taken for a new prompt. It is then attended without its cached context:
less work, a faster step, and wrong output. A random-token benchmark with
`--ignore-eos` cannot see that. The misclassification is read off the source and
was not observed. `ROCM_ATTN` and `TRITON_ATTN` do not split the batch this way.
A single request never mixes with another, so c001 at all three lengths is
sound, and those rows are where `mfu` comes from. The row's c008–c256 run
anyway, because the benchmark rows are one product with the serve rows. They
are recorded and **not read** as measurements.

**`VLLM_ROCM_USE_AITER` is not set.** It is an environment variable, and the
sweep hands its own environment to every server it spawns
(`vllm/benchmarks/sweep/server.py`), so it would apply to all five rows at once
and switch several kernel families together. The forced backend does not need
it. What it would switch, and what it does not on gfx942, is in the glossary's
*AITER* entry.

---

## External claims, checked against `v0.27.1` (2026-09-27)

| Claim | Source |
|---|---|
| `--attention-backend` exists and takes a backend name; the sweep key `attention_backend` becomes that flag | `vllm/engine/arg_utils.py`; the dry-run |
| A forced backend that fails validation raises at startup, with no fallback to `ROCM_ATTN` | `vllm/platforms/rocm.py`, `get_attn_backend_cls` |
| A forced backend logs `Using <NAME> backend (selected via --attention-backend).`; run 1's auto-selection logged `Overriding with ROCM_ATTN out of potential backends: […]` | the same function; run 1's `sweep.log` |
| `ROCM_AITER_FA`'s validation checks the CDNA generation and not the `aiter` package, which is imported lazily. With a broken package the row logs the line above and **then** fails at warmup, before `Starting vLLM server` | `rocm_aiter_fa.py`, `supports_compute_capability`; `vllm/_aiter_ops.py` |
| `ROCM_AITER_FA` takes kernel block sizes 16 and 32 and head size 128; `TRITON_ATTN` takes any multiple of 16 | `rocm_aiter_fa.py`, `triton_attn.py` |
| `aiter` v0.1.19 is built with prebuilt kernels for gfx942 and gfx950 into vLLM's ROCm base image, and its branch is written to `/app/versions.txt` | `docker/Dockerfile.rocm_base` |
| The package is present in the published image: run 1's log lists `ROCM_AITER_UNIFIED_ATTN`, which is listed only when `find_spec("aiter")` succeeds on CDNA 3 or later. Presence, not a working import | run 1's `sweep.log`; `rocm.py`, `_get_backend_priorities` |
| The default `max_num_batched_tokens` for an API server on a card of 70 GiB or more is 8 192; `long_prefill_token_threshold` defaults to 0, so a lone 8 000-token prompt is one step at 8 192 | `arg_utils.py`; `vllm/config/scheduler.py` |
| The CUDA graph capture sizes are the same at all three budgets | `vllm/config/vllm.py` |
| Between serve rows the sweep SIGKILLs the old server's process group and starts the next at once. vLLM refuses to start when free memory is below `gpu_memory_utilization` × total | `benchmarks/sweep/server.py`; `v1/worker/utils.py`, `request_memory` |
| `--resume` skips a repeat whose `run=N.json` exists, and skips a serve row's launch when every benchmark row under it has a `summary.json`. `summary.json` is rewritten from `run=0 … num_runs − 1` every time, and `summary.csv` is written once, when a launch finishes, from those files | `benchmarks/sweep/serve.py` |

---

## 0 · Before the card (no credits spent)

- [x] `bench/sweep/dry-run.sh mi300x-run-2-serve.json mi300x-run-2-bench.json`:
      five server launches, each with the intended budget and backend flags; exit 0
      (2026-09-27). The script prints 120 benchmark commands at its own
      `--num-runs 3`; the run's `--num-runs 2` gives 80.
- [x] `python3 bench/predictions.py`, table 12 open beside the terminal.
- [x] This sheet reviewed.
- [x] This sheet committed (`7c13a57`).
- [x] Price read off the console and written into the header: $1.990/h at
      creation, as written.
- [x] Image: *Quick Start → ROCm Software*, version 7.14, the same host as run 1.

---

## 1 · Droplet up (~15 min)

On the droplet, in this order, each a gate for the next:

```
rocm-smi --showproductname --showmeminfo vram      # one MI300X VF, 205 822 885 888 B
docker ps -a                                        # the preinstalled `rocm` container, as in run 1
docker stop rocm
mkdir -p /workspace/hf /workspace/run2
docker pull vllm/vllm-openai-rocm:v0.27.1
```

From the laptop: `scp bench/sweep/mi300x-run-2-*.json root@<ip>:/workspace/run2/`.

Then, inside `tmux new -s run2` (reattach with `tmux attach -t run2`):

```
docker run -it --rm --name run2 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v /workspace:/workspace -v /workspace/hf:/root/.cache/huggingface \
  --entrypoint bash vllm/vllm-openai-rocm:v0.27.1
```

Inside it, before anything else:

```
vllm --version                                      # 0.27.1+rocm723
cat /app/versions.txt                               # AITER_BRANCH: v0.1.19
pip list 2>/dev/null | grep -i aiter
python3 -c "import aiter; print(aiter.__file__)"    # gate for aiter-fa
hf download Qwen/Qwen3-8B
```

**If `import aiter` fails**, delete `aiter-fa` from the droplet's copy of
`mi300x-run-2-serve.json` before launching, and note it here. `triton` then
carries the attention question alone. This is the one row whose loss §5 does not
list as a drop.

---

## 2 · Checkpoint A — five startup logs

Each serve row prints its own startup log into `sweep.log`. Read the first while
`b2048`'s c001 runs, and each later one as its row starts:

```
grep -E 'non-default args|GPU KV cache size|Maximum concurrency|selected via --attention-backend|out of potential backends|Graph capturing|non-torch|activation|start build \[|finish build|Traceback|Starting vLLM server' /workspace/run2/sweep.log
```

| Log line | Predicted | Gate |
|---|---|---|
| `GPU KV cache size`, `b2048` | **~1 123 000** tokens: 192 GiB less the 2.1 % shortfall (table 12 prints the seats, 267 at 4 200). Run 1 logged 1 123 065 and 1 138 444 | ±5 %, the band `docs/adding-a-run.md` §1 sets; there is one figure to confirm here, not two to separate. On a miss, account for it from the `non-torch` and `activation` lines. Relaunch only if the account finds something in the launch that differs from run 1's |
| `GPU KV cache size`, every other row | Within a few percent of `b2048`'s, as run 1's two launches were, and lower at a larger budget by the change in `activation` | Not a gate. Record it. At 8 192 a pool below **1 075 200** (256 × 4 200) means c256 may preempt there, which is a finding |
| backend | `Overriding with ROCM_ATTN out of potential backends` for the three `b*` rows, `TRITON_ATTN (selected via --attention-backend)`, `ROCM_AITER_FA (selected via --attention-backend)` | A forced row passes only with its line **and** a `Starting vLLM server` after it with no `Traceback` between. The line alone does not prove the kernel loaded (*External claims*) |
| `non-default args` | `max_num_batched_tokens` as the row says, `max_num_seqs` 256, and `attention_backend` on the forced rows | Anything else means the override did not apply |
| `start build [` / `finish build […] cost` | None expected: the kernels this model needs are prebuilt | Not a gate. A build line means the launch is compiling; its time goes into §5's clock |
| `Graph capturing finished … took` | 4.12 GiB at `max_num_seqs` 256 (run 1) | Not a gate. It sits outside the pool on this build (run 1 §2) |

---

## 3 · The sweep — five serve rows, eight benchmark rows, two repeats

Write the line below into `/workspace/run2/sweep.sh` and run `bash sweep.sh`, so
the raw capture keeps the exact launch, as run 1's did:

```
cd /workspace/run2
vllm bench sweep serve \
  --serve-cmd "vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
    --gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching \
    --max-num-batched-tokens 2048 --kv-cache-dtype auto" \
  --bench-cmd "vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions --model Qwen/Qwen3-8B \
    --dataset-name random --random-input-len 4000 --random-output-len 200 \
    --random-range-ratio 0 --ignore-eos --request-rate inf \
    --max-concurrency 8 --num-prompts 48 --metric-percentiles 50,90,99" \
  --serve-params /workspace/run2/mi300x-run-2-serve.json \
  --bench-params /workspace/run2/mi300x-run-2-bench.json \
  --after-bench-cmd "bash -c 'curl -s 127.0.0.1:8000/metrics | grep -E \"^vllm:(num_preemptions_total|num_requests_waiting|num_requests_running)\" >> /workspace/run2/metrics-after.log; echo -- >> /workspace/run2/metrics-after.log'" \
  --num-runs 2 --server-ready-timeout 1800 \
  -o /workspace/run2/results -e mi300x-run-2 --show-stdout \
  2>&1 | tee -a /workspace/run2/sweep.log
```

A relaunch is the same line with `--resume` added, written to
`sweep-resume.sh`, with the time appended to `sweep.log` first (`echo "===== RESUME
$(date -u) =====" >> sweep.log`). Run 1's line, with three changes:

- **`--num-runs 2`.** Run 1's repeats agreed on the median ITL to 0.01 ms at batch 1
  and c008, and to 2 % at c032. The third repeat buys less than a fifth serve row.
- **`--server-ready-timeout 1800`.** Run 1's launches took 67 s cold and 37 s warm,
  from `non-default args` to `Starting vLLM server`. A launch that compiles
  kernels is the unknown. A build that is killed here is dropped, not retried
  (§5).
- **`tee -a` from the first launch.** A relaunch appends to the log instead of
  truncating it (run 1, departure 4).

The benchmark rows are `num_prompts = max(20, 4 × c)`: c001, c008, c016, c024,
c032, c256, and c001 at 2 000 and 8 000 tokens. c016 and c024 fill the gap in
which run 1 could only place the crossing. c256 is run 1's c256.

### Predictions (table 12, `eff_mem` 0.46 / `mfu` 0.166, fitted by run 1)

| Row | Decode step | TTFT alone | TPOT p50 | Inside 50 ms |
|---|---|---|---|---|
| c001 | 6.97 ms | 256.3 ms | 6.97 ms | yes |
| c008 | 8.71 ms | — | 17.68 ms | yes |
| c016 | 10.69 ms | — | 29.91 ms | yes |
| c024 | 12.68 ms | — | 42.15 ms | yes |
| c032 | 14.66 ms | — | 54.38 ms | **no** |
| c001-in2000 | 6.85 ms | 128.1 ms | — | — |
| c001-in8000 | 7.22 ms | 512.5 ms | — | — |

TPOT p50 crosses 50 ms at c = 30. The p99 is not modelled: run 1's crossed
between 8 and 32, near 20 by interpolation, and c016 and c024 place it to within
8 seats. The decode step and TPOT columns are the prediction for every serve row
whose kernel is `ROCM_ATTN`; for `triton` they are the null the kernel is read
against.

| Budget | Prefill steps, 2 000 / 4 000 / 8 000 tokens | Token-budget ceiling | Running at c256 | Set by |
|---|---|---|---|---|
| 2 048 | 1 / 2 / 4 | 97.5 | 98 | token budget |
| 4 096 | 1 / 1 / 2 | 195.0 | 195 | token budget |
| 8 192 | 1 / 1 / 1 | 390.1 | 256 | `max_num_seqs` = concurrency |

At 8 192 the cap and the client's own concurrency are both 256, so c256 there
cannot tell them apart and cannot test the 390. It shows only that the pool
admits 256.

| `mfu` | TTFT alone, 4 000 tokens | Seats inside 50 ms, p50 |
|---|---|---|
| 0.166 (run 1) | 256.3 ms | 29 |
| 0.200 | 212.7 ms | 33 |
| 0.250 | 170.2 ms | 40 |
| 0.300 | 141.8 ms | 45 |
| 0.450 (the prior) | 94.5 ms | 60 |

**The plateau breaks even with the L40S's $0.873/1M at 633 output tok/s**, 19.1 %
above run 1's 531.7 at c256, at $1.99/h. That is the number a serve row's c256 is
read against.

### The five reads

1. **The faced coefficients (`b2048`).** The median ITL at c001–c032 against the
   decode step column, and TTFT p50 at 4 000 tokens against 256.3 ms. Every row
   goes into the report's §9 table with *fitted to this run: no*. That is the
   first time an MI300X row can say so of a measured coefficient.
2. **Interference, per serve row read at concurrency.** TPOT p50 at c008–c032
   against the model with that row's own measured c001 TTFT in it, and TPOT p99's
   crossing placed between two rows 8 seats apart.
3. **What the budget moved.** The running count at c256 from the engine's
   10-second lines, a floor on the peak as in run 1; output tok/s at c256 against
   531.7 ± 5 %; the median ITL at c256; and TPOT p50 at c008–c032 across `b2048`,
   `b4096` and `b8192`, which the model says coincide.
4. **`mfu`, per serve row.** Implied `mfu` at 2 000, 4 000 and 8 000 tokens and the
   two doubling ratios. Chunking made the ×2.50 if `b8192`'s second ratio is below
   ×2.33 and `b2048`'s is not. A kernel was the cost if `triton` or `aiter-fa`
   moves TTFT at 4 000 by more than `b2048` moves from run 1's 256.9 ms.
5. **Cost, per serve row read at concurrency.** $/1M at c256 against 633 tok/s;
   at the p99 crossing, interpolated like run 1 §7, against the L40S's ≈ $1.14.

### Stop conditions

- `b2048`'s pool misses ±5 %: stop and account for it (§2).
- A forced row fails §2's gate: stop. Its numbers belong to another kernel.
- **The sweep raises at a serve-row boundary.** Read the traceback in `sweep.log`
  above the sweep's own `Server process crashed`. If it says `Free memory on
  device … is less than desired GPU memory utilization`, the previous server's
  memory was not yet released. Run `rocm-smi --showmeminfo vram --showpids` on
  the host until it shows ~0.3 GB in use and no process, as at boot in run 1, then
  relaunch with `--resume`. A process stuck in state `D` (`ps -eo pid,stat,cmd |
  grep ' D'`) will not release it: reboot the droplet. A `Selected backend … is
  not valid` error means deleting that row from the serve JSON and relaunching.
  Either way, checkpoint A is read again for the new launch.
- Any row reports failed requests: note the row and let the sweep continue.
- At every serve-row boundary, §5's clock rule.
- The hard stop: Ctrl-C, then §4 regardless.

---

## 4 · Close — harvest, then destroy, then write

```
ls -d /workspace/run2/results/mi300x-run-2/SERVE--*/ | wc -l        # 40 = 5 serve × 8 bench rows
find /workspace/run2/results/mi300x-run-2 -name 'run=*.json' | wc -l   # 80
find /workspace/run2/results/mi300x-run-2 -name 'summary.json' | wc -l # 40
wc -l /workspace/run2/results/mi300x-run-2/summary.csv              # 81 = 80 repeats + header
grep -c -- '^--$' /workspace/run2/metrics-after.log                 # 80 hook reads
tar czf /workspace/run2.tgz -C /workspace run2
```

Write the expected counts down before comparing: each dropped serve row takes 8
directories and 16 `run=N.json` with it, and each serve row run at `--num-runs 1`
takes 8 `run=N.json`. After a `--num-runs 1` relaunch, `summary.json` and
`summary.csv` fold only `run=0` of every row, including rows finished earlier,
as run 1's did. The `run=N.json` files are the measurement.

From the laptop: `scp` the archive into `docs/benchmarks/raw/mi300x-<date>/`,
where it unpacks as `run2/`. On the day of run 1 that is beside run 1's `run1/`.
Check the checksum on both ends, then unpack, count, and delete the `.tgz`:
`.gitignore` says nothing about `*.tgz`. Then destroy the droplet and confirm the
billing line has stopped.

Off the clock: `docs/benchmarks/mi300x-run2.md` and
`bench/measured_mi300x_run2.py`. Every row carries the three §9 columns, with the
coefficient each prediction used: `MI300X_RUN1`'s 0.46 / 0.166, *not fitted to
this run*. A lever that moved `mfu` gets its own `Accelerator` instance, as
`MI300X_RUN1` did, rather than an edit to run 1's.

---

## 5 · Budget, drop order, and what "done" means

**Expected clock.**

| Step | Time |
|---|---|
| Droplet and image pull | 10 min |
| Weights and container | 5 min |
| Five launches, ~1.5 min each | ~8 min |
| The sweep: two repeats × five serve rows × ≈ 12.1 min | ≈ 121 min |
| Harvest and destroy | 10 min |
| **Total** | **≈ 2.6 h** |

Why the launches take ~1.5 min: a new budget or backend recompiles, and run 1
compiled for 19 s cold.

Where the 12.1 min per serve row and repeat comes from, at `b2048`'s pace:
- c256: 6.4 min, 1 024 × 200 output tokens at run 1's 531.7 tok/s;
- the three single-request rows: 1.7 min;
- c008–c032: 2.4 min;
- the sweep's own gap between repeats, measured from run 1's `run=N.json` times:
  1.6 min.

A faster kernel shortens its own row.

**The clock rule, at every serve-row boundary:** if the rows left × 26 min plus
15 min for harvest would pass the 3.5 h hard stop, take the next drop. Every drop
is Ctrl-C at the boundary, so no repeat is lost half-done, then an edit, then a
relaunch with `--resume`:

1. `--num-runs 2 → 1` for the serve rows not yet started. The edit is to the
   line, not the JSON. §4's counts change with it.
2. `b4096`: delete it from the serve JSON. Its budget claim loses its middle point;
   `b8192` still says whether the budget moves TPOT p50.
3. `aiter-fa`: delete it from the serve JSON. `triton` keeps the kernel question.

**Never dropped:** checkpoint A at every launch; `b2048`, `b8192` and `triton`,
whole; the harvest count.

**Done for MI300X run 2**, five sentences with a number in each:

- run 1's `eff_mem` 0.46 and `mfu` 0.166 against a run that did not fit them: the
  decode step at five batches and TTFT at 4 000 tokens, each with its error;
- where TPOT p99 crosses 50 ms, placed within 8 seats, against the model's c = 30
  for p50;
- what the budget moved: running at c256 against 98 / 195, output against 531.7,
  the median ITL at c256, and TPOT p50 at c008–c032 against "unchanged";
- which lever moved `mfu`, and by how much: `b8192`'s second doubling ratio
  against ×2.33, and TTFT at 4 000 tokens under `triton` and `aiter-fa` against
  `b2048`'s;
- the cost verdict: the best serve row read at concurrency, its plateau against
  633 tok/s and its p99 crossing against the L40S's ≈ $1.14.

---

## Not in this run, and where each goes

- **FP8 KV.** Room the card did not use at this geometry. It is worth a run once a
  configuration fills the pool. `b8192`'s c256 will say how close 256 running
  comes to that launch's pool.
- **`aiter-fa` at concurrency, and the two levers together.** The batch-order
  mismatch above has to be settled first, by the V1 model runner
  (`VLLM_USE_V2_MODEL_RUNNER=0`, with its own `b2048` control) or by a vLLM
  release that fixes it. Either way it is a launch of its own.
- **`VLLM_ROCM_USE_AITER=1` as a whole-stack switch.** It is an environment
  variable and several levers at once; alone it still leaves attention on
  `ROCM_ATTN`, which comes first among the candidates (`rocm.py`).
- **The GEMM itself.** Run 1 §8's "untuned GEMMs". Neither lever here touches the
  BF16 GEMM path on gfx942. It comes next if neither lever moves `mfu`, and it
  needs its own check of what tuning costs on the clock.
- **What the 6.6 ms batch-1 step is made of** (run 1 §8): the weights read at
  under half of peak, or a per-step cost. The sampler that `generation_config`
  imposes is one candidate; separating it is a different run.
- **`eff_mem` above batch 32**: it needs output-heavy geometry (run 1 §8).
- **The router on one card**: `mi300x-run-3.md`, re-derived after this run,
  because its seat effect depends on how many seats per engine sit inside 50 ms.

---

## If it goes sideways

| Symptom | First move |
|---|---|
| `import aiter` fails | Delete `aiter-fa` before launching (§1) |
| A forced row logs its backend line, then a `Traceback` before `Starting vLLM server` | The kernel did not load. Delete the row and relaunch with `--resume`; the rows before it are kept |
| `Selected backend … is not valid` at startup | The reason is in the message. Delete the row and relaunch with `--resume` |
| `Free memory on device … is less than desired` at a serve-row boundary | Wait for `rocm-smi --showpids` to show no process, then `--resume` (§3) |
| `TimeoutError` at a forced row's launch | A kernel build ran past 30 min. Read where it stood in `sweep.log`, then drop the row: a second attempt costs another 30 |
| c256 at `b8192` preempts | Expected only if that launch's pool is below 1 075 200 tokens (§2). Record the count in `metrics-after.log`: that is the pool, not a failure |
| `Cannot overwrite existing experiment_dir` | A real launch already used `-e mi300x-run-2`: `--resume` to continue it, never delete the directory |
