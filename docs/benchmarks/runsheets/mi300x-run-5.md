# Runsheet — MI300X run 5: the pair with its card split, and a public claim at TP = 1

MI300X run 4 left one question it could not see into: two engines of one model
on one card are equal alone. Together, the engine launched first serves each
token about 1.9× slower than the other, and no CPU thread separates them
(`docs/benchmarks/mi300x-run4.md` §3, §8). **Block B splits the card's compute
units between the two engines and asks whether the asymmetry survives.** Block C
spends the rest of the session on a different kind of row. It faces a published
claim, InferenceX's Llama-3.3-70B FP8 on MI300X, with a ceiling written before
the card and on the image this repository pins.

**Status: written 2026-10-17, the day the credits expire, in one sitting.** The
internal pass checked the scripts against `bench/` and against run 4's capture.
The external pass checked ROCr, the KFD driver, vLLM `v0.27.1` and InferenceX's
own repository (§0). Usually a sheet waits for two passes on different days.
This one could not, and its §8 is wider because of that.

**The commands are scripts**, as in run 4. Every stage is a file in
`docs/benchmarks/raw/mi300x-2026-10-17/run5/`, copied to
`/workspace/i-serve/run5/` beside `bench/`. A block printed below is an excerpt,
and the file wins.

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X at **$1.99/h**
on-demand, as read off the create page on 2026-10-17. MI300X × 8 was listed at
the same price per GPU, but out of capacity in ATL1, and RIC1 had no GPUs at
all. Expected clock **≈ 1.6 h ≈ $3.2**, session budget **≤ 2.5 h ≈ $5**, hard
stop. The credits expire **2026-10-18**, and *Billing* gives no time of day.
This sheet assumes 00:00 UTC, so **the droplet is destroyed by 23:00 UTC on
2026-10-17** whatever has been measured. About $85 of credit is left after
runs 1–4.

**Predicted figures come from two places.** Block B's come from run 4's measured
rows. Block C's come from `ceiling.py` in this run's directory, which calls
`bench/roofline.py`'s own functions on a `Model` built from the checkpoint's
`config.json`. The model is not added to `predictions.MODELS`, because a model
there reaches the page by itself (`bench/tests/test_export_site_data.py`).
Promoting it is a separate change. Where this sheet disagrees with a measured
number, `docs/benchmarks/` wins.

---

## What this run is not

- **Not the router.** No block-B arm. The fleet question needs two cards, and
  on 2026-10-17 this account could get neither: MI300X × 8 was out of capacity,
  and a second 1 × droplet needs a separate access request.
- **Not a reproduction of InferenceX.** Their frontier is a maximum over
  TP ∈ {1, 2, 4, 8} on vLLM 0.10.1. This run measures TP = 1 on vLLM 0.27.1,
  and §5 says why that still faces the claim.
- **Not a coefficient fit for a new model.** One server line, three
  concurrencies. The readings face `eff_mem` from run 2; they do not replace it.

## What run 4 established, and which of it transfers

| From run 4 | Status here |
|---|---|
| Each engine alone: 12.43 / 12.34 ms median ITL at c = 32 | Not re-measured: block 0a's question is answered |
| Together: the first launched 1.90× and 1.96× slower, in both orders | B0 re-reads it on this droplet as the control for B1 |
| A cold `torch.compile` cache logged a 2.9 % smaller pool for the engine that compiled | `s1.sh` warms the cache with one engine first |
| Pools: each engine 506 525 at 0.45, warm cache | Checkpoint A's reference |
| Droplet check 5.04 ms, TTFT 156 ms at c = 1: `FIT = mi300x-run2` | Re-read: another droplet may be rented. The TTFT is B1's baseline |
| CPU: two busy EngineCore threads per engine, each on its own core | Ruled out; `top -H` stays as a record, `ps -L` and `pin.sh` are dropped |

---

## 0 · Before the card (no credits spent)

- [x] **`HSA_CU_MASK`**, read in ROCr's source (`core/util/flag.cpp`,
      `amd_aql_queue.cpp`, rocm-systems `develop`, not the 7.2.3 build in the
      image):
      - The syntax is `GPU_list:CU_list`, so `0:0-151` is valid.
      - The mask is applied to every AQL queue the process creates, so the
        PyTorch inside vLLM inherits it without knowing.
      - A bad set is ignored silently, and so is everything after it.
- [x] **What a contiguous range removes**, read in the KFD driver
      (`kfd_mqd_manager.c`, `mqd_symmetrically_map_cu_mask`, Linux `master`).
      On GC 9.4.3 the mask bits are dealt round-robin over the XCCs first. So
      `0-151` is 19 CUs on each of the eight XCDs, not four whole XCDs. Both
      engines keep every XCD's L2 and every path to memory, and only the CUs
      are split.
- [x] **What cannot confirm the mask**: `rocminfo` and
      `hipDeviceProp.multiProcessorCount` read the physical count, 304, either
      way. Two things can:
      - the TTFT at c = 1, because a 4 000-token prefill is compute-bound
        (§3, the gate);
      - `/sys/kernel/debug/kfd/mqds` on the host, which is each queue's MQD
        with its CU-mask registers (raw evidence only, read off-card).
- [x] **How the card arbitrates two processes**, from the same driver source:
      - MI300X has no MES. GC 9.4.3, the IP version behind gfx942, runs HWS:
        the CP firmware maps the queues of up to `hws_max_conc_proc`
        processes at once, from a runlist KFD writes.
      - Processes are time-sliced, each for its quantum, only when the
        runlist is *oversubscribed*: more processes than `hws_max_conc_proc`,
        or more compute queues than the CP has. Then `dmesg` reads "Runlist is
        getting oversubscribed".
      - HIP opens 4 hardware queues per process per device by default
        (`GPU_MAX_HW_QUEUES`).
      - Nothing documented says the first process loses. `hostrec.sh` reads
        `dmesg` after every paired call, and B2 holds each process to one
        queue.
- [x] **The InferenceX claim**, from its page and from its repository at
      `c198b20e`, the last commit before the 70B config was removed on
      2025-10-29:

      | | |
      |---|---|
      | Claim | 1 482 tok/s per GPU at 50 tok/s per user; 1 985 at 30, 718 at 75, 457 at 100; 8k in / 1k out; vLLM, FP8; run 2025-10-29 |
      | Checkpoint | `amd/Llama-3.3-70B-Instruct-FP8-KV`, not gated, 72 674 025 601 B of files |
      | Image | `rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1` |
      | Sweep | TP ∈ {1, 2, 4, 8} × concurrency ∈ {4, 8, 16, 32, 64}; the page does not say which TP gives which point |
      | Server line | `--dtype float16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --max-num-seqs $CONC --max-num-batched-tokens 131072 --max-model-len 9216 --no-enable-prefix-caching --async-scheduling --swap-space 64`, `custom_ops` `-rms_norm,-quant_fp8,-silu_and_mul`; `VLLM_ROCM_USE_AITER_MHA=1` at c ≥ 16 |
      | Client | `kimbochen/bench_serving`, `--ignore-eos`, 10 × c prompts, random lengths in **[0.8 L, L]** (`sample_uniform`, read in its source) |
- [x] **vLLM `v0.27.1`**, read at the tag:
      - Every flag above exists except `--swap-space`, which is removed and
        rejected by argparse.
      - The three `custom_ops` names are still registered.
      - `--async-scheduling` is already the default.
      - `VLLM_ROCM_USE_AITER` defaults to off, the image does not set it, and
        `_MHA` does nothing without it.
      - FP8 checkpoints are converted to the variant gfx942 computes in
        (e4m3fnuz), and the checkpoint's per-layer KV scales are read.
- [x] **The one trap in the client.** `vllm bench serve --random-range-ratio r`
      samples **[L (1 − r), L (1 + r)]** at `v0.27.1`, not InferenceX's
      [r L, L]. At r = 0.8 that is up to 14 745 input tokens against a
      9 216 limit. `s5.sh` centres at 0.9 L with r = 0.111: 6 553–8 191 in
      after one special token, 819–1 025 out, InferenceX's distribution. The
      server clamps `max_tokens` to the model length, so no request fails on
      it.
- [x] The adversarial review (2026-10-17) found two blockers, one
      near-blocker and nine minor defects, each fixed where it lived:

      | # | Defect | Fix |
      |---|---|---|
      | B1 | `s3.sh` stopped only on `MASK_NOT_TAKEN`, so a reader's traceback let B1 run ungated | B1 runs only on `MASK_TAKEN` or `MASK_UNCLEAR` |
      | B2 | A failed 70B download ended `s1.sh` before the droplet check, which gives `FIT` and B1's baseline | `s1.sh` notes it and goes on; `s5.sh` fetches again |
      | N1 | Each `dmesg-*.txt` held everything since boot, so B0's warning would also show in B2's file | `hostrec.sh` keeps only the lines a call adds |
      | M1 | `vllm bench serve` drops per-request records unless asked | `--save-detailed` |
      | M2 | `dropcheck` killed the harness once the summary existed, before its last two files | it waits for `-requests.jsonl`, the level's last write |
      | M3 | A failed harness in a pair still ended in `S2DONE` / `S3DONE` | `together` stops on `PAIR_FAILED` |
      | M4 | No branch for B0 not reproducing run 4 | §3's first table |
      | M5 | Dropping c = 16 relaunches the 70B | decided before `s5.sh` starts (§6) |
      | M6 | The HWS description implied time-slicing without oversubscription | corrected above |
- [x] The scripts' readers were run on run 4's capture:
      - `together`'s pair line reads run 4's 0b as 52.23 / 27.51 ms, ratio 1.90.
      - `s3.sh`'s mask verdict reads run 4's two unmasked checks as
        `MASK_NOT_TAKEN`, ratio 1.00.
      - `bench/harness.py --dry-run` exits 0 on both harness lines.
- [ ] The droplet's price on the create page at creation, written into §6.

---

## 1 · Droplet up, and the plain pair (~20 min, mostly download)

Run 3's §1 verbatim, with `run5` for `run3`: the droplet, the image
`vllm/vllm-openai-rocm:v0.27.1`, the container with its device flags and
`--network host`, and `tmux new -s run5` with the container in window 0,
`hostrec.sh` in window 1 and a host shell in window 2. `scp` carries `bench/` and this run's scripts to
`/workspace/i-serve/` and `/workspace/i-serve/run5/`. No router image is built.

```
mkdir -p /workspace/run5 && date -u > /workspace/run5/droplet-up.txt   # host
bash /workspace/i-serve/run5/s1.sh        # container; progress in /workspace/run5/stage
```

`s1.sh` runs these steps in order:

1. Starts the 70B download in the background.
2. Downloads Qwen3-8B.
3. Launches one engine alone to fill the compile cache, then stops it.
4. Launches the pair serially, gated as in run 4.
5. Records each process's `HSA_CU_MASK` and `GPU_MAX_HW_QUEUES` into
   `env-plain.txt`. Only those names are printed, because the full environ
   would carry the HF token.
6. Waits for the download to finish before the droplet check. A failed
   download is written to `download-status.txt` and does not stop the script.

The check prints the median ITL and the median TTFT at c = 1 against 8001.
Set `FIT` from the ITL, as in run 4, and **keep the TTFT**: it is B1's baseline.

---

## 2 · Checkpoint A

| Log line | Predicted | On a miss |
|---|---|---|
| `GPU KV cache size`, each engine, plain launch | 506 525 (runs 3–4) | outside 5 %: stop and account for it from the two memory lines |
| The same, masked launch (B1) | 506 525: a mask changes no allocation | outside 5 %: the mask reached the profiler's run; note it, carry on |
| `Available KV cache memory`, 8000 against 8001 | equal | the warm step did not take: kill both and relaunch per §1 |
| dev mode, `/reset_prefix_cache` route, idle reset | as run 4 | as run 4 §8 |

---

## 3 · Block B — the pair with its card split (~20 min, with two relaunches)

Each step uses `fleet-bill-half` on both engines at once, with a cache reset
first, exactly run 4's 0b. Each step prints one line into `pairs.txt`: both
medians, which port is slow, and the ratio.

| Step | What runs | Predicted |
|---|---|---|
| B0 | the plain pair, 8000 first (`s2.sh`) | run 4: 8000 slow, ratio **1.90–1.96** |
| B1 check | the pair relaunched with `HSA_CU_MASK=0:0-151` on 8000 and `0:152-303` on 8001, then the c = 1 check on masked 8001 | TTFT ratio **≈ 2** against s1's baseline; gate at ≥ 1.5 |
| B1 | the masked pair, 8000 first (`s3.sh`) | below |
| B2 | only on the branch: the plain pair, each process at `GPU_MAX_HW_QUEUES=1` (`s4.sh`) | below |

```
bash /workspace/i-serve/run5/hostrec.sh        # host, window 1, before s2; left running
bash /workspace/i-serve/run5/s2.sh $FIT        # container: B0
bash /workspace/i-serve/run5/s3.sh $FIT        # container: B1
# s3, s4 and s5 wait at B1_WAIT_CARD / B2_WAIT_CARD / C_WAIT_CARD: in window 2, run
#   rocm-smi --showmeminfo vram --showpids
# until ~0.3 GB is used and no KFD process is listed, then touch /workspace/run5/card-empty
```

**Is B0 run 4's pair?** B1 means something only against a control that shows
the asymmetry.

| B0 ratio | Next |
|---|---|
| ≥ 1.6, 8000 slow | as run 4: B1 |
| ≤ 1.2 | this droplet does not reproduce it. Record the line, skip B1 and B2, go to §4 |
| between | B1, read against B0's own ratio rather than run 4's |

**The mask's gate.** A 4 000-token prefill at c = 1 is compute-bound, so half
the CUs should come close to doubling the TTFT. `s3.sh` writes the verdict to
`mask-verdict.txt`:

| TTFT ratio | Verdict | What follows |
|---|---|---|
| ≥ 1.5 | `MASK_TAKEN` | B1 runs |
| 1.15–1.5 | `MASK_UNCLEAR` | B1 runs; read it with the ratio beside it |
| ≤ 1.15 | `MASK_NOT_TAKEN` | `s3.sh` stops; the variable did not reach the queues. Run B2 if the clock allows, then §4 |
| no verdict | `MASK_GATE_FAILED` | the reader failed: `droplet-check*.txt` and the two JSONs, then §8 |

**What B1 decides.** The absolute ITL of a half-card engine is not predicted.
No run has measured decode on 152 CUs, and decode reads memory that both halves
still share. The reading is the ratio:

| B1 ratio | Reading | Next |
|---|---|---|
| ≤ 1.10, either port | The asymmetry needs two processes on the same CUs. The CP's workgroup dispatch between processes is the candidate left, not a proven mechanism | §4; B2 is not needed |
| ≥ 1.6, the first launched slow | Not the CUs. What both halves still share is left: L2, fabric, memory, or the runlist's time-slicing | B2, then §4 |
| between | Partly the CUs | B2 if the clock allows, then §4 |

**B2** holds each process to one hardware queue. If `dmesg-*.txt` shows
"Runlist is getting oversubscribed" for B0, this is the step that should remove
it, and the ratio should fall towards 1. If the ratio stays at B0's, the queues
are ruled out too. What remains then needs a profiler that sees per-process
GPU time, which is a run of its own (§7).

**What `hostrec.sh` records per paired call:**
- `rocm-smi` every 5 s, device-wide;
- `/sys/kernel/debug/kfd/mqds` and `/sys/kernel/debug/kfd/rls`, the queues and
  the runlist, once at the call's start;
- the `dmesg` lines the call added, filtered for KFD and runlist lines.

`host.txt` adds the module's `sched_policy` and `hws_max_conc_proc`.

---

## 4 · Checkpoint C — the 70B launch

```
bash /workspace/i-serve/run5/s5.sh        # container; or s5.sh 4 64 if §6 dropped c = 16
```

`s5.sh` stops whichever pair is live, waits for the card to be empty and
completes the 70B download if `s1.sh` could not. Then it
launches InferenceX's server line at TP = 1, with three changes:
- no `--swap-space`, which `v0.27.1` rejects;
- **one server at `--max-num-seqs 64`** for every level, where InferenceX
  restarted per level with `$CONC`, because the client's cap holds the running
  count to c anyway;
- the image's AITER default, off.

| Log line | Predicted | On a miss |
|---|---|---|
| `GPU KV cache size` | **≤ 739 334** tokens (`ceiling.py`; an over-estimate, since activations at a 131 072-token budget are not modelled) | below 589 824 = 64 × 9 216, c = 64 can preempt. Read `claim-c64-metrics-*.txt` and keep the level |
| backend line | recorded, not gated | a startup failure that names the KV dtype or the backend: §8 |
| `kv_cache_dtype` | `fp8` | `auto` means the flag was not obeyed: the ceiling is `ceiling.py`'s BF16-KV rows |

---

## 5 · Block C — InferenceX's claim at TP = 1 (~35 min)

The claim's 50 tok/s per user means a decode step of at most 20 ms. On this
card, at TP = 1, the weights alone are 72.66 GB per step.

**The ceiling, before the card** (`python3
docs/benchmarks/raw/mi300x-2026-10-17/run5/ceiling.py`, at a mean context of
7 834 tokens: 7 373 in plus half of 922 out). Decode only: prefill stalls and
TTFT are not in it, so every measured figure should fall below it.

| c | TPOT floor, `mi300x-run2` (eff_mem 0.57) | ≤ tok/s per user | ≤ tok/s per GPU | ≤ tok/s per user at the 0.70 prior |
|---|---|---|---|---|
| 4 | 25.7 ms | **38.8** | 155 | 47.7 |
| 16 | 30.8 ms | **32.4** | 519 | 39.8 |
| 64 | 51.2 ms | **19.5** | 1 249 | 24.0 |

Two derived statements face the claim before anything is measured:

- **On run 2's coefficient, TP = 1 never reaches 50 tok/s per user.** Batch 1,
  reading weights only, tops out at 41.6.
- **Even at 100 % of peak bandwidth**, which no run has measured, 50 tok/s per
  user holds only up to batch 25. That is ≤ 1 250 tok/s per GPU against the
  claim's 1 482. So the claim's headline point cannot come from TP = 1 on this
  card. It comes from TP ≥ 2, where each GPU reads only part of the weights per
  step.

What the card tests is the coefficient. **If c = 4 reads above 38.8 tok/s per
user, run 2's `eff_mem` does not transfer from Qwen3-8B BF16 to this model's
FP8 kernels**, and the share of ceiling is the number to report. The threshold
errs on the generous side. Both models' memory sides count an embedding that a
decode step never reads, 7.6 % of Qwen3-8B's bytes against ≈ 2.9 % of this
model's. Counting only the bytes read lowers the c = 4 bound by ≈ 5 %, to
≈ 36.9. Per-user
figures are 1 / median TPOT; per-GPU figures are `output_throughput` on one GPU.

**The load:**
- `vllm bench serve --ignore-eos --save-detailed`, 7 373 ± 11.1 % in and
  922 ± 11.1 % out (§0);
- c ∈ {4, 16, 64}, `--seed 0`;
- **5 × c prompts per level**, where InferenceX sent 10 × c. At c = 64 the
  prefill alone is ≈ 2.4 M tokens, and the clock decides it;
- `num_preemptions` read off `/metrics` before and after each level.

Each level appends one line to `claim.txt`.

---

## 6 · Budget, drop order, and what "done" means

| | |
|---|---|
| Expected clock | ≈ 1.6 h: 20 min up, 20 min block B, 10 min the 70B launch, 35 min block C, the rest harvest and destroy |
| Hard stop | **2.5 h ≈ $5**, and **23:00 UTC** on 2026-10-17, whichever comes first |
| Drop order | B2 first, then c = 16, **decided before `s5.sh` starts**, because a later drop relaunches the 70B: `bash /workspace/i-serve/run5/s5.sh 4 64`. **Never B0, B1 or c = 4**: B0 is B1's control, and c = 4 is where the per-user ceiling is faced |
| Done | B0 and B1 read against §3's tables, with the mask verdict; c = 4 and c = 64 read against §5's ceiling |
| Price at creation | _read off the create page_ |

Everything lands in `docs/benchmarks/raw/mi300x-2026-10-17/run5/` beside the
scripts, with its README. The README is read once for a credential before
staging, and the file count is checked across `git add` (`docs/adding-a-run.md`
§2). If the run takes place on another date, the directory is renamed to it.

---

## 7 · Not in this run, and where each goes

- **Per-process GPU time on a shared card**, if B1 and B2 leave the mechanism
  open. That needs `rocprofv3` or an equivalent tracer in the image: a sheet of
  its own.
- **InferenceX at TP ≥ 2.** It needs a multi-GPU droplet, the capacity this run
  could not get.
- **Everything in run 4's §7**: the decode-step bill, the 0.04 at N = 31, Zipf,
  two cards.

---

## 8 · If it goes sideways

| Symptom | First move |
|---|---|
| `stage` reads `TOOL_MISSING <tool>` | `apt-get install -y procps` in the container, then `s1.sh` again |
| `download-status.txt` reads `DOWNLOAD_FAILED` | Nothing to do before §4: `s5.sh` fetches again. If `stage` then reads `DOWNLOAD_FAILED`, `hf-download-70b.txt`'s tail; block C is dropped |
| `stage` reads `GATE_<what>_<port>` or `E<port>_DIED` | run 4 §8 |
| `stage` reads `MASK_NOT_TAKEN` | `env-masked.txt` shows whether the variable reached every process. If it did, ROCr ignored it: B2, then §4 |
| `stage` reads `MASK_GATE_FAILED` | `droplet-check.txt` and `droplet-check-masked.txt`. A missing baseline: `. /workspace/i-serve/run5/lib.sh; stop 8000 8001`, relaunch the plain 8001 alone, rerun the check, then `s3.sh` |
| `stage` reads `PAIR_FAILED <tag>` | `<tag>-800x-harness.txt`'s tail; the pair line was not written |
| `stage` reads `E70B_DIED`, and the log names the KV cache dtype or the attention backend | `C_ENV=VLLM_ROCM_USE_AITER=1 bash /workspace/i-serve/run5/s5.sh`. A second failure: `C_FLAGS="--kv-cache-dtype auto"`, read against `ceiling.py`'s BF16-KV rows. Each rerun waits at `C_WAIT_CARD`: an orphaned EngineCore shows there |
| `stage` reads `E70B_DIED` with an out-of-memory line | `C_FLAGS="--max-num-batched-tokens 32768" bash /workspace/i-serve/run5/s5.sh`, written into the README as a change from InferenceX's line |
| `stage` reads `C_LEVEL_FAILED <c>` | `claim-c<c>.txt`'s tail. The levels already written stand; the engine log says whether to relaunch |
| The clock passes 2 h, or 22:30 UTC | Harvest what exists and destroy the droplet |
