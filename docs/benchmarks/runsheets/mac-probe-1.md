# Runsheet — Mac probe 1: is the ceiling informative?

This sheet rests on one bet: a decode ceiling computed from the bytes a step
reads, divided by the device's peak bandwidth, says something about what an
engine actually delivers. It registers that ceiling for Qwen3-8B on the
author's Mac, for two weight files and three engines: the first single-stream
probe here, on a laptop rather than a rented card. **The ceiling was registered
on 2026-10-25 before attempt 0 began at 13:23 UTC**, in a private tree; this
public copy arrives after the fact, so the order rests on the logs' timestamps
(`raw/mac-2026-10-25/*/steps.log`) and the author's word.

**Status: stopped 2026-10-25, §5 not read.** §0–§3 were done and the
measurement was attempted twice (§4): attempt 0 failed on Ollama's repeat
limit, attempt 1 was stopped by the author under thermal throttling. No further
measurement runs on this Mac until a thermal protocol exists (§4). The engines,
the weights and Ollama's copy were deleted the same day, before §5 was read,
overriding §6's keep-until-read rule; the raw logs of both attempts are kept.
Nothing in §5 is filled. The bet itself was read instead on the rented rows:
`docs/benchmarks/ledger.jsonl` holds every decode step the L40S and MI300X runs
measured, beside its ceiling.

**Cost: 0 credit-hours.** Everything runs on the Mac. Disk: ≈ 5.0 GB (GGUF) +
4.6 GB (MLX) + 5.0 GB (Ollama's copy of the GGUF) ≈ 15 GB; 602 GiB free on
2026-10-25.

**Where things live.** The raw logs of both attempts:
`docs/benchmarks/raw/mac-2026-10-25/`, with a README of its files.

---

## What this run is not

- **Not a quality comparison.** Q4_K_M and MLX 4-bit are different bits on
  different source weights (§0); nothing here says which answers better.
- **Not prefill.** Prefill speeds are printed by every tool and logged, but no
  ceiling is registered for them; they face nothing.
- **Not vLLM.** No CUDA or ROCm device is in hand; vLLM is the rented
  cards' engine (`docs/benchmarks/`).
- **Not a $/task calculation.** Nothing here prices a task.

---

## 0 · Pins (done 2026-10-25)

### Device

| | |
|---|---|
| Machine | MacBook Pro 16″, `Mac14,6`, Apple M2 Max, 38-core GPU, 32 GB unified memory |
| Peak memory bandwidth | **400 GB/s** (Apple, M2 Pro/Max announcement, 2023-01-17) — decimal GB, the only bandwidth figure in §1 |
| macOS | 26.3.1 (a), build 25D771280a — re-read on the day (§3) |

### Model and files

| | Repository @ revision | File, bytes | sha256 |
|---|---|---|---|
| Source | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | BF16 safetensors (not downloaded) | — |
| GGUF | `Qwen/Qwen3-8B-GGUF` @ `7c41481f57cb95916b40956ab2f0b139b296d974` | `Qwen3-8B-Q4_K_M.gguf`, 5 027 783 488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| MLX | `mlx-community/Qwen3-8B-4bit` @ `545dc4251c05440727734bcd94334791f6ab0192` | `model.safetensors`, 4 607 835 174 | `f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8` |

Two facts read from the files' own headers:

- **The GGUF is not converted from the pinned BF16 revision.** Its metadata
  reads `general.name = Qwen3 8B Awq Compatible Instruct`,
  `general.finetune = awq-compatible-Instruct`. Its architecture fields match
  `config.json` exactly (36 blocks, 32 heads, 8 KV heads, key and value length
  128, context 40 960), so the bytes and the ceiling are unaffected. Output
  quality could be, and §4 does not measure it.
- **The MLX file is mlx-community's, not Qwen's own `Qwen/Qwen3-8B-MLX-4bit`.**
  mlx-community's uses group size 64, which is `mlx_lm.convert`'s default and
  the file an earlier hand calculation's 93.9 tok/s was computed from (§1
  reproduces it to the digit). Qwen's own uses group size 128: 4 351 884 216 B, and a
  zero-context ceiling of 99.5 tok/s. Choosing it would be a second row, not a
  correction.

### Engines

| Engine | Version | Source commit | Install |
|---|---|---|---|
| llama.cpp | **b11845** (2026-10-23) | `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e` | built from source, Metal (§2, *Which llama.cpp step A runs*) |
| Ollama | `v0.40.11` (2026-10-24) | `675150d6ad2747df1e218ef59de9194bc4a7e538` | Homebrew, built from source |
| ↳ its `llama-server` | llama.cpp b11845 | `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e` | fetched and built by Ollama's own CMake: the same commit as step A |
| mlx-lm / mlx | `0.32.2` (2026-10-16) / `0.32.4` (2026-10-14) | PyPI | `uv tool` |

llama.cpp now tags semantic versions (`v0.7.0`, the Homebrew bottle) beside
the `bNNNNN` builds, which GitHub marks as pre-releases. Step A runs b11845, the
build inside Ollama, not the bottle (§2). The local `brew info` was stale on
2026-10-25 (Ollama 0.40.6): `brew update` comes first in §3.

---

## 1 · The ceiling (done 2026-10-25)

One decode step reads every weight it multiplies by, plus the whole KV cache it
attends over. The token embedding is a lookup: one row per step, 2 304 B for
Q4_K, so it is left out. The output projection (`output.weight`, `lm_head`) is
a full matrix multiply and stays in.

```
ceiling_tok_s(c̄, B) = B × 400e9 / (W + B × 147 456 × c̄)
```

- `W`: weight bytes read per step, from the file's tensor table.
- `147 456`: KV bytes per token = 2 × 36 layers × 8 KV heads × 128 × 2 B,
  `docs/SLO.md` §3, traced through Raschka's `standalone-qwen3-plus-kvcache.ipynb`
  on 2026-10-25. The 2 B is an F16/BF16 cache, the default in all three
  engines (§2).
- `c̄`: mean context over the timed window, prompt + 64 for a 128-token
  generation.
- `B`: concurrent sequences.

### W, per file, by tensor role

| GGUF Q4_K_M | Type | Bytes |
|---|---|---:|
| `ffn_gate`, `ffn_up` | Q4_K | 2 038 431 744 |
| `ffn_down` | Q6_K / Q4_K (layer by layer) | 1 252 786 176 |
| `output` | Q6_K | 510 504 960 |
| `attn_q`, `attn_output` | Q4_K | 679 477 248 |
| `attn_k` | Q4_K | 84 934 656 |
| `attn_v` | Q6_K / Q4_K | 104 398 848 |
| norms | F32 | 1 232 896 |
| **W** (tensors 5 021 827 072 − `token_embd` 350 060 544) | **4.938 bits/weight** | **4 671 766 528** |

The remaining 5 956 416 B of the file are the header: metadata, the tokenizer
and alignment. A step does not read them.

| MLX 4-bit, group 64 | Type | Bytes |
|---|---|---:|
| layer weights | U32 (packed 4-bit) | 3 472 883 712 |
| layer scales + biases | BF16 | 434 110 464 |
| layer norms | BF16 | 616 448 |
| `lm_head` (weight + scales + biases) | U32 + BF16 | 350 060 544 |
| **W** (tensors 4 607 731 712 − `embed_tokens` 350 060 544) | **4.500 bits/weight** | **4 257 671 168** |

### Registered ceilings

| File | Point | `c̄` | KV share of the step | Ceiling, tok/s | Step, ms |
|---|---|---:|---:|---:|---:|
| GGUF | short, B = 1 | 576 | 1.8 % | **84.09** | 11.89 |
| GGUF | 4 K, B = 1 | 4 160 | 11.6 % | **75.68** | 13.21 |
| GGUF | batch, B = 4 | 576 | — | **319.3** (79.8 per sequence) | 12.53 |
| MLX | short, B = 1 | 576 | 2.0 % | **92.11** | 10.86 |
| MLX | 4 K, B = 1 | 4 160 | 12.6 % | **82.12** | 12.18 |
| MLX | batch, B = 4 | 576 | — | **348.0** (87.0 per sequence) | 11.49 |

Where a row's measured prompt length differs from the nominal one (Ollama
reports its own count), its ceiling is the formula at that row's `c̄`, not the
number in this table.

**Against the earlier hand calculation.** At zero context it gave 85.5 and
93.9 tok/s. Here they are 85.62 and 93.95. The GGUF gap is the header: it subtracted
the embedding from the file size, so the tokenizer was counted as read every
step. The MLX figure reproduces exactly.

**Why B = 4 is still bandwidth-bound.** A step at B = 4 is ≈ 2 × 6.95e9 × 4 =
56 GFLOP against ≈ 12.5 ms of memory time. No dense FP16 figure for the M2 Max
GPU is published by Apple, so none is registered here. The cost of
dequantising inside the kernels is the named suspect if the batch rows fall
further below the ceiling than the B = 1 rows (§6).

---

## 2 · The external layer (read 2026-10-25)

Each default was read from source at the pinned commit, not from a blog:
llama.cpp `620d222` (v0.7.0), Ollama `675150d`, mlx-lm `93e6d4a`. After the decision
below moved step A to b11845, every llama.cpp file read here was diffed against
b11845. `llama-bench.cpp`, `batched-bench.cpp` and `ggml-metal-device.m` are
identical. `common.h`, `llama-kv-cache.cpp` and `llama-context.cpp` changed
only for other architectures and server features, so the reading holds for
b11845. **No engine quantises the KV cache by default, so
the 2 B in §1 stands for all three, and no ceiling changes.** Two findings
change how a row is read (the first-token bias, below), and one changes what
the llama.cpp-against-Ollama comparison means (the build gap, below).

### What each engine does by default

| | llama.cpp `llama-bench` | llama.cpp `llama-batched-bench` | Ollama | mlx-lm `mlx_lm.benchmark` |
|---|---|---|---|---|
| Runs on | llama.cpp b11845 | llama.cpp b11845 | **upstream `llama-server`, build b11845**, started as a subprocess | mlx-lm on MLX |
| KV type | F16 (`type_k`, `type_v`) | F16 (`common.h`) | F16: `OLLAMA_KV_CACHE_TYPE` unset → no `--cache-type-*` flag → `llama-server`'s F16 | BF16: the keys' dtype, which is the model dtype, taken from the scales (`utils.py`); `kv_bits` None |
| Flash attention | AUTO | AUTO | `--flash-attn auto` on Metal, unless `OLLAMA_FLASH_ATTENTION` is set | MLX's `scaled_dot_product_attention` |
| KV read per step | the used cells, **padded up to a multiple of 256** (`get_n_kv`) | as `llama-bench` | as `llama-bench` | exactly `offset` cells (`keys_and_values`) |
| Layers on GPU | all (−1) | all | `-ngl` left to `llama-server`'s auto unless `num_gpu` is set | all (unified memory) |
| Prompt | none, then random tokens; the depth prefill is **not** timed and is restored from a saved state after the first rep | not shared (`-pps` off) | the text sent; `raw: true` skips Ollama's template | random tokens, `mx.random.seed(0)` |
| EOS | not checked (random tokens fed back) | not checked | **honoured**: no `ignore_eos` | disabled (`_eos_token_ids = {}`) |
| Warm-up | one prompt and one token | none | none: the script's discarded request | one full run |
| Repetitions | 5 | none: §4 runs it three times | the script's 5 | 5 |
| Concurrency | 1 | `-npl` | `-np` = `OLLAMA_NUM_PARALLEL`, default 1; `-c` = `num_ctx` × that | `-b` |

- **AUTO flash attention resolves by a probe.** It is switched on if the
  device that runs the layer also supports the fused op. Metal supports it at
  head size 128 (`ggml-metal-device.m`), so it should resolve to *enabled*.
  `llama-bench` silences that log line unless `-v` is given, so §4 passes
  `-v`. Ollama's server log prints it without asking.
- **The 256-cell padding costs ≈ 28 MB a step** at both points: 768 cells read
  against 576 on average, and 4 352 against 4 160. That is 0.6 % and 0.5 % of
  the step. It is the engine's overhead, not the model's bytes, so it goes into
  the gap, not into the ceiling.
- **Ollama's KV pool is sized for `num_ctx`, but each step reads only the used
  cells.** The 8 192 set in §4 costs memory, not bandwidth.

### The first-token bias: two of four tools overstate decode by 128/127

The first generated token comes from the prefill's last logits, so the
decode window of an n-token generation is n − 1 steps long.

| Tool | Tokens counted | Time counted | Reported ÷ true |
|---|---|---|---|
| `llama-bench` | 128 | 128 decode steps | 1 |
| `llama-batched-bench` | B × 128 | 128 decode steps | 1 |
| Ollama | `eval_count` = `predicted_n` = n, counting the free token | `eval_duration` = `predicted_ms`, from the end of the prompt to the last token: n − 1 steps | **128/127 = 1.008** |
| `mlx_lm.benchmark`, B = 1 | n + 1 at index n, so 128 | from the first token to the last: 127 steps | **128/127 = 1.008** |
| `mlx_lm.benchmark`, B = 4 | the tokens decode steps return | decode time only | 1 |

`llama-server` knows it: its own `predicted_per_token_ms` divides by n − 1.
Ollama passes the raw pair through. **Corrected in the read-out:** the Ollama
script now computes `(eval_count − 1) / eval_duration`, and §5 multiplies
MLX's B = 1 `generation_tps` by 127/128. The raw value is kept beside it.

### The build gap: llama.cpp is not one version here

Ollama v0.40.11 builds llama.cpp at **b11845** (`LLAMA_CPP_VERSION`; commit
`9ff0d7e`, 2026-10-23). That is 176 commits after `v0.7.0`, the one
Homebrew bottles. Its own patch (`llama/compat/001-llama-cpp-hooks.patch`) adds
load-time hooks to `llama-model-loader.cpp` and `clip.cpp` only, and no kernel
code. So at b11845 the GGUF runs through the same kernels in both, and the
only difference is Ollama's layer: HTTP, per-token sampling in
`llama-server`'s loop, slot management. Against v0.7.0, a gap mixes that layer
with 176 commits of kernels. **Decided below, before §4.**

**An earlier reading of Ollama at 28 % of ceiling against llama.cpp's 64 % on
one file is probably about an older Ollama.** It predates
the move to `llama-server`, when Ollama ran its own Go runner on ggml. It stays
a suspect to test, not a prediction for v0.40.11.

### Still open, readable only on the day

- [ ] AUTO flash attention resolved to *enabled* in all three llama.cpp logs.
- [ ] `ollama ps` reads 100 % GPU; `llama-server`'s `-ngl auto` put every layer
      on Metal (its log).
- [ ] Every Ollama row has `eval_count` = 128, i.e. no early EOS.
- [x] **Homebrew's service is not this run's Ollama** (read 2026-10-25). The
      formula's `service do` block sets `OLLAMA_FLASH_ATTENTION=1` and
      `OLLAMA_KV_CACHE_TYPE=q8_0`, so `brew services start ollama` quantises
      the KV cache to ≈ 1.06 B per element, and its rows would need their own
      ceiling. §3 starts `ollama serve` by hand with neither variable set, and
      the F16 reading above holds for it.
- [x] **A shallow clone prints build 1.** `llama-bench --version` reads
      `0.7.0-dev (build 1, commit 9ff0d7e)`: the build number counts commits,
      and `--depth 1` has one. The commit is the identity, and it matches §0.
- [ ] **The Mac.** Power mode (`pmset` `powermode`: 0 automatic, 1 low,
      2 high; 0 on 2026-10-25), and what the GPU clock does under a sustained
      decode load — read from `powermetrics` during the run, not assumed.

### Which llama.cpp step A runs — decided 2026-10-25: b11845, built from source

| | A on the Homebrew bottle, v0.7.0 | A built from source at b11845 |
|---|---|---|
| llama.cpp against Ollama measures | Ollama's layer **plus** 176 commits of kernels | Ollama's layer alone: the premise of §4 |
| Install | `brew install llama.cpp` | `brew install cmake`, then a clone and a CMake build of `llama-bench` and `llama-batched-bench` (Metal is on by default, and the shaders are compiled at run time, so Xcode's `metal` tool is not needed) |
| Cost | none | one CMake build on the CPU; 0 credit-hours |

Decided: **build at b11845.** Without it, §4's question "is the gap the
engine?" has no clean answer. The bottle is not run.

---

## 3 · Before the first measurement

- [ ] **The decision without the probe, written before anything is
      installed.** What the author would choose without it, and why:

      > **No prior (2026-10-25).** The author has never run a local engine
      > and holds no view on which is better. Without the probe, the author
      > would take the path of least effort: **Ollama**, one install and
      > a model pulled by name. That is a default by convenience, not a
      > judgement, and it is the baseline the probe is measured against.
      >
      > **What would move it.** Another engine is chosen instead if it
      > delivers ≥ 10 % more decode tok/s than Ollama at the 4 K point,
      > its spread resolving the difference (§5). The 4 K point is the one
      > that resembles real use. Otherwise Ollama stays, and the probe has
      > confirmed the default, not changed it.

      Written from the author's statement that there is no opinion yet. A
      no-prior baseline is weaker evidence than a reasoned one: a probe can
      only confirm or replace a default, not overturn an argument.

- [ ] **Install and freeze the engines:**
      ```bash
      brew update
      brew install cmake ollama
      brew pin ollama
      uv tool install mlx-lm==0.32.2 --with mlx==0.32.4

      S=~/src/llama.cpp-b11845
      git clone --depth 1 --branch b11845 https://github.com/ggml-org/llama.cpp $S
      cmake -S $S -B $S/build -DCMAKE_BUILD_TYPE=Release
      cmake --build $S/build --config Release -j --target llama-bench llama-batched-bench
      L=$S/build/bin

      git -C $S rev-parse HEAD; $L/llama-bench --version
      ollama --version; uv tool list | grep -A3 mlx-lm
      ```
      The versions and the commit printed must match §0. If they do not, stop
      and amend §0.
- [ ] **Download the pinned files:**
      ```bash
      M=~/models/qwen3-8b; mkdir -p $M
      uvx --from huggingface_hub hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf \
          --revision 7c41481f57cb95916b40956ab2f0b139b296d974 --local-dir $M/gguf
      uvx --from huggingface_hub hf download mlx-community/Qwen3-8B-4bit \
          --revision 545dc4251c05440727734bcd94334791f6ab0192 --local-dir $M/mlx
      shasum -a 256 $M/gguf/Qwen3-8B-Q4_K_M.gguf $M/mlx/model.safetensors
      ```
      Both digests must match §0.
- [ ] **Import the same GGUF into Ollama:**
      ```bash
      ollama serve > docs/benchmarks/raw/mac-2026-10-25/ollama-serve.log 2>&1 &
      echo "FROM $M/gguf/Qwen3-8B-Q4_K_M.gguf" > /tmp/Modelfile.qwen3-8b
      ollama create qwen3-8b-q4km -f /tmp/Modelfile.qwen3-8b
      ls ~/.ollama/models/blobs | grep d98cdcbd03e17ce4
      ```
      The blob's digest must be the file's: if the import re-quantised or
      rewrote anything, the "same file" premise of §4 is gone.
- [ ] **Record the state**, into `docs/benchmarks/raw/mac-2026-10-25/state.txt`:
      ```bash
      R=docs/benchmarks/raw/mac-2026-10-25
      { date -u; sw_vers; pmset -g; pmset -g batt; pmset -g therm; } > $R/state.txt
      ```
      In a second terminal, for the whole of §4:
      `sudo powermetrics --samplers gpu_power,thermal -i 1000 -o $R/powermetrics.log`

      **Amended 2026-10-25, before the first measurement:** `sudo` asked for
      a password the run did not have, so `powermetrics` is replaced by two
      sources that need none. `macmon` 0.10.0 (Homebrew) reads Apple's
      IOReport counters once a second: GPU frequency, GPU active ratio, GPU
      and DRAM power, GPU temperature, fans, as JSON. Thermal pressure comes
      from the system notification key (`0` is nominal):
      ```bash
      macmon pipe -i 1000 > $R/macmon.jsonl &
      while :; do echo "$(date -u +%FT%TZ) $(notifyutil -g com.apple.system.thermalpressurelevel)"; sleep 5; done > $R/thermal.log &
      ps -Ao pcpu,rss,comm -r | head -15 > $R/background-load.txt
      ```
      What this cannot see that `powermetrics` would: the GPU's per-state
      frequency residency (only the sampled mean frequency is logged).
- [ ] **Quiet the machine.** On AC power, lid open, other GPU apps closed
      (browser tabs with video, IDE previews); Spotlight indexing finished.
      Only one engine holds the model at a time: `ollama stop qwen3-8b-q4km`
      before the llama.cpp and MLX steps.

---

## 4 · The measurement

Three rounds, each A → B → C, so that drift over the session shows up as a
round effect rather than as an engine difference. Each step writes to its own
file in `$R`, named with the round number `$N`.

**A · llama.cpp**, the GGUF, single stream at both points, then the batch point:
```bash
$L/llama-bench -m $M/gguf/Qwen3-8B-Q4_K_M.gguf -p 0 -n 128 -d 512,4096 -r 5 -o jsonl -v \
    > $R/llama-bench-r$N.jsonl 2> $R/llama-bench-r$N.log
$L/llama-batched-bench -m $M/gguf/Qwen3-8B-Q4_K_M.gguf -c 4096 -npp 512 -ntg 128 -npl 1,4 \
    > $R/batched-bench-r$N.log 2>&1
```

**B · Ollama**, the same GGUF, single stream at both points:
```bash
python3 docs/benchmarks/raw/mac-2026-10-25/ollama_decode.py 508 4092 > $R/ollama-r$N.jsonl
ollama ps >> $R/ollama-r$N.jsonl; ollama stop qwen3-8b-q4km
```
The two arguments are filler-word counts. The resulting prompt lengths are in
each row's `prompt_eval_count`, and the row's ceiling is computed from them.
Ollama has no batch point: concurrency there is `OLLAMA_NUM_PARALLEL`, which
is a server restart, and one batch point is enough for the read-out.

**C · MLX**, its own file, single stream at both points, then the batch point:
```bash
for P in 512 4096; do
  mlx_lm.benchmark --model $M/mlx -p $P -g 128 -n 5 > $R/mlx-p$P-r$N.log 2>&1
done
mlx_lm.benchmark --model $M/mlx -p 512 -g 128 -b 4 -n 5 > $R/mlx-b4-r$N.log 2>&1
```

**Attempt 0, aborted (2026-10-25, 13:23–13:25 UTC).** Round 1 step A ran;
step B failed on Ollama's repeat limit (§6); a diagnostic request then ran
while step C was loading MLX, so C is contaminated. Its logs are kept in
`raw/mac-2026-10-25/attempt-0-aborted/` and are not read in §5. The three
rounds were restarted from round 1.

**Attempt 1, stopped by the author (2026-10-25, 13:26–13:30 UTC).** Round 1
steps A and B completed, C was cut short. The GPU ran at 93–99 °C, thermal
pressure reached level 2 (heavy), and the GPU clock fell from 1 398 to
1 279 MHz during step A: a −8.5 % swing, larger than §5's 5 % resolution.
Logs, including `macmon.jsonl` and `thermal.log` for both attempts, are in
`raw/mac-2026-10-25/attempt-1-stopped/` and are not read in §5. Open before the
next attempt: a cool-down rule between steps, so that thermal state is
controlled rather than only logged.

Recorded beside each row in §5: engine version, date, macOS version, power
source and mode (from `state.txt`), GPU frequency and thermal pressure over the
row's window (from `macmon.jsonl` and `thermal.log`, §3).

---

## 5 · The read-out

Per row: median over 3 rounds × 5 repetitions, the spread (min–max), and the
share of ceiling = median ÷ ceiling. A gap larger than the spread gets a named
mechanism, or "not explained". Ollama's rows use the script's corrected
`tok_s`. MLX's B = 1 rows use `generation_tps` × 127/128 (§2, the first-token
bias), with the raw value beside them.

| Engine | File | Point | Ceiling | Median | Spread | Share | Mechanism |
|---|---|---|---:|---:|---:|---:|---|
| llama.cpp | GGUF | short | 84.09 | | | | |
| llama.cpp | GGUF | 4 K | 75.68 | | | | |
| llama.cpp | GGUF | B = 4 | 319.3 | | | | |
| Ollama | GGUF | short | per row | | | | |
| Ollama | GGUF | 4 K | per row | | | | |
| MLX | MLX | short | 92.11 | | | | |
| MLX | MLX | 4 K | 82.12 | | | | |
| MLX | MLX | B = 4 | 348.0 | | | | |

**The slope test.** Per engine, the ratio of the 4 K speed to the short one is
predicted at 0.900 for the GGUF and 0.892 for MLX. That ratio tests the KV
term of §1 with the bandwidth cancelled out. A ratio far from it means the
engine's step cost does not grow with context the way bytes do.

**The bet's two clauses, written before any measurement:** the median share of
ceiling ≥ 0.5, and the spread between repetitions smaller than the difference
between engines.

**The second clause, as read here (decided 2026-10-25):** each row's spread (max − min over its 15 values) is at most
5 % of its median. The instrument then resolves an engine difference of 5 % or
more, and two engines closer than that read "not distinguishable", which is a
valid result, not a failure. Read literally, the first wording would fail
whenever two engines are genuinely equal, however good the measurement.

---

## 6 · Risks

- **Thermals and background load read as engine differences.** Rounds are
  interleaved, and the GPU clock is logged per row, for this reason.
- **The tools' defaults are decisions too**: warm-up, prompt cache, random
  versus real tokens, EOS handling. §2 reads them; the Ollama script makes
  every prompt distinct so that its prompt cache cannot shorten the prefill.
- **Ollama can stop early** at an end-of-sequence token: it has no
  `ignore_eos`. `eval_count` is logged, and a row with fewer than 128 tokens
  is flagged, not dropped.
- **Ollama aborts a run of identical tokens (found 2026-10-25, attempt 0).**
  The first filler, `" hello"` repeated, made greedy decoding emit `hello`
  over and over, and Ollama returns HTTP 500 "prediction aborted, token repeat
  limit reached" after 100 identical tokens in a row (`llm/llama_server.go`,
  v0.40.11; hard-coded, no option). The filler is now ten distinct words
  cycled; tested at both points, 128 tokens each. The ceiling does not depend
  on the prompt's content, so nothing in §1 moves.
- **Different bits per weight.** The GGUF reads 4.938 bits per weight and the
  MLX file 4.500, so the two ceilings differ by design. Only llama.cpp against
  Ollama compares engines directly; MLX compares by share.
- **Dequantisation is not in the ceiling.** If it costs time at B = 1, every
  share falls together and the slope test still holds. If it bites only at
  B = 4, the batch rows show it.
- **Deleting the files too early.** The weights, Ollama's copy and the
  llama.cpp build stay on disk until §5 is read: a failed spread clause means
  a quiet rerun on the same pinned bytes, and a later probe may run the same
  engines on the Mac. Once §5 is read and no probe needs them:
  ```bash
  ollama rm qwen3-8b-q4km           # Ollama removes its own blob
  rm -rf ~/models/qwen3-8b ~/src/llama.cpp-b11845
  rm -rf ~/.cache/huggingface/xet   # look inside first: it predates this run
  brew unpin ollama && brew uninstall ollama macmon cmake
  uv tool uninstall mlx-lm
  ```
  The raw logs in `docs/benchmarks/raw/mac-2026-10-25/` are not deleted:
  they are the record of the order.
