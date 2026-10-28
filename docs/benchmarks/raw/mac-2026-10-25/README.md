# Raw evidence — Mac probe 1, 2026-10-25

Verbatim capture, no interpretation. The sheet, with the record of both
attempts, is `docs/benchmarks/runsheets/mac-probe-1.md` §4; its read-out (§5)
was never filled, so nothing in `bench/` reads these files yet.

| Path | What it is |
|---|---|
| `ollama_decode.py` | step B's client: one JSON line per request, the first-token bias corrected (sheet §2) |
| `attempt-0-aborted/` | round 1: step A complete, step B failed on Ollama's repeat limit, step C contaminated by a diagnostic request |
| `attempt-1-stopped/` | round 1: steps A and B complete, step C cut short at thermal pressure level 2 |
| `*/steps.log` | the start and end of each step, UTC |
| `*/llama-bench-r1.jsonl`, `*/llama-bench-r1.log` | step A: `-o jsonl` and the verbose log, which carries the flash-attention line |
| `*/batched-bench-r1.log` | step A's batch point |
| `*/ollama-r1.jsonl` | step B: one line per request, then `ollama ps` |
| `*/mlx-*-r1.log` | step C; a zero-byte file is a step cut before it printed |
| `attempt-1-stopped/macmon.jsonl` | GPU frequency, active ratio, power and temperature once a second |
| `attempt-1-stopped/thermal.log` | the thermal pressure level every five seconds |
| `attempt-1-stopped/state.txt`, `background-load.txt` | macOS version, power settings and source; the busiest processes before the run |
| `attempt-1-stopped/ollama-serve.log` | the server's log for the session |
