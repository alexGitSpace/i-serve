# Raw evidence — MI300X run 4, 2026-10-17

Verbatim capture, no interpretation. One run, one droplet:

| Directory | Report | Read-out | Runsheet |
|---|---|---|---|
| `run4/` | `docs/benchmarks/mi300x-run4.md` | `bench/measured_mi300x_run4.py` | `docs/benchmarks/runsheets/mi300x-run-4.md` |

## Run 4

| Path | What it is |
|---|---|
| `run4/s1.sh` … `run4/blockb.sh`, `run4/lib.sh`, `run4/arm.sh`, `run4/hostrec.sh`, `run4/pin.sh` | the scripts, committed before the card and run unedited; `pin.sh` was not run, block 0 did not take its branch |
| `run4/engine-8000-cold.log`, `run4/engine-8001-cold.log` | the first launch of the pair, 8000 first, on an empty `torch.compile` cache: 8000 compiled the graph and logged the smaller pool; stopped after checkpoint A |
| `run4/engine-8000.log`, `run4/engine-8001.log` | the pair relaunched, 8000 first, both loading the compiled graph: blocks 0a and 0b |
| `run4/engine-8001-r2.log`, `run4/engine-8000-r2.log` | block 0c's relaunch, 8001 first: 0b again, then all of block B |
| `run4/checkpoint-a*.txt`, `run4/reset-gate*.txt` | each launch's checkpoint-A lines and the idle `POST /reset_prefix_cache` answer per engine; `-cold` is the first launch |
| `run4/droplet-check*.txt`, `run4/results/check*/` | `fleet-bill`'s c = 1 level against 8001, once per launch of `s1.sh` |
| `run4/results/alone-800x/` | block 0a: `fleet-bill-half` on one engine, the other idle |
| `run4/results/together-<order>-800x/` | block 0b: `fleet-bill-half` on both engines at once; `<order>` names the engine launched first |
| `run4/results/<arm>-n<N>[-bl1]/` | one block-B level; `-bl1` is `-bounded-load 1`, the bound off |
| `run4/results/*/<level>.json`, `-metrics.json`, `-requests.jsonl`, `run.json` | the harness's per-level summary, Prometheus increments, one record per measured request, and the call |
| `run4/*-harness.txt`, `run4/arm-*.txt` | what each harness call printed |
| `run4/top-<order>.txt` | `top -b -H -d 5` over both engines' process trees during that 0b call: `%CPU` per thread per interval |
| `run4/psr-<order>.txt` | `ps -L -o pid,tid,psr,comm` every 5 s over the same pids: the core each thread last ran on |
| `run4/ps-<order>.txt` | `ps -eo pid,ppid,args` at the start of that call: which pid serves which port |
| `run4/rocm-smi-<order>.txt` | the host's `rocm-smi --showuse --showpower --showclocks` every 5 s during that call; device-wide |
| `run4/card-empty-check.txt`, `run4/card-empty` | the host's `rocm-smi` read before 0c's relaunch, and the file that released it |
| `run4/router-<arm>-n<N>.log` | `docker logs` of the router that served that level |
| `run4/arm-progress` | `ARMDONE` and the stop-rule check per level, then `BLOCKB_DONE` |
| `run4/host.txt` | date, CPU model, `nproc`, kernel, amdgpu version |
| `run4/rocm-smi-end.txt`, `run4/docker-images.txt` | VRAM and KFD processes after both engines stopped; the images and digests on the host |
| `run4/vllm-version.txt`, `run4/hf-download*.txt` | `vllm --version` and the weights' snapshot path |
| `run4/droplet-up.txt`, `run4/harvest.txt` | the host clock when the work directory was made, and at the harvest |
| `run4/stage`, `run4/*.pid` | the scripts' handshakes: the last stage marker and the engines' PIDs |

## What differs from the runsheet's commands

- **The pair was launched twice before block 0.** The first launch logged 67.64
  against 69.62 GiB (`-cold` files). §2's row sends that miss to a relaunch,
  and the pair was relaunched without changing a flag. The first launch's
  `results/check` was moved to `results/check-cold` before the relaunch, so
  that `s1.sh`'s wait for the JSON could not be satisfied by the old file.
- **Between the first launch and the second, the card was confirmed empty from
  the host after "No KFD PIDs" alone, while 126 GB were still in use.** The
  second launch's two pools match run 3's to the token, so the release
  finished before vLLM measured memory. Before 0c the wait also required used
  VRAM under 1 GB (`card-empty-check.txt`, 0.30 GB).
- The commands were sent from a laptop over SSH into the `tmux` session, not
  typed: window 0 the container, window 1 `hostrec.sh`, window 2 `blockb.sh`.
- The host's `rocm` container, Jupyter Lab on port 8888, which the image starts
  at boot, was stopped before anything else ran, as in run 1.
