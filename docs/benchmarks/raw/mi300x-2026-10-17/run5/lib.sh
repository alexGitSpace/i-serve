# Sourced by s1.sh … s4.sh, in the container: run 4's launch, gates and paired call,
# with an environment per engine, and the record that says the environment arrived.
R=/workspace/run5
cd /workspace/i-serve
# /reset_prefix_cache is attached only under it (run 4 runsheet §0).
export VLLM_SERVER_DEV_MODE=1
# Block B's halves of the card: ROCr deals CU bits round-robin over the XCDs, so
# each range is 19 CUs on every one of the eight (runsheet §0).
MASK_8000=0:0-151
MASK_8001=0:152-303

launch() {   # launch <port> <log> [VAR=value ...]: one engine at 0.45, back once /health answers
  env "${@:3}" vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port $1 --dtype auto \
    --gpu-memory-utilization 0.45 --max-model-len 9000 \
    --max-num-batched-tokens 2048 > $R/$2 2>&1 &
  echo $! > $R/e$1.pid
  until curl -sf 127.0.0.1:$1/health >/dev/null; do
    kill -0 $(cat $R/e$1.pid) 2>/dev/null || { echo "E$1_DIED" > $R/stage; exit 1; }
    sleep 2
  done
}

stop() {   # stop <port>...: kill each engine and wait until every process of its tree is gone
  # The EngineCore holds the card, and it can outlive the APIServer that was killed.
  local pids=$(for p in "$@"; do tree $(cat $R/e$p.pid); done)
  for p in "$@"; do kill $(cat $R/e$p.pid); done
  for pid in $pids; do
    while kill -0 $pid 2>/dev/null; do sleep 2; done
  done
}

gate() {   # gate <port> <log>: dev mode reached this engine, and the reset answers
  grep -q 'Development endpoints are enabled' $R/$2 \
    || { echo "GATE_DEVMODE_$1" > $R/stage; exit 1; }
  grep -q 'Route: /reset_prefix_cache' $R/$2 \
    || { echo "GATE_ROUTE_$1" > $R/stage; exit 1; }
  echo "$1 $2 $(curl -s -X POST 127.0.0.1:$1/reset_prefix_cache)" >> $R/reset-gate.txt
  tail -1 $R/reset-gate.txt | tr -d ' ' | grep -q '"success":true' \
    || { echo "GATE_RESET_$1" > $R/stage; exit 1; }
}

checkpoint() {   # checkpoint <log>...: checkpoint A's lines, appended to one file
  grep -E 'GPU KV cache size|Available KV cache memory|prefix_caching|max_num_batched_tokens|backend|Graph capturing finished' \
    "${@/#/$R/}" >> $R/checkpoint-a.txt
}

tree() {   # tree <pid>: the pid and every descendant, one per line
  echo $1
  for c in $(pgrep -P $1); do tree $c; done
}

envrec() {   # envrec <port> <tag>: the HSA_* and GPU_* variables each process of the engine holds
  # Only those names: the whole environ would carry the HF token (adding-a-run.md §2).
  for pid in $(tree $(cat $R/e$1.pid)); do
    echo "$1 $pid $(tr '\0' '\n' < /proc/$pid/environ | grep -E '^(HSA_CU_MASK|GPU_MAX_HW_QUEUES)=' | paste -sd' ' -)"
  done >> $R/env-$2.txt
}

dropcheck() {   # dropcheck <port> <log> <tag>: fleet-bill's c = 1 level, stopped once its JSON exists
  python3 bench/harness.py --scenario fleet-bill --accelerator mi300x-run2 \
    --port $1 --startup-log $R/$2 --reference-pool 506525 \
    --out $R/results/$3 > $R/$3-harness.txt 2>&1 &
  local check=$!
  # The per-request file is the level's last write (bench/harness.py).
  until [ -f $R/results/$3/fleet-c001-h00-requests.jsonl ]; do
    kill -0 $check 2>/dev/null || { echo "CHECK_DIED $3" > $R/stage; exit 1; }
    sleep 2
  done
  sleep 1
  kill $check
  python3 -c "import json; d = json.load(open('$R/results/$3/fleet-c001-h00.json')); \
print(f\"$3: median ITL {d['median_itl_ms']:.2f} ms, median TTFT {d['median_ttft_ms']:.1f} ms\")" \
    | tee $R/$3.txt
}

together() {   # together <tag> <log 8000> <log 8001> <FIT>: fleet-bill-half on both at once
  ps -eo pid,ppid,args > $R/ps-$1.txt
  pids=$( { tree $(cat $R/e8000.pid); tree $(cat $R/e8001.pid); } | head -20 | paste -sd, -)
  top -b -H -d 5 -p $pids > $R/top-$1.txt &
  local top_pid=$!
  echo "0B_START $1" > $R/stage
  python3 bench/harness.py --scenario fleet-bill-half --accelerator $4 \
    --port 8000 --reset-cache --startup-log $R/$2 --reference-pool 506525 \
    --out $R/results/$1-8000 > $R/$1-8000-harness.txt 2>&1 &
  local h0=$!
  python3 bench/harness.py --scenario fleet-bill-half --accelerator $4 \
    --port 8001 --reset-cache --startup-log $R/$3 --reference-pool 506525 \
    --out $R/results/$1-8001 > $R/$1-8001-harness.txt 2>&1 &
  local h1=$!
  wait $h0; local r0=$?
  wait $h1; local r1=$?
  kill $top_pid
  echo "0B_DONE $1" > $R/stage
  [ $r0 -eq 0 ] && [ $r1 -eq 0 ] || { echo "PAIR_FAILED $1" > $R/stage; exit 1; }
  python3 - $R/results/$1 <<'EOF' | tee -a $R/pairs.txt
import json, sys
itl = {p: json.load(open(f"{sys.argv[1]}-{p}/fleet-c032-h00.json"))["median_itl_ms"]
       for p in (8000, 8001)}
slow = max(itl, key=itl.get)
print(f"{sys.argv[1].rsplit('/', 1)[1]}: 8000 {itl[8000]:.2f} ms, 8001 {itl[8001]:.2f} ms, "
      f"slow {slow}, ratio {max(itl.values()) / min(itl.values()):.2f}")
EOF
}
