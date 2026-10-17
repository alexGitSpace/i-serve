# Sourced by s1.sh, s2.sh, s3.sh and pin.sh, in the container: the launch and its
# gates, block 0's paired call, and the CPU record that goes with it.
R=/workspace/run4
cd /workspace/i-serve
# /reset_prefix_cache is attached only under it (runsheet §0); every engine
# started below inherits it from this shell.
export VLLM_SERVER_DEV_MODE=1

launch() {   # launch <port> <log>: one engine at 0.45, back once /health answers
  vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port $1 --dtype auto \
    --gpu-memory-utilization 0.45 --max-model-len 9000 \
    --max-num-batched-tokens 2048 > $R/$2 2>&1 &
  echo $! > $R/e$1.pid
  until curl -sf 127.0.0.1:$1/health >/dev/null; do
    kill -0 $(cat $R/e$1.pid) 2>/dev/null || { echo "E$1_DIED" > $R/stage; exit 1; }
    sleep 2
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

together() {   # together <tag> <log 8000> <log 8001>: fleet-bill-half on both at once
  ps -eo pid,ppid,args > $R/ps-$1.txt
  # top's -p takes at most 20 pids; one engine is an APIServer and its EngineCore.
  pids=$( { tree $(cat $R/e8000.pid); tree $(cat $R/e8001.pid); } | head -20 | paste -sd, -)
  # %CPU per thread over each 5 s interval. ps's pcpu is a lifetime average and
  # cannot show a core pinned for the 45 s of a level.
  top -b -H -d 5 -p $pids > $R/top-$1.txt &
  top_pid=$!
  # The core each thread last ran on; psr is instantaneous, pcpu is not read.
  ( while :; do date -u +%T; ps -L -o pid,tid,psr,comm -p $pids; sleep 5; done ) \
    > $R/psr-$1.txt &
  psr_pid=$!
  echo "0B_START $1" > $R/stage
  python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
    --port 8000 --reset-cache --startup-log $R/$2 --reference-pool 506525 \
    --out $R/results/$1-8000 > $R/$1-8000-harness.txt 2>&1 &
  h0=$!
  python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
    --port 8001 --reset-cache --startup-log $R/$3 --reference-pool 506525 \
    --out $R/results/$1-8001 > $R/$1-8001-harness.txt 2>&1 &
  h1=$!
  wait $h0 $h1
  kill $top_pid $psr_pid
  echo "0B_DONE $1" > $R/stage
}
