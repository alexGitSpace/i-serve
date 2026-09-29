# §3: the pair, serially -- the second engine only once the first answers /health.
set -u
cd /workspace/i-serve
R=/workspace/run3
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.45 --max-model-len 9000 \
  --max-num-batched-tokens 2048 > $R/engine-8000.log 2>&1 &
echo $! > $R/e8000.pid
until curl -sf 127.0.0.1:8000/health >/dev/null; do
  kill -0 $(cat $R/e8000.pid) 2>/dev/null || { echo E8000_DIED > $R/stage; exit 1; }; sleep 2; done
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8001 --dtype auto \
  --gpu-memory-utilization 0.45 --max-model-len 9000 \
  --max-num-batched-tokens 2048 > $R/engine-8001.log 2>&1 &
echo $! > $R/e8001.pid
until curl -sf 127.0.0.1:8001/health >/dev/null; do
  kill -0 $(cat $R/e8001.pid) 2>/dev/null || { echo E8001_DIED > $R/stage; exit 1; }; sleep 2; done
echo PAIR_UP > $R/stage
