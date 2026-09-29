# §1: in the container. Version, weights, the solo engine, then block A's solo levels.
set -u
cd /workspace/i-serve
R=/workspace/run3
vllm --version > $R/vllm-version.txt 2>&1
hf download Qwen/Qwen3-8B > $R/hf-download.txt 2>&1
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.90 --max-model-len 9000 \
  --max-num-batched-tokens 2048 > $R/engine-solo.log 2>&1 &
echo $! > $R/solo.pid
until curl -sf 127.0.0.1:8000/health >/dev/null; do
  kill -0 $(cat $R/solo.pid) 2>/dev/null || { echo SOLO_DIED > $R/stage; exit 1; }
  sleep 2
done
echo SOLO_UP > $R/stage
python3 bench/harness.py --scenario fleet-bill --accelerator mi300x-run2 \
  --host 127.0.0.1 --port 8000 --startup-log $R/engine-solo.log \
  --reference-pool 1123065 --out $R/results/solo > $R/solo-harness.txt 2>&1
echo "S1DONE $?" > $R/stage
