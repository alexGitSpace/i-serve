# §5, in the container: the Qwen pair down, InferenceX's Llama-3.3-70B FP8 server
# line at TP = 1, and three concurrencies of its 8k/1k load. Run as: bash s5.sh [c ...]
# §8's relaunches: C_ENV="VLLM_ROCM_USE_AITER=1" or C_FLAGS="--kv-cache-dtype auto";
# argparse keeps the last value of a repeated flag.
set -u
. /workspace/i-serve/run5/lib.sh
M=amd/Llama-3.3-70B-Instruct-FP8-KV
LEVELS=${*:-4 16 64}
mkdir -p $R/results/claim

stop 8000 8001
echo C_WAIT_CARD > $R/stage
until [ -f $R/card-empty ]; do sleep 2; done
rm $R/card-empty
# Complete already, unless s1.sh noted a failure: then this is the fetch.
hf download $M >> $R/hf-download-70b.txt 2>&1 || { echo DOWNLOAD_FAILED > $R/stage; exit 1; }

# InferenceX's 70b_fp8_mi300x_docker.sh at c198b20e; its three changes are runsheet §4.
echo C_LAUNCH > $R/stage
env ${C_ENV:-} vllm serve $M --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --dtype float16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 \
  --max-model-len 9216 --max-num-seqs 64 --max-num-batched-tokens 131072 \
  --no-enable-prefix-caching --async-scheduling \
  --compilation-config '{"custom_ops":["-rms_norm","-quant_fp8","-silu_and_mul"]}' \
  ${C_FLAGS:-} > $R/engine-70b.log 2>&1 &
echo $! > $R/e8000.pid
until curl -sf 127.0.0.1:8000/health >/dev/null; do
  kill -0 $(cat $R/e8000.pid) 2>/dev/null || { echo E70B_DIED > $R/stage; exit 1; }
  sleep 5
done
grep -E 'GPU KV cache size|Available KV cache memory|Maximum concurrency|backend|kv_cache_dtype|Graph capturing finished' \
  $R/engine-70b.log > $R/checkpoint-c.txt

# InferenceX's [r L, L] in v0.27.1's [L(1 - r), L(1 + r)] sampling (runsheet §0).
for c in $LEVELS; do
  echo "C_LEVEL $c" > $R/stage
  curl -s 127.0.0.1:8000/metrics | grep -E '^vllm:num_preemptions' > $R/claim-c$c-metrics-before.txt
  vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $M \
    --dataset-name random --random-input-len 7373 --random-output-len 922 \
    --random-range-ratio 0.111 --ignore-eos --seed 0 \
    --max-concurrency $c --num-prompts $((5 * c)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir $R/results/claim --result-filename c$c.json \
    > $R/claim-c$c.txt 2>&1 || { echo "C_LEVEL_FAILED $c" > $R/stage; exit 1; }
  curl -s 127.0.0.1:8000/metrics | grep -E '^vllm:num_preemptions' > $R/claim-c$c-metrics-after.txt
  python3 - $R/results/claim/c$c.json <<'EOF' | tee -a $R/claim.txt
import json, sys
d = json.load(open(sys.argv[1]))
print(f"c = {d['max_concurrency']}: {d['completed']} done, {d['output_throughput']:.0f} tok/s per GPU, "
      f"{1000 / d['median_tpot_ms']:.1f} tok/s per user (median TPOT {d['median_tpot_ms']:.1f} ms), "
      f"median TTFT {d['median_ttft_ms']:.0f} ms")
EOF
done
echo S5DONE > $R/stage
