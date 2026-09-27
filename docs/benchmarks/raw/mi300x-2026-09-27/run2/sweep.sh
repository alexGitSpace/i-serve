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
