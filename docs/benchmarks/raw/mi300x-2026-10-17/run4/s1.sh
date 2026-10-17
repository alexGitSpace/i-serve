# §1–§2, in the container: version, weights, the pair serially, its gates, and
# the droplet check against 8001. Run as: bash s1.sh
set -u
. /workspace/i-serve/run4/lib.sh
mkdir -p $R/results
# Block 0's record needs these in the image; found missing here, not at 0b.
for t in top pgrep taskset paste; do
  command -v $t >/dev/null || { echo "TOOL_MISSING $t" > $R/stage; exit 1; }
done
vllm --version > $R/vllm-version.txt 2>&1
hf download Qwen/Qwen3-8B > $R/hf-download.txt 2>&1

# Serially: the second engine only once the first answers /health (run 3 §2).
launch 8000 engine-8000.log
launch 8001 engine-8001.log
gate 8000 engine-8000.log
gate 8001 engine-8001.log
checkpoint engine-8000.log engine-8001.log
echo PAIR_UP > $R/stage

# The droplet check: fleet-bill's c = 1 level against 8001, stopped once that
# level's JSON exists. Its seeds are not fleet-bill-half's (bench/scenarios/fleet.py).
python3 bench/harness.py --scenario fleet-bill --accelerator mi300x-run2 \
  --port 8001 --startup-log $R/engine-8001.log --reference-pool 506525 \
  --out $R/results/check > $R/check-harness.txt 2>&1 &
check=$!
until [ -f $R/results/check/fleet-c001-h00.json ]; do
  kill -0 $check 2>/dev/null || { echo CHECK_DIED > $R/stage; exit 1; }
  sleep 2
done
kill $check
python3 -c "import json; d = json.load(open('$R/results/check/fleet-c001-h00.json')); \
print(f\"droplet check: median ITL {d['median_itl_ms']:.2f} ms -- 5.04 run 3, 5.22 run 2, 6.61 run 1\")" \
  | tee $R/droplet-check.txt
echo S1DONE > $R/stage
