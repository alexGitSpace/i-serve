# §3: block A against the pair, half the concurrency to each engine, at once.
set -u
cd /workspace/i-serve
R=/workspace/run3
FIT=${1:?FIT}
python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
  --port 8000 --startup-log $R/engine-8000.log \
  --reference-pool 507050 --out $R/results/pair-8000 > $R/pair-8000-harness.txt 2>&1 &
python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
  --port 8001 --startup-log $R/engine-8001.log \
  --reference-pool 507050 --out $R/results/pair-8001 > $R/pair-8001-harness.txt 2>&1 &
wait
echo S3DONE > $R/stage
