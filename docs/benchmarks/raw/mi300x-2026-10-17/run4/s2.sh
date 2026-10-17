# §3 steps 0a and 0b, in the container: each engine alone, then both together.
# Run as: bash s2.sh <FIT>
set -u
. /workspace/i-serve/run4/lib.sh
FIT=${1:?FIT}

# 0a. --reset-cache on every call: fleet-bill-half's seed is fixed (runsheet §3).
for p in 8000 8001; do
  echo "0A $p" > $R/stage
  python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
    --port $p --reset-cache --startup-log $R/engine-$p.log --reference-pool 506525 \
    --out $R/results/alone-$p > $R/alone-$p-harness.txt 2>&1 \
    || { echo "0A_FAILED $p" > $R/stage; exit 1; }
done

# 0b. Run 3's two concurrent calls, 8000 launched first.
together together-8000first engine-8000.log engine-8001.log
echo S2DONE > $R/stage
