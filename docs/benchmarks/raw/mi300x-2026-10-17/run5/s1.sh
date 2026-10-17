# §1–§2, in the container: both weights, the compile cache warmed, the plain pair
# serially, its gates, and the droplet check against 8001. Run as: bash s1.sh
set -u
. /workspace/i-serve/run5/lib.sh
mkdir -p $R/results
for t in top pgrep paste; do
  command -v $t >/dev/null || { echo "TOOL_MISSING $t" > $R/stage; exit 1; }
done
vllm --version > $R/vllm-version.txt 2>&1
# Block C's 72.7 GB fetches while block B's engines start; it must end before
# the droplet check, the first figure it could disturb. A failure is noted, not
# fatal: FIT and B1's baseline come from the check, and s5.sh fetches again.
hf download amd/Llama-3.3-70B-Instruct-FP8-KV > $R/hf-download-70b.txt 2>&1 &
dl=$!
hf download Qwen/Qwen3-8B > $R/hf-download.txt 2>&1

# Run 4's cold launch: the engine that compiled the graph logged a smaller pool.
# One engine compiles alone and is stopped, so both of the pair load the graph.
echo WARM > $R/stage
launch 8000 engine-warm.log
stop 8000

launch 8000 engine-8000.log
launch 8001 engine-8001.log
gate 8000 engine-8000.log
gate 8001 engine-8001.log
checkpoint engine-8000.log engine-8001.log
envrec 8000 plain
envrec 8001 plain
echo PAIR_UP > $R/stage

echo WAIT_DOWNLOAD > $R/stage
wait $dl || echo DOWNLOAD_FAILED >> $R/download-status.txt
# 8000 idle: the unmasked baseline the masked check is read against (§3).
dropcheck 8001 engine-8001.log droplet-check
echo S1DONE > $R/stage
