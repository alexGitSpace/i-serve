# §3 step 0c, in the container: both engines down, the card confirmed empty from
# the host, the pair relaunched 8001 first, and 0b again. Run as: bash s3.sh <FIT>
set -u
. /workspace/i-serve/run4/lib.sh
FIT=${1:?FIT}

for p in 8000 8001; do kill $(cat $R/e$p.pid); done
for p in 8000 8001; do
  while kill -0 $(cat $R/e$p.pid) 2>/dev/null; do sleep 2; done
done
# Only the host's rocm-smi can say the card is empty. The operator reads it
# there and touches the file (runsheet §3, 0c).
echo 0C_WAIT_CARD > $R/stage
until [ -f $R/card-empty ]; do sleep 2; done

# New log names: the first launch's logs are checkpoint A's evidence.
launch 8001 engine-8001-r2.log
launch 8000 engine-8000-r2.log
gate 8001 engine-8001-r2.log
gate 8000 engine-8000-r2.log
checkpoint engine-8001-r2.log engine-8000-r2.log
together together-8001first engine-8000-r2.log engine-8001-r2.log
echo S3DONE > $R/stage
