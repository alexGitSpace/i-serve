# §3 step B2, in the container, only on §3's branch: the plain pair relaunched
# 8000 first, each process held to one hardware queue. Run as: bash s4.sh <FIT>
set -u
. /workspace/i-serve/run5/lib.sh
FIT=${1:?FIT}

stop 8000 8001
echo B2_WAIT_CARD > $R/stage
until [ -f $R/card-empty ]; do sleep 2; done
rm $R/card-empty

launch 8000 engine-8000-q1.log GPU_MAX_HW_QUEUES=1
launch 8001 engine-8001-q1.log GPU_MAX_HW_QUEUES=1
gate 8000 engine-8000-q1.log
gate 8001 engine-8001-q1.log
checkpoint engine-8000-q1.log engine-8001-q1.log
envrec 8000 q1
envrec 8001 q1
together q1-8000first engine-8000-q1.log engine-8001-q1.log $FIT
echo S4DONE > $R/stage
