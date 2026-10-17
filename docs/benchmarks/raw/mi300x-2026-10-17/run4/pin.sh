# §3's CPU branch, in the container: pin each live engine's whole tree to its
# own ten cores, then 0b once more. Run as: bash pin.sh <FIT> <log 8000> <log 8001>
set -u
. /workspace/i-serve/run4/lib.sh
FIT=${1:?FIT}

# -a: every thread, not only the main one (runsheet §3).
for pid in $(tree $(cat $R/e8000.pid)); do taskset -a -cp 0-9 $pid; done > $R/taskset.txt
for pid in $(tree $(cat $R/e8001.pid)); do taskset -a -cp 10-19 $pid; done >> $R/taskset.txt
together together-pinned ${2:?log 8000} ${3:?log 8001}
echo PINDONE > $R/stage
