# §3 step B0, in the container: run 4's 0b on this droplet, the plain pair,
# 8000 launched first. Run as: bash s2.sh <FIT>
set -u
. /workspace/i-serve/run5/lib.sh
FIT=${1:?FIT}

together plain-8000first engine-8000.log engine-8001.log $FIT
echo S2DONE > $R/stage
