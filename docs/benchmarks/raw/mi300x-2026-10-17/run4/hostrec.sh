# On the host, started before s2.sh and left running: the droplet record once,
# then rocm-smi every 5 s while each 0b call runs. Device-wide figures only --
# rocm-smi cannot say which engine had the card. Run as: bash hostrec.sh
R=/workspace/run4
{ date -u; lscpu | grep 'Model name'; nproc; uname -r; cat /sys/module/amdgpu/version
} > $R/host.txt 2>&1
while :; do
  until grep -q '^0B_START' $R/stage 2>/dev/null; do sleep 1; done
  tag=$(cut -d' ' -f2 $R/stage)
  while grep -q '^0B_START' $R/stage; do
    { echo "--- $(date -u +%T)"; rocm-smi --showuse --showpower --showclocks; } \
      >> $R/rocm-smi-$tag.txt 2>&1
    sleep 5
  done
done
