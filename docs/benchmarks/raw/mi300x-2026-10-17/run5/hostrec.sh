# On the host, started before s2.sh and left running: the droplet record once, then
# per paired call rocm-smi every 5 s, the queues' MQDs and the runlist once, and
# the kernel's lines logged during it. Run as: bash hostrec.sh, in its own window
R=/workspace/run5
{ date -u; lscpu | grep 'Model name'; nproc; uname -r; cat /sys/module/amdgpu/version
  cat /sys/module/amdgpu/parameters/sched_policy /sys/module/amdgpu/parameters/hws_max_conc_proc
} > $R/host.txt 2>&1
while :; do
  until grep -q '^0B_START' $R/stage 2>/dev/null; do sleep 1; done
  tag=$(cut -d' ' -f2 $R/stage)
  # Each queue's MQD, CU-mask registers included: raw evidence, read off-card (§3).
  cat /sys/kernel/debug/kfd/mqds > $R/mqds-$tag.txt 2>&1
  cat /sys/kernel/debug/kfd/rls > $R/rls-$tag.txt 2>&1
  # dmesg holds everything since boot: only what this call adds is its evidence.
  n=$(dmesg | wc -l)
  while grep -q '^0B_START' $R/stage; do
    { echo "--- $(date -u +%T)"; rocm-smi --showuse --showpower --showclocks; } \
      >> $R/rocm-smi-$tag.txt 2>&1
    sleep 5
  done
  # "Runlist is getting oversubscribed" is KFD's own word for time-slicing processes.
  dmesg | tail -n +$((n + 1)) | grep -iE 'oversubscri|kfd|amdgpu.*(hang|reset|timeout)' \
    > $R/dmesg-$tag.txt 2>&1
done
