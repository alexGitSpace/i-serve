# On the host, while block A's first levels run: the record that places this droplet.
R=/workspace/run3
until grep -q SOLO_UP $R/stage 2>/dev/null; do sleep 2; done
sleep 15
{ date -u; lscpu | grep 'Model name'; nproc; uname -r; cat /sys/module/amdgpu/version
  for i in 1 2 3; do echo "--- sample $i $(date -u +%T)"
    rocm-smi --showclocks --showpower --showperflevel --showuse --showtemp; sleep 10; done
} > $R/host-under-load.txt 2>&1
