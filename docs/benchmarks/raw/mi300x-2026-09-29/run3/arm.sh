# §4, on the host: one level per call, on a freshly launched router.
R=/workspace/run3
FIT=${FIT:?FIT}
arm() {   # arm <prefix|round_robin> <031|063|127|255|511>
  docker rm -f router >/dev/null 2>&1
  docker run -d --name router --network host prefix-router:dev \
    -listen :8080 -upstreams http://127.0.0.1:8000,http://127.0.0.1:8001 \
    -policy $1 -dial-timeout 250ms >/dev/null
  until curl -sf localhost:8080/v1/models >/dev/null; do sleep 1; done
  docker exec -w /workspace/i-serve run3 python3 bench/harness.py \
    --scenario router-arm-n$2 --accelerator $FIT --port 8080 --expect-policy $1 \
    --metrics-endpoint 127.0.0.1:8000 --metrics-endpoint 127.0.0.1:8001 \
    --startup-log /workspace/run3/engine-8000.log --reference-pool 507050 \
    --out /workspace/run3/results/$1-n$2 > $R/arm-$1-n$2.txt 2>&1
  rc=$?; docker logs router > $R/router-$1-n$2.log 2>&1
  echo "ARMDONE $1 $2 $rc" >> $R/arm-progress
}
