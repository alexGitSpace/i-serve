# §4, on the host: one level per call, on a freshly launched router, each level
# checked against the stop rule as soon as its JSON lands. Sourced by blockb.sh;
# FIT and LOG come from the host's shell, which s1.sh's container does not share.
R=/workspace/run4
FIT=${FIT:?FIT}
# The live engine-8000 startup log: engine-8000.log, or engine-8000-r2.log after 0c.
LOG=${LOG:?LOG}

arm() {   # arm <prefix|round_robin> <031|063|127|255|511> [bounded-load]
  local tag=$1-n$2${3:+-bl$3}
  docker rm -f router >/dev/null 2>&1
  docker run -d --name router --network host prefix-router:dev \
    -listen :8080 -upstreams http://127.0.0.1:8000,http://127.0.0.1:8001 \
    -policy $1 -dial-timeout 250ms -bounded-load ${3:-1.25} >/dev/null
  until curl -sf localhost:8080/v1/models >/dev/null; do sleep 1; done
  docker exec -w /workspace/i-serve run4 python3 bench/harness.py \
    --scenario router-arm-n$2 --accelerator $FIT --port 8080 --expect-policy $1 \
    --metrics-endpoint 127.0.0.1:8000 --metrics-endpoint 127.0.0.1:8001 \
    --reset-cache --warmup-passes 2 \
    --startup-log $R/$LOG --reference-pool 506525 \
    --out $R/results/$tag > $R/arm-$tag.txt 2>&1
  rc=$?; docker logs router > $R/router-$tag.log 2>&1
  echo "ARMDONE $tag $rc" >> $R/arm-progress
  [ $rc -eq 0 ] && check $R/results/$tag/router-n$2.json $1 $2
}

check() {   # check <json> <arm> <N>: the stop rule of runsheet §4
  python3 - "$@" <<'EOF' | tee -a $R/arm-progress
import json, sys
path, arm, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(path))
h = d["harness_measured_hit_rate"]
reset, passes = d["harness_cache_reset"], d["harness_warmup_passes"]
print(f"  {arm} N={n}: h {h:.3f}, cache_reset {reset:.0f}, warmup_passes {passes:.0f}")
stop = []
if reset != 1:
    stop.append("no cache reset")
if passes != 2:
    stop.append("not two warmup passes")
if h > 0.81:
    stop.append("h above 0.8: the reset did not take")
# Round robin carries no bound, so inside its room nothing explains a shortfall.
# Affinity below 0.8 is read against the split, never stopped on.
if arm == "round_robin" and n <= 63 and h < 0.78:
    stop.append("round robin below 0.8 with every prefix seeded on both engines")
if stop:
    print("  STOP: " + "; ".join(stop))
    sys.exit(1)
EOF
  return ${PIPESTATUS[0]}
}
