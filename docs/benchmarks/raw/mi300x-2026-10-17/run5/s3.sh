# §3 step B1, in the container: the pair relaunched 8000 first with disjoint CU
# masks, the mask's check against s1's baseline, then the paired call.
# Run as: bash s3.sh <FIT>
set -u
. /workspace/i-serve/run5/lib.sh
FIT=${1:?FIT}

stop 8000 8001
# Only the host's rocm-smi can say the card is empty (run 4 runsheet §3, 0c).
echo B1_WAIT_CARD > $R/stage
until [ -f $R/card-empty ]; do sleep 2; done
rm $R/card-empty

launch 8000 engine-8000-masked.log HSA_CU_MASK=$MASK_8000
launch 8001 engine-8001-masked.log HSA_CU_MASK=$MASK_8001
gate 8000 engine-8000-masked.log
gate 8001 engine-8001-masked.log
checkpoint engine-8000-masked.log engine-8001-masked.log
envrec 8000 masked
envrec 8001 masked

dropcheck 8001 engine-8001-masked.log droplet-check-masked
# A 4 000-token prefill is compute-bound: half the CUs should near double it.
python3 - $R <<'EOF' | tee $R/mask-verdict.txt
import json, sys
ttft = [json.load(open(f"{sys.argv[1]}/results/{t}/fleet-c001-h00.json"))["median_ttft_ms"]
        for t in ("droplet-check", "droplet-check-masked")]
r = ttft[1] / ttft[0]
verdict = "TAKEN" if r >= 1.5 else "NOT_TAKEN" if r <= 1.15 else "UNCLEAR"
print(f"TTFT {ttft[0]:.1f} -> {ttft[1]:.1f} ms, ratio {r:.2f}: MASK_{verdict}")
EOF
grep -q MASK_NOT_TAKEN $R/mask-verdict.txt && { echo MASK_NOT_TAKEN > $R/stage; exit 1; }
# Only a verdict lets B1 run: a reader's traceback is not one.
grep -qE 'MASK_(TAKEN|UNCLEAR)' $R/mask-verdict.txt || { echo MASK_GATE_FAILED > $R/stage; exit 1; }

together masked-8000first engine-8000-masked.log engine-8001-masked.log $FIT
echo S3DONE > $R/stage
