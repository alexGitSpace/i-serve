# §4, on the host: the eleven levels, stopping at the first that fails the rule.
# Run as: FIT=<fit> LOG=<engine-8000 log> bash blockb.sh
. /workspace/i-serve/run4/arm.sh
for level in "prefix 031" "round_robin 031" "round_robin 063" "prefix 063" \
             "prefix 127" "round_robin 127" "round_robin 255" "prefix 255" \
             "prefix 511" "round_robin 511" "prefix 127 1"; do
  arm $level || { echo "STOPPED at $level" >> $R/arm-progress; exit 1; }
done
echo BLOCKB_DONE >> $R/arm-progress
