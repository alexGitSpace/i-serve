import glob, json, os
R = "/workspace/run3/results"
print(f"{'arm':<12}{'N':>4}{'h':>7}{'h8000':>7}{'h8001':>7}{'→8000':>7}{'→8001':>7}{'pol':>6}{'TTFTp50':>9}{'TTFTp99':>9}{'TPOTp50':>9}{'TPOTp99':>9}{'medITL':>8}{'tok/s':>8}  invalid/warn")
for arm in ("prefix", "round_robin"):
    for n in ("031", "063", "127", "255", "511"):
        f = f"{R}/{arm}-n{n}/router-n{n}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        g = lambda k: d.get(k, float("nan"))
        pol = g(f"harness_policy_{arm}")
        print(f"{arm:<12}{n:>4}{g('harness_measured_hit_rate'):>7.3f}{g('harness_hit_rate_127.0.0.1:8000'):>7.3f}"
              f"{g('harness_hit_rate_127.0.0.1:8001'):>7.3f}{g('harness_upstream_http://127.0.0.1:8000'):>7.0f}"
              f"{g('harness_upstream_http://127.0.0.1:8001'):>7.0f}{pol:>6.0f}{g('median_ttft_ms'):>9.1f}{g('p99_ttft_ms'):>9.1f}"
              f"{g('median_tpot_ms'):>9.2f}{g('p99_tpot_ms'):>9.2f}{g('median_itl_ms'):>8.2f}{g('output_throughput'):>8.1f}  "
              + "; ".join(x[:60] for x in d.get("harness_invalid", []) + d.get("harness_warnings", []) if "closed loop" not in x))
