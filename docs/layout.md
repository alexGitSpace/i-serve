# Layout

The map of the tree, not a route: one row per component, what it is and how
far it got. Directories map one-to-one onto the components the platform is
built from. The routes are in [../README.md](../README.md).

| Path | What | State |
|---|---|---|
| `up.sh`, `down.sh` | The eleven commands of [running-on-kind.md](running-on-kind.md) as one entry point | up and down |
| `docs/SLO.md` | Targets, floors, concurrency ceiling, each derived | ✅ |
| `docs/GLOSSARY.md` | Vocabulary and notation | ✅ |
| `docs/model-anatomy.md` | The served model at five zoom levels | ✅ |
| `docs/accelerator-landscape.md` | Which decode-equation term each vendor attacks | ✅ snapshot, 2026-08-23 |
| `docs/architecture.md` | Request path, ingress to GPU | ✅ drawn from the running objects |
| `docs/runbook.md` | Canary rollout, SLO-breach tree, morning triage, two engines on one card, OOM | canary run once on `kind`; two engines on one card measured once on a droplet; OOM procedure not written |
| `docs/symptom-map.md`, `.json` | The decision tree, and the checked subset the site evaluates | 75 nodes; the JSON's 38 nodes held equal by a test |
| `docs/running-on-kind.md` | The eleven commands, the two waits, the stub contract | ✅ |
| `docs/adding-a-run.md` | Why runs accumulate here, and the checklist | first user: MI300X run 1 |
| `docs/benchmarks/` | One report per run, registered by its front matter; the run journal (`README.md`, generated); the ledger (`ledger.jsonl`, generated); the chart above | every run written up, raw evidence committed; the count and what each found are in the journal |
| `docs/benchmarks/runsheets/` | The sheet written **before** each run | one or two per run, written before its card was rented |
| `docs/instrument-vllm-bench-sweep.md` | What `vllm bench sweep` does at `v0.27.1` | verified off-card against that tag |
| `bench/` | Load harness, `roofline.py`, chart generator, site export, the run journal (`runs.py`), the ledger (`ledger.py`), checkpoint snapshots of every priced model (`models/`); tests in `bench/tests/` | model calibrated on Qwen3-8B; two more architectures priced, never served; the harness has run on both cards |
| `site/` | Calculator and advisor as one static page for Pages | live; parity 438/438 rows |
| `.github/workflows/` | Tests on Python 3.10 and 3.14, the quick-start commands, the chart, the run journal, the parity check | installs nothing, which is the claim it tests |
| `.githooks/` | Pre-commit: regenerates the chart, `site/data/`, the run journal and the ledger; refuses a status glyph on the map, non-English content, a credential | opt-in: `git config core.hooksPath .githooks`; CI runs the three refusals on a branch no hook saw |
| `deploy/kind/` | Local CPU-only cluster | control plane + two workers |
| `deploy/manifests/` | vLLM as Deployment, Service, Ingress; a `kind` overlay swaps in the stub | base is the GPU artefact; a canary overlay splits `/v1` by weight |
| `deploy/ingress/` | ingress-nginx overlay, the edge before the Service | live on `kind`; edge timeout derived and measured |
| `deploy/helm/vllm/` | vLLM chart | empty |
| `deploy/keda/` | Queue-depth autoscaling on `vllm:num_requests_waiting` | watched 1→4→1 on `kind` |
| `deploy/observability/` | Prometheus rules, Grafana dashboards as code | queue alert watched firing; histogram panels dark until a card |
| `deploy/router/` | The router on `kind`: manifests, fleet-filling script, the flag's cost | watched over three replicas; staleness priced |
| `deploy/terraform/` | GPU node provisioning | one MI300X droplet; validated, never applied |
| `router/` | Go prefix-aware router: the replica that already holds the prompt | binary and tests; on `kind`, a second host before the stub Pods |
| `controllers/modelwarmup/` | Go operator: warm a model before it joins routing | architecture note only, arguing against the code |
