/* calc.js -- bench/roofline.py and predictions.what_if_point(), ported.
 *
 * The one second implementation allowed, because selftest.js holds it to the
 * rows of site/data/golden.json. Keep the ORDER OF OPERATIONS identical to
 * roofline.py: they agree to 1e-9, and a floor() of 23.999999999999996 is a seat
 * in one language only. Every constant comes in through `data` but the physical
 * ones typed at the top, which that parity holds equal -- the model priced and
 * the one the interference fit may be lent to included. window.Roofline in a
 * browser, module.exports under node: the page and the CI check are one file.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) { module.exports = factory(); }
  else { root.Roofline = factory(); }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const FLOPS_PER_MAC = 2;

  // Python's round(): half to even. round(2.5) is 2 there and 3 in JS.
  function pyRound(x) {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  const finite = (v) => (v === null || v === undefined || !isFinite(v)) ? null : v;

  // --- specs -----------------------------------------------------------------
  function weightsBytes(m) { return m.params_total * m.weight_dtype_bytes; }
  function kvBytesPerToken(m) {
    return 2 * m.num_kv_heads * m.head_dim * m.kv_dtype_bytes * m.num_layers;
  }
  function withKvDtype(model, bytes) {
    if (bytes === model.kv_dtype_bytes) return model;
    return Object.assign({}, model, { name: model.name + " FP8 KV", kv_dtype_bytes: bytes });
  }

  // --- traffic ---------------------------------------------------------------
  function decodeStepBytes(m, batch, ctx) {
    return weightsBytes(m) + kvBytesPerToken(m) * batch * ctx;
  }
  function prefillBytes(m, promptTokens) {
    return weightsBytes(m) + kvBytesPerToken(m) * promptTokens;
  }
  function kvCacheTokens(m, a, gmu) {
    const kvSpace = a.memory_bytes * gmu - weightsBytes(m);
    if (kvSpace < 0) {
      throw new Error(
        m.name + " weights (" + (weightsBytes(m) / 1e9).toFixed(1) + " GB) exceed the " +
        Math.round(gmu * 100) + "% share of " + a.name + " (" +
        (a.memory_bytes * gmu / 1e9).toFixed(1) + " GB): no KV cache remains");
    }
    return kvSpace / kvBytesPerToken(m);
  }

  // --- comparison ------------------------------------------------------------
  function roofline(memory, compute) {
    let seconds, boundBy, other;
    if (memory > compute) { seconds = memory; boundBy = "memory"; other = compute; }
    else { seconds = compute; boundBy = "compute"; other = memory; }
    return { seconds, bound_by: boundBy, ratio: other > 0 ? seconds / other : Infinity };
  }

  // --- floors ----------------------------------------------------------------
  function tpotFloor(m, a, batch, ctx) {
    const memory = decodeStepBytes(m, batch, ctx) / (a.peak_bandwidth * a.achieved_bandwidth);
    const compute = FLOPS_PER_MAC * m.params_non_embedding * batch / (a.peak_flops * a.mfu);
    return roofline(memory, compute);
  }
  function ttftFloor(m, a, promptTokens) {
    const memory = prefillBytes(m, promptTokens) / (a.peak_bandwidth * a.achieved_bandwidth);
    const compute = FLOPS_PER_MAC * m.params_non_embedding * promptTokens / (a.peak_flops * a.mfu);
    return roofline(memory, compute);
  }

  // --- inversions ------------------------------------------------------------
  function maxNumSeqsFromSlo(m, a, ctx, tpotTarget) {
    if (ctx <= 0) throw new Error("context_len must be positive");
    const byteBudget = tpotTarget * a.peak_bandwidth * a.achieved_bandwidth;
    const kvPerSeq = kvBytesPerToken(m) * ctx;
    const byBandwidth = (byteBudget - weightsBytes(m)) / kvPerSeq;
    const byCompute = (tpotTarget * a.peak_flops * a.mfu) / (FLOPS_PER_MAC * m.params_non_embedding);
    return Math.max(Math.floor(Math.min(byBandwidth, byCompute)), 0);
  }
  function seatsUnderPrefillInterference(m, a, ctx, tpotTarget, slope, intercept, hitRate) {
    const stepPerSeat = (kvBytesPerToken(m) * ctx) / (a.peak_bandwidth * a.achieved_bandwidth);
    const stepAtZero = weightsBytes(m) / (a.peak_bandwidth * a.achieved_bandwidth);
    const uncached = 1.0 - hitRate;
    const denominator = stepPerSeat + uncached * slope;
    if (denominator <= 0) throw new Error("non-positive interference slope");
    return (tpotTarget - stepAtZero - uncached * intercept) / denominator;
  }
  function concurrencyCeiling(m, a, ctx, gmu) {
    if (ctx <= 0) throw new Error("context_len must be positive");
    return Math.floor(kvCacheTokens(m, a, gmu) / ctx);
  }
  function maxNumSeqs(m, a, ctx, tpotTarget, gmu) {
    const byLatency = maxNumSeqsFromSlo(m, a, ctx, tpotTarget);
    const byCapacity = concurrencyCeiling(m, a, ctx, gmu);
    const boundBy = byCapacity < byLatency ? "capacity" : "latency";
    const sequences = Math.min(byCapacity, byLatency);
    const ratio = sequences > 0 ? Math.max(byCapacity, byLatency) / sequences : Infinity;
    return { sequences, bound_by: boundBy, by_capacity: byCapacity, by_latency: byLatency, ratio };
  }

  // --- money -----------------------------------------------------------------
  function aggregateTokensPerSec(m, a, batch, ctx) {
    return batch / tpotFloor(m, a, batch, ctx).seconds;
  }
  function costPer1mTokens(aggregate, hourly) {
    return hourly / (aggregate * 3600) * 1e6;
  }

  // --- the operating point, as predictions.what_if_point() builds it ---------
  // inputs: {accelerator, prompt_tokens, output_tokens, hit_rate, tpot_target,
  //          ttft_target, kv_dtype_bytes, gpu_memory_utilization, hourly_rate|null}
  function whatIfPoint(data, inputs) {
    const key = inputs.accelerator;
    const a = data.accelerators[key];
    if (!a) throw new Error("unknown accelerator " + key);
    const modelKey = inputs.model || data.defaults.model;
    const base = data.models[modelKey];
    if (!base) throw new Error("unknown model " + modelKey);
    const m = withKvDtype(base, inputs.kv_dtype_bytes);
    const ctx = inputs.prompt_tokens + inputs.output_tokens;
    const prompt = inputs.prompt_tokens;
    const tpot = inputs.tpot_target, ttft = inputs.ttft_target;
    const gmu = inputs.gpu_memory_utilization;
    const h = inputs.hit_rate;
    if (!(h >= 0 && h <= 1)) throw new Error("hit_rate is a share of prompt tokens, so 0 <= h <= 1");
    const hourly = (inputs.hourly_rate === null || inputs.hourly_rate === undefined)
      ? a.hourly_rate : inputs.hourly_rate;

    const point = {
      inputs: {
        accelerator: key, model: modelKey, context_len: ctx, prompt_tokens: prompt,
        tpot_target_s: tpot, ttft_target_s: ttft, gpu_memory_utilization: gmu,
        kv_dtype_bytes: inputs.kv_dtype_bytes, hourly_rate: hourly, hit_rate: h,
      },
      model_name: m.name,
      accelerator_name: a.name,
      coefficients: { eff_mem: a.achieved_bandwidth, mfu: a.mfu, provenance: a.provenance },
      kv_pool_tokens: null, seats: null, tpot_floor_at_max_num_seqs: null,
      ttft_floor: null, ttft_floor_uncached: null, aggregate_tokens_per_sec: null,
      cost_per_1m_output_tokens: null, service: null, error: null,
    };

    const prefill = ttftFloor(m, a, prompt);
    const uncached = Math.max(1, pyRound(prompt * (1.0 - h)));
    const prefillUncached = ttftFloor(m, a, uncached);
    point.ttft_floor = { seconds: prefill.seconds, bound_by: prefill.bound_by,
                         ratio: finite(prefill.ratio), share_of_target: prefill.seconds / ttft };
    point.ttft_floor_uncached = { seconds: prefillUncached.seconds, uncached_tokens: uncached,
                                  share_of_target: prefillUncached.seconds / ttft };

    let pool;
    try { pool = kvCacheTokens(m, a, gmu); }
    catch (e) { point.error = e.message; return point; }
    const seats = maxNumSeqs(m, a, ctx, tpot, gmu);
    point.kv_pool_tokens = pool;
    point.seats = { by_capacity: seats.by_capacity, by_latency: seats.by_latency,
                    max_num_seqs: seats.sequences, bound_by: seats.bound_by,
                    gap: seats.sequences ? finite(seats.ratio) : null };
    if (seats.sequences === 0) {
      point.error = "Not one sequence fits: " + ctx.toLocaleString("en-US") +
        " tokens of reserved context against a " + (pool / 1e6).toFixed(2) +
        " M token pool, or a TPOT target below the batch-1 decode step. Lower the " +
        "context length, loosen the target, raise gpu_memory_utilization, or take a " +
        "bigger card -- there is no operating point here to price.";
      return point;
    }

    const decode = tpotFloor(m, a, seats.sequences, ctx);
    const total = aggregateTokensPerSec(m, a, seats.sequences, ctx);
    point.tpot_floor_at_max_num_seqs = { seconds: decode.seconds, bound_by: decode.bound_by,
                                         ratio: finite(decode.ratio), share_of_target: decode.seconds / tpot };
    point.aggregate_tokens_per_sec = total;
    point.cost_per_1m_output_tokens = costPer1mTokens(total, hourly);

    const fit = modelKey === data.measured.model ? data.interference[key] : null;
    if (fit) {
      const byInterference = seatsUnderPrefillInterference(
        m, a, ctx, tpot, fit.slope_s_per_seat, fit.intercept_s, h);
      let shipped = Math.max(Math.floor(byInterference), 0);
      let boundBy = "interference";
      if (seats.by_capacity < shipped) { shipped = seats.by_capacity; boundBy = "capacity"; }
      const service = {
        seats_at_hit_rate: byInterference, seats_shipped: shipped, bound_by: boundBy,
        tpot_floor_at_seats_s: null, aggregate_tokens_per_sec: null, cost_per_1m_output_tokens: null,
        fit: { slope_s_per_seat: fit.slope_s_per_seat, intercept_s: fit.intercept_s,
               provenance: fit.provenance },
        note: h > 0.5
          ? "a floor, not an estimate: at h = 0.8 run 3 measured 37.8 seats against 24.3 predicted (docs/SLO.md section 6)"
          : null,
      };
      if (shipped > 0) {
        service.tpot_floor_at_seats_s = tpotFloor(m, a, shipped, ctx).seconds;
        const rate = aggregateTokensPerSec(m, a, shipped, ctx);
        service.aggregate_tokens_per_sec = rate;
        service.cost_per_1m_output_tokens = costPer1mTokens(rate, hourly);
      }
      point.service = service;
    }
    return point;
  }

  return {
    FLOPS_PER_MAC, pyRound, withKvDtype, weightsBytes, kvBytesPerToken,
    decodeStepBytes, prefillBytes, kvCacheTokens, roofline, tpotFloor, ttftFloor,
    maxNumSeqsFromSlo, seatsUnderPrefillInterference, concurrencyCeiling, maxNumSeqs,
    aggregateTokensPerSec, costPer1mTokens, whatIfPoint,
  };
});
