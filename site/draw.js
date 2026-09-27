/* draw.js -- the two pictures, as hand-built SVG strings whose classes the
 * stylesheet colours, so they follow the reader's theme.
 *
 *   Draw.seats -- seats against ms: step(n) solid, the served step dashed where a
 *     card has an interference fit, the target and both limits as rules; measured
 *     points only inside a run's geometry (4 000 in, 200 out, BF16 KV, per card).
 *   Draw.cost  -- $ per 1M output tokens against the TPOT target, on a log axis.
 *
 * A legend entry is a name of two to four words; the formula goes to `tip`, and
 * what the picture turns out to show comes back as `note` under the chart.
 */
(function (root) {
  "use strict";
  const R = root.Roofline;
  const W = 640, H = 340, ML = 56, MR = 16, MT = 14, MB = 40;
  const PW = W - ML - MR, PH = H - MT - MB;

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  function el(tag, attrs, inner) {
    const a = Object.entries(attrs).map(([k, v]) => ` ${k}="${esc(v)}"`).join("");
    return inner === undefined ? `<${tag}${a}/>` : `<${tag}${a}>${inner}</${tag}>`;
  }
  const fmtMs = (x) => (Math.abs(x - Math.round(x)) < 1e-6 ? String(Math.round(x)) : x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
  const fmtUsd = (x) => "$" + (x >= 10 ? x.toFixed(1) : x.toFixed(2));

  function niceTicks(max, n) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => max / s <= n) || 10 * mag;
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(v);
    return out;
  }

  function frame(xTicks, yTicks, xLabel, yLabel, sx, sy, yFmt) {
    let g = "";
    for (const v of yTicks) {
      g += el("line", { class: "grid", x1: ML, x2: ML + PW, y1: sy(v), y2: sy(v) });
      g += el("text", { class: "axis", x: ML - 8, y: sy(v) + 4, "text-anchor": "end" }, yFmt(v));
    }
    for (const v of xTicks) {
      g += el("text", { class: "axis", x: sx(v), y: MT + PH + 18, "text-anchor": "middle" }, String(v));
    }
    g += el("line", { class: "axis-line", x1: ML, x2: ML + PW, y1: MT + PH, y2: MT + PH });
    g += el("text", { class: "axis label", x: ML + PW / 2, y: H - 6, "text-anchor": "middle" }, xLabel);
    g += el("text", { class: "axis label", x: 14, y: MT + PH / 2, "text-anchor": "middle",
                      transform: `rotate(-90 14 ${MT + PH / 2})` }, yLabel);
    return g;
  }

  function interferenceFit(data, state) {
    if (state.model && state.model !== data.measured.model) return null;
    return data.interference[state.accelerator] || null;
  }

  // Which card's runs may be drawn here: "l40s", "mi300x", "mi300x-run2", or
  // null. Every run so far was taken at 4 000-token prompts, 200 out, BF16 KV.
  function measuredGeometryMatches(state, data) {
    if (data && state.model && state.model !== data.measured.model) return null;
    const geometry = state.prompt_tokens >= 3500 && state.prompt_tokens <= 4500 &&
      state.output_tokens >= 150 && state.output_tokens <= 250 &&
      state.kv_dtype_bytes === 2;
    if (!geometry) return null;
    if (state.accelerator === "l40s" || state.accelerator === "l40s-run1") return "l40s";
    // Both MI300X runs had prefix caching off: their points say nothing at h > 0.
    if (state.hit_rate > 0.05) return null;
    // Run 2 was a second droplet and ran faster: each fit is drawn with its own run.
    if (state.accelerator === "mi300x-run2") return "mi300x-run2";
    if (state.accelerator === "mi300x" || state.accelerator === "mi300x-run1") return "mi300x";
    return null;
  }

  function seats(data, state, point, opts) {
    opts = opts || {};
    const legend = [];
    if (!point.seats) {
      return { svg: el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" },
        el("text", { class: "axis", x: W / 2, y: H / 2, "text-anchor": "middle" },
           "no operating point to draw: " + point.error)), legend, hidden: null, note: null };
    }
    const a = data.accelerators[state.accelerator];
    const m = R.withKvDtype(data.models[state.model], state.kv_dtype_bytes);
    const ctx = state.prompt_tokens + state.output_tokens;
    const targetMs = state.tpot_target * 1e3;
    const fit = interferenceFit(data, state);
    const showMeasured = measuredGeometryMatches(state, data);
    const onMi = showMeasured === "mi300x" || showMeasured === "mi300x-run2";
    const mi = showMeasured === "mi300x-run2" ? data.measured.mi300x_run2_seats
      : data.measured.mi300x_run1_seats;
    const measuredMaxN = showMeasured === "l40s" ? 56
      : onMi ? Math.max(...mi.levels.map((lv) => lv.concurrency)) : 0;

    const nMax = Math.max(16, Math.ceil(Math.max(point.seats.by_capacity, point.seats.by_latency, measuredMaxN) * 1.12));
    const step = (n) => R.tpotFloor(m, a, n, ctx).seconds * 1e3;
    const served = (n) => step(n) + (1 - state.hit_rate) * (fit.slope_s_per_seat * n + fit.intercept_s) * 1e3;
    let yMax = Math.max(targetMs * 1.4, step(nMax) * 1.05);
    if (showMeasured === "l40s") yMax = Math.max(yMax, 95);
    const sx = (n) => ML + (n / nMax) * PW;
    const sy = (ms) => MT + PH - (Math.min(ms, yMax) / yMax) * PH;

    let g = frame(niceTicks(nMax, 8).filter((v) => v > 0), niceTicks(yMax, 6),
                  "seats: people served at once", "ms per token", sx, sy, fmtMs);

    // the target, and the two limits
    g += el("line", { class: "target", x1: ML, x2: ML + PW, y1: sy(targetMs), y2: sy(targetMs) });
    g += el("text", { class: "axis note", x: ML + PW - 4, y: sy(targetMs) - 5, "text-anchor": "end" },
            `your promise, ${fmtMs(targetMs)} ms`);
    for (const [n, name] of [[point.seats.by_latency, "time limit"], [point.seats.by_capacity, "room limit"]]) {
      if (n <= 0 || n > nMax) continue;
      g += el("line", { class: "limit", x1: sx(n), x2: sx(n), y1: MT, y2: MT + PH });
      // A tick near the right edge would have its label run off the picture:
      // flip it to the left of its own line rather than clip it.
      const wide = `${name} ${n}`.length * 5.6;
      const flip = sx(n) + 4 + wide > ML + PW;
      g += el("text", { class: "axis note", x: sx(n) + (flip ? -4 : 4), "text-anchor": flip ? "end" : "start",
                        y: MT + 12 + (name[0] === "r" ? 14 : 0) }, `${name} ${n}`);
    }

    // the decode step, and the served step
    const pts = [];
    for (let n = 1; n <= nMax; n++) pts.push(`${sx(n).toFixed(1)},${sy(step(n)).toFixed(1)}`);
    g += el("polyline", { class: "line-model", points: pts.join(" ") });
    legend.push({ swatch: "line-model", label: "alone on the card",
                  tip: "step(n): the weights read once per step, plus every seat's KV, over the memory bus." });
    if (fit && state.hit_rate < 0.999 && opts.served !== false) {
      const from = Math.max(1, Math.ceil(-fit.intercept_s / fit.slope_s_per_seat));
      const sp = [];
      // the served step leaves the picture through the top rather than
      // running flat along it: a clamped line would read as a plateau
      for (let n = from; n <= nMax; n++) { const v = served(n); if (v > yMax) break; sp.push(`${sx(n).toFixed(1)},${sy(v).toFixed(1)}`); }
      g += el("polyline", { class: "line-served", points: sp.join(" ") });
      legend.push({ swatch: "line-served", label: `with newcomers arriving, ${Math.round(state.hit_rate * 100)} % cached`,
                    tip: "step(n) + (1 − h)·I(n): the decode step plus the prefill of arriving prompts, from the run-1 interference fit -- an interpolation over 13-32 seats, not a law." });
    }

    // the two answers, on the promise line: the solid line's crossing is what
    // the card can hold, the dashed line's what a service can promise
    if (point.seats.max_num_seqs > 0 && point.seats.max_num_seqs <= nMax) {
      g += el("circle", { class: "pt-now", cx: sx(point.seats.max_num_seqs), cy: sy(targetMs), r: 5.5,
                          "data-tip": `${point.seats.max_num_seqs} seats: the card can hold this many at ${fmtMs(targetMs)} ms, bound by ${point.seats.bound_by === "latency" ? "time" : "room"}` });
      legend.push({ swatch: "pt-now", label: "the card can hold", tip: "max_num_seqs: the smaller of the time limit and the room limit." });
    }
    if (fit && point.service && point.service.seats_shipped > 0 && point.service.seats_shipped <= nMax && opts.served !== false) {
      g += el("circle", { class: "pt-service", cx: sx(point.service.seats_shipped), cy: sy(targetMs), r: 5.5,
                          "data-tip": `${point.service.seats_shipped} seats: what a service can promise once newcomers' prefill is priced in` });
      legend.push({ swatch: "pt-service", label: "you can safely promise", tip: "The seat count with the prefill of arriving prompts priced into every decode step, from the run-1 interference fit." });
    }

    // measured points, only inside the runs' geometry
    let hidden = null, note = null;
    if (onMi) {
      let above = 0;
      for (const lv of mi.levels) {
        if (lv.concurrency > nMax) continue;
        if (lv.p99_tpot_ms > yMax) { above++; continue; }
        g += el("circle", { class: "pt-h0", cx: sx(lv.concurrency), cy: sy(lv.p99_tpot_ms), r: 4.5,
                            "data-tip": `${mi.label}, ${lv.concurrency} seats: TPOT p99 ${lv.p99_tpot_ms.toFixed(1)} ms, median ITL ${lv.median_itl_ms.toFixed(1)} ms, ${lv.max_running} running at most` });
        if (lv.concurrency <= mi.decode_step_through) {
          g += el("circle", { class: "pt-run1", cx: sx(lv.concurrency), cy: sy(lv.median_itl_ms), r: 3.5,
                              "data-tip": `${mi.label}, ${lv.concurrency} seats: median ITL ${lv.median_itl_ms.toFixed(1)} ms -- the decode step itself` });
        }
      }
      legend.push({ swatch: "pt-h0", label: "measured TPOT p99",
                    tip: `TPOT p99 measured in ${mi.label}, 2026-09-27, prefix caching off.` });
      legend.push({ swatch: "pt-run1", label: "measured decode step alone",
                    tip: `Median ITL on the MI300X, drawn only through ${mi.decode_step_through} seats: above that every step carries a prefill chunk and the median is no longer a decode step.` });
      const lv = mi.levels;
      const cross = lv.findIndex((x) => x.p99_tpot_ms > targetMs);
      const where = cross < 0 ? `never crossed ${fmtMs(targetMs)} ms in the sweep`
        : cross === 0 ? `was already over ${fmtMs(targetMs)} ms at ${lv[0].concurrency} seat`
        : `crossed ${fmtMs(targetMs)} ms between ${lv[cross - 1].concurrency} and ${lv[cross].concurrency} seats`;
      const running = Math.max(...lv.map((x) => x.max_running));
      // The cause is named only where the crossing row's median ITL is still a
      // decode step and under the target: then the tail is newcomers' prefill.
      const why = cross > 0 && lv[cross].concurrency <= mi.decode_step_through &&
        lv[cross].median_itl_ms < targetMs ? ", through newcomers' prefill rather than the decode step" : "";
      note = `Measured, TPOT p99 ${where}${why}; the pool and the cap were never reached, because a ${String(mi.max_num_batched_tokens).replace(/\B(?=(\d{3})+(?!\d))/g, " ")}-token step budget held at most ${running} running.` +
        (above ? ` ${above} measured points sit above the top of this picture, at up to ${fmtMs(Math.max(...mi.levels.map((lv) => lv.p99_tpot_ms)))} ms.` : "");
    } else if (showMeasured === "l40s") {
      const r3 = data.measured.run3_seats;
      for (const series of r3.series) {
        const cls = series.hit_rate > 0.5 ? "pt-h80" : "pt-h0";
        for (const lv of series.levels) {
          if (lv.concurrency > nMax) continue;
          const tip = `run 3, h = ${series.hit_rate.toFixed(1)}, ${lv.concurrency} seats: TPOT p99 ${lv.p99_tpot_ms.toFixed(1)} ms, median ITL ${lv.median_itl_ms.toFixed(1)} ms`;
          if (series.hit_rate > 0.5) {
            g += el("rect", { class: cls, x: sx(lv.concurrency) - 4, y: sy(lv.p99_tpot_ms) - 4, width: 8, height: 8, "data-tip": tip });
          } else {
            g += el("circle", { class: cls, cx: sx(lv.concurrency), cy: sy(lv.p99_tpot_ms), r: 4.5, "data-tip": tip });
          }
        }
      }
      for (const lv of data.measured.run1_decode.levels) {
        if (lv.ran > nMax) continue;
        g += el("circle", { class: "pt-run1", cx: sx(lv.ran), cy: sy(lv.decode_step_ms), r: 3.5,
                            "data-tip": `run 1, ${lv.ran} seats: median ITL ${lv.decode_step_ms.toFixed(1)} ms -- the decode step itself` });
      }
      legend.push({ swatch: "pt-h0", label: "measured, cold cache",
                    tip: "TPOT p99 measured on 2026-08-30, cold prefix cache." });
      legend.push({ swatch: "pt-h80", label: "measured, 80 % cached",
                    tip: "TPOT p99 measured on 2026-08-30, with 80 % of each prompt already in the prefix cache." });
      legend.push({ swatch: "pt-run1", label: "measured decode step alone",
                    tip: "Median ITL measured on 2026-08-18: the decode step by itself, with no arriving prompt landing in it." });
      note = (fit && opts.served !== false)
        ? (Math.abs(state.hit_rate - 0.8) < 0.05
            ? "The squares sit under the dashed line: the fit has the mechanism right and the seat count low — run 3 measured 37.8 seats where this line says 24.3. The model's honest miss."
            : "Set the repeats to 80 % to lay the dashed line over the orange squares: that is the comparison run 3 was rented for.")
        : null;
    } else {
      hidden = state.model && state.model !== data.measured.model
        ? `measured points hidden: every run decoded ${data.models[data.measured.model].name}, and a point measured on one architecture says nothing about another. This model is predicted only.`
        : "measured points hidden: every run was at 4 000-token prompts, 200 output tokens, BF16 KV, on an L40S or an MI300X, the MI300X with nothing cached. Move the sliders there to see them.";
    }
    return { svg: el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" }, g), legend, hidden, note,
             caption: "The more people, the slower each token; where a line meets your promise is the seat count." };
  }

  function cost(data, state, point) {
    const legend = [];
    const tMax = Math.max(300, state.tpot_target * 1e3 * 1.2);
    const tMin = 5;
    const xs = [];
    for (let i = 0; i <= 80; i++) xs.push(tMin + (tMax - tMin) * i / 80);
    const hw = [], sv = [];
    let yMin = Infinity, yMax = 0;
    for (const t of xs) {
      const p = R.whatIfPoint(data, Object.assign({}, state, { tpot_target: t / 1e3 }));
      const c = p.cost_per_1m_output_tokens, s = p.service && p.service.cost_per_1m_output_tokens;
      hw.push([t, c]); sv.push([t, s]);
      for (const v of [c, s]) if (v) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
    }
    if (!isFinite(yMin)) {
      return { svg: el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" },
        el("text", { class: "axis", x: W / 2, y: H / 2, "text-anchor": "middle" },
           "no target between 5 and 300 ms has an operating point here")), legend, hidden: null, note: null };
    }
    const lo = Math.pow(10, Math.floor(Math.log10(yMin))), hi = Math.pow(10, Math.ceil(Math.log10(yMax)));
    const sx = (t) => ML + ((t - tMin) / (tMax - tMin)) * PW;
    const sy = (v) => MT + PH - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * PH;
    const yTicks = [];
    for (let d = lo; d <= hi * 1.001; d *= 10) for (const k of [1, 2, 5]) if (d * k <= hi * 1.001) yTicks.push(d * k);
    let g = "";
    for (const v of yTicks) {
      g += el("line", { class: "grid", x1: ML, x2: ML + PW, y1: sy(v), y2: sy(v) });
      g += el("text", { class: "axis", x: ML - 8, y: sy(v) + 4, "text-anchor": "end" }, fmtUsd(v));
    }
    for (const v of niceTicks(tMax, 8)) {
      if (v < tMin) continue;
      g += el("text", { class: "axis", x: sx(v), y: MT + PH + 18, "text-anchor": "middle" }, String(v));
    }
    g += el("line", { class: "axis-line", x1: ML, x2: ML + PW, y1: MT + PH, y2: MT + PH });
    g += el("text", { class: "axis label", x: ML + PW / 2, y: H - 6, "text-anchor": "middle" }, "your promise, ms per token");
    g += el("text", { class: "axis label", x: 14, y: MT + PH / 2, "text-anchor": "middle",
                      transform: `rotate(-90 14 ${MT + PH / 2})` }, "$ per million tokens, log axis");

    const path = (rows) => rows.filter(([, v]) => v).map(([t, v]) => `${sx(t).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
    g += el("polyline", { class: "line-model", points: path(hw) });
    legend.push({ swatch: "line-model", label: "every seat filled",
                  tip: `Every seat the target permits, all of them filled, at $${(point.inputs.hourly_rate).toFixed(2)}/h. Nothing real reaches it.` });
    if (sv.some(([, v]) => v)) {
      g += el("polyline", { class: "line-served", points: path(sv) });
      legend.push({ swatch: "line-served", label: `what a service can promise, ${Math.round(state.hit_rate * 100)} % cached`,
                    tip: "The seats left once the prefill of arriving prompts is priced into everyone's decode step." });
    }
    const t0 = state.tpot_target * 1e3;
    const cur = (point.service && point.service.cost_per_1m_output_tokens) || point.cost_per_1m_output_tokens;
    g += el("line", { class: "limit", x1: sx(t0), x2: sx(t0), y1: MT, y2: MT + PH });
    if (cur) {
      g += el("circle", { class: "pt-now", cx: sx(t0), cy: sy(cur), r: 5 });
      g += el("text", { class: "axis note", x: sx(t0) + 8, y: sy(cur) - 8 }, `${fmtUsd(cur)} / 1M at ${fmtMs(t0)} ms`);
    } else {
      g += el("text", { class: "axis note", x: sx(t0) + 8, y: MT + 14 }, `no operating point at ${fmtMs(t0)} ms`);
    }
    const bt = data.slo_classes.batch.tpot_s * 1e3, it = data.slo_classes.interactive.tpot_s * 1e3;
    for (const [t, name] of [[it, "chat"], [bt, "batch"]]) {
      if (t > tMax) continue;
      g += el("text", { class: "axis note", x: sx(t), y: MT + PH - 6, "text-anchor": "middle" }, name);
    }
    return { svg: el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" }, g), legend, hidden: null,
             caption: "Relax the promise and the price falls, until nobody is reading anyway." };
  }

  root.Draw = { seats, cost, measuredGeometryMatches, interferenceFit };
})(typeof self !== "undefined" ? self : this);
