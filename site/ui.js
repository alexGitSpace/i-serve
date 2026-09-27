/* ui.js -- one situation, three steps, everything else folded away.
 *
 * State is one flat object in the URL hash, so an operating point is a link, and
 * every render is a pure function of it: Roofline.whatIfPoint() the numbers,
 * Draw.* the pictures, Advisor.evaluate() the branch. This file computes nothing
 * the Python does not; it chooses what a first-time reader sees, and the
 * situation is a sentence whose values open the real controls.
 */
(function () {
  "use strict";
  const data = window.MODEL_DATA, map = window.SYMPTOM_MAP;
  const R = window.Roofline;
  const $ = (id) => document.getElementById(id);
  const REPO = document.body.dataset.repo || "";
  const REPO_OK = Boolean(REPO) && !REPO.startsWith("<");
  const repoLink = (path) => REPO_OK ? REPO.replace(/\/$/, "") + "/blob/main/" + path : null;
  const docLink = (target) => repoLink("docs/" + target);

  // --- state -----------------------------------------------------------------
  const state = {
    accelerator: data.defaults.accelerator,
    model: data.defaults.model,
    prompt_tokens: data.defaults.prompt_tokens,
    output_tokens: data.defaults.output_tokens,
    hit_rate: data.defaults.hit_rate,
    slo_class: "interactive",   // which obligation is being read, not which number
    tpot_target: data.slo_classes.interactive.tpot_s,
    ttft_target: data.slo_classes.interactive.ttft_s,
    kv_dtype_bytes: data.defaults.kv_dtype_bytes,
    gpu_memory_utilization: data.defaults.gpu_memory_utilization,
    hourly_rate: null,
    pin: null,          // a second operating point, kept for comparison
  };
  const POINT_KEYS = ["accelerator", "model", "prompt_tokens", "output_tokens", "hit_rate", "tpot_target",
                      "ttft_target", "kv_dtype_bytes", "gpu_memory_utilization", "hourly_rate"];
  // The guided flow's four answers, and the expert form's raw readings. The
  // flow wins while the expert form is closed; the form wins once opened.
  const flow = { queue: "dk", seats: "dk", ttft: "dk", tpot: "dk", changed: false };
  const readings = {
    alert: "none", track: "stable", waiting: 0, running: 0, max_num_seqs: 256,
    kv_usage: null, preemptions_per_s: null, ttft_p99_ms: null, tpot_p99_ms: null,
    median_itl_ms: null, replicas: 1, max_replicas: 4, targets_down: 0, changed_recently: false,
  };
  // Links written when the page had three tabs carry `tab=`; the step it
  // named is scrolled to once, so an old link still lands where it pointed.
  let landing = null;
  const OLD_TABS = { slo: "step-speed", cost: "step-cost", spike: "step-trouble" };

  function readHash() {
    const h = new URLSearchParams(location.hash.replace(/^#/, ""));
    // A link shared before the class was part of the state carries only the two
    // targets. Adopt the class they spell, so an old batch link still reads as
    // a batch link rather than as an interactive one with a loose promise.
    if (!h.has("slo_class") && h.has("tpot_target") && h.has("ttft_target")) {
      for (const [name, c] of Object.entries(data.slo_classes)) {
        if (!c || typeof c !== "object") continue;
        if (Number(h.get("tpot_target")) === c.tpot_s && Number(h.get("ttft_target")) === c.ttft_s) state.slo_class = name;
      }
    }
    if (h.has("tab") && OLD_TABS[h.get("tab")]) landing = OLD_TABS[h.get("tab")];
    for (const k of Object.keys(state)) {
      if (!h.has(k)) continue;
      const v = h.get(k);
      if (k === "slo_class") { if (data.slo_classes[v]) state.slo_class = v; }
      else if (k === "pin") state.pin = v || null;
      else if (k === "accelerator") { if (data.accelerators[v]) state[k] = v; }
      else if (k === "model") { if (data.models[v]) state[k] = v; }
      else if (k === "hourly_rate") state[k] = v === "" ? null : Number(v);
      else if (k === "kv_dtype_bytes") state[k] = Number(v) === 1 ? 1 : 2;
      else state[k] = Number(v);
    }
  }
  function writeHash() {
    const h = new URLSearchParams();
    for (const [k, v] of Object.entries(state)) if (v !== null) h.set(k, String(v));
    const next = "#" + h.toString();
    if (location.hash !== next) history.replaceState(null, "", next);
  }

  // --- formatting --------------------------------------------------------------
  const msNum = (s) => {
    const x = s * 1e3;
    return Math.abs(x - Math.round(x)) < 1e-6 ? String(Math.round(x)) : (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
  };
  const ms = (s) => msNum(s) + " ms";
  const ordinal = (n) => { const r = n % 100; if (r >= 11 && r <= 13) return n + "th"; return n + ({1: "st", 2: "nd", 3: "rd"}[n % 10] || "th"); };
  const usd = (x) => "$" + x.toFixed(2);
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const pct = (h) => Math.round(h * 100) + " %";
  const escAttr = (x) => String(x).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  const P_MIN = 128, P_MAX = 32768;
  const promptToSlider = (p) => Math.round(1000 * Math.log2(p / P_MIN) / Math.log2(P_MAX / P_MIN));
  const sliderToPrompt = (v) => Math.round(P_MIN * Math.pow(2, (v / 1000) * Math.log2(P_MAX / P_MIN)));
  const cardShort = (key) => data.accelerators[key].name.replace(" (run 1 coefficients)", "").replace("NVIDIA ", "").replace("AMD Instinct ", "");
  const rateOf = (st) => st.hourly_rate === null ? data.accelerators[st.accelerator].hourly_rate : st.hourly_rate;
  // The batch class reads the same point as throughput and time to a million,
  // not milliseconds (docs/SLO.md sections 1-2); these helpers are the difference.
  const isBatch = () => state.slo_class === "batch";
  const perSec = (x) => (x >= 100 ? int(x) : x.toFixed(1));
  const spanFor = (tokens, tokensPerSec) => {
    const sec = tokens / tokensPerSec;
    if (sec < 90) return `${sec.toFixed(0)} s`;
    if (sec < 5400) return `${sec < 600 ? (sec / 60).toFixed(1) : Math.round(sec / 60)} min`;
    return `${(sec / 3600).toFixed(1)} h`;
  };
  // What the point actually delivers: the service figure where a run has
  // measured the interference, the hardware floor where none has.
  const delivered = (p) => {
    if (!p.seats || p.error) return null;
    const sv = p.service && p.service.seats_shipped > 0 ? p.service : null;
    return { seats: sv ? sv.seats_shipped : p.seats.max_num_seqs,
             rate: sv ? sv.aggregate_tokens_per_sec : p.aggregate_tokens_per_sec,
             cost: sv ? sv.cost_per_1m_output_tokens : p.cost_per_1m_output_tokens,
             from_service: Boolean(sv) };
  };
  // One cell of the answer row: the number, its unit, the sentence under it.
  function setStat(prefix, num, unit, text, badge) {
    $(prefix + "-num").innerHTML = `${num}${unit ? `<small>${unit}</small>` : ""}`;
    $(prefix + "-text").innerHTML = text || "";
    const b = $(prefix + "-badge");
    if (b) { b.className = "badge " + (badge ? badge.cls : ""); b.textContent = badge ? badge.text : ""; b.title = badge && badge.tip ? badge.tip : ""; }
  }

  // --- the rail: three steps, one page ----------------------------------------
  function bindSteps() {
    for (const b of document.querySelectorAll("[data-step]")) {
      b.addEventListener("click", () => $(b.dataset.step).scrollIntoView({ behavior: "smooth", block: "start" }));
    }
    let raf = 0;
    const mark = () => {
      raf = 0;
      let on = null;
      for (const id of Object.values(OLD_TABS)) {
        if ($(id).getBoundingClientRect().top <= 140) on = id;
      }
      for (const b of document.querySelectorAll("[data-step]")) b.classList.toggle("on", b.dataset.step === on);
    };
    window.addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(mark); }, { passive: true });
    mark();
  }

  // --- the sentence: values that open their controls --------------------------
  function closePops(except) {
    for (const b of document.querySelectorAll("[data-pop]")) {
      const pop = $(b.dataset.pop);
      if (pop === except) continue;
      pop.hidden = true; b.setAttribute("aria-expanded", "false");
    }
  }
  function bindPops() {
    for (const b of document.querySelectorAll("[data-pop]")) {
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        const pop = $(b.dataset.pop);
        const open = pop.hidden;
        closePops(pop);
        pop.hidden = !open; b.setAttribute("aria-expanded", String(open));
        if (open) { const f = pop.querySelector("select, input"); if (f) f.focus({ preventScroll: true }); }
      });
    }
    document.addEventListener("click", (e) => { if (!e.target.closest || !e.target.closest(".edit-wrap")) closePops(null); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePops(null); });
  }
  function bindCopy() {
    const b = $("copy-link");
    b.addEventListener("click", async () => {
      const url = location.href;
      let ok = false;
      try { await navigator.clipboard.writeText(url); ok = true; }
      catch (e) {
        const t = document.createElement("textarea");
        t.value = url; t.setAttribute("readonly", ""); t.style.position = "fixed"; t.style.left = "-9999px";
        document.body.appendChild(t); t.select();
        try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
        t.remove();
      }
      b.textContent = ok ? "Copied" : "Copy failed; use the address bar";
      setTimeout(() => { b.textContent = "Copy link"; }, 1800);
    });
  }
  // Links inside the answers that move the sliders: "Try it", "Switch card".
  function bindAnswerLinks() {
    document.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("a[data-set-hit], a[data-set-card]");
      if (!a) return;
      e.preventDefault();
      if (a.dataset.setHit !== undefined) state.hit_rate = Number(a.dataset.setHit);
      if (a.dataset.setCard !== undefined) { state.accelerator = a.dataset.setCard; state.hourly_rate = null; }
      update();
    });
  }

  // --- the promise: the one slider -----------------------------------------------
  function bindPromise() {
    $("tpot-range").addEventListener("input", () => { state.tpot_target = Number($("tpot-range").value) / 1e3; update(); });
    $("tpot-num").addEventListener("change", () => {
      const v = Math.round(Number($("tpot-num").value));
      if (v >= 1 && v <= 1000) state.tpot_target = v / 1e3;
      update();
    });
    for (const b of document.querySelectorAll("[data-preset]")) {
      b.addEventListener("click", () => {
        const c = data.slo_classes[b.dataset.preset];
        state.slo_class = b.dataset.preset;
        state.ttft_target = c.ttft_s; state.tpot_target = c.tpot_s; update();
      });
    }
  }
  function reflectPromise() {
    const t = Math.round(state.tpot_target * 1e3);
    $("tpot-range").value = t; $("tpot-num").value = t;
    for (const b of document.querySelectorAll("[data-preset]")) b.classList.toggle("active", b.dataset.preset === state.slo_class);
    const tight = ms(data.slo_classes.interactive.tpot_s);
    $("tpot-label").textContent = isBatch() ? "ms per token, stall guard" : "ms per token";
    $("slo-question").textContent = isBatch()
      ? "How many tokens must land before the deadline?"
      : "How fast should each word appear?";
    $("slo-why").textContent = isBatch()
      ? "Nobody is watching a token arrive, so the per-token threshold is only a guard against a stall. What matters is tokens per second, and how long a million takes."
      : `People reading a chat notice anything slower than about ${tight} per token. A batch job only needs its deadline, so it can trade speed for seats.`;
    $("cost-question").textContent = isBatch()
      ? "What does a million tokens cost, and how long do they take?"
      : "What does a million tokens cost at that speed?";
    $("cost-why-line").textContent = isBatch()
      ? "Only tokens inside the stall guard count. Loosening the guard lets more seats share the hourly bill, until room rather than time sets the seat count."
      : "Only tokens that arrived on time count. A faster promise means fewer people share the hourly bill, so each token costs more.";
    $("hw-lab").textContent = isBatch() ? "the card delivers" : "the card can hold";
  }

  // --- what the per-token threshold is worth to a batch pipeline --------------
  // What the 200 ms stall guard buys over the interactive threshold -- nothing
  // on a card where room binds (docs/SLO.md section 1).
  function guardLine(point) {
    const here = delivered(point);
    if (!here) return "";
    const tight = data.slo_classes.interactive;
    const alt = delivered(R.whatIfPoint(data, Object.assign({}, state,
      { tpot_target: tight.tpot_s, ttft_target: tight.ttft_s })));
    const t = ms(state.tpot_target), tightMs = ms(tight.tpot_s);
    const lever = point.seats.bound_by === "capacity"
      ? `room binds first, so the lever is <b>capacity</b> (FP8 KV, or a shorter reserved context) and not the threshold`
      : `time still binds, so here the threshold is the lever after all`;
    const head = `The ${t} threshold is a <b>stall guard</b>, not a promise: in the batch class nobody is watching an individual token arrive.`;
    if (!alt) return `${head} At the interactive ${tightMs} this card has no operating point at all.`;
    const gain = here.rate / alt.rate - 1;
    if (gain > 0.01) return `${head} Loosening it from the interactive ${tightMs} bought <b>${Math.round(gain * 100)} % more throughput</b> and no more: ${lever}.`;
    if (gain < -0.01) return `${head} And this one is tighter than the interactive ${tightMs}: it costs <b>${Math.round(-gain * 100)} % of the throughput</b> to hold a metric no one in this class reads.`;
    return `${head} Loosening it from the interactive ${tightMs} buys <b>nothing at all</b> here: ${lever}.`;
  }

  // --- step 1: speed -------------------------------------------------------------
  function renderSlo(point) {
    const card = cardShort(state.accelerator);
    const prompt = int(state.prompt_tokens);
    const t = ms(state.tpot_target);
    const seatsUnit = isBatch() ? "seats" : "people at once";

    // the hardware
    if (!point.seats) {
      setStat("hw", "—", "", `On an ${card} this model does not fit at all: ${point.error}`);
    } else if (point.seats.max_num_seqs === 0) {
      setStat("hw", "0", seatsUnit, `Even one person alone would get tokens slower than ${t} on this card. Relax the promise, or take a card with a faster memory bus.`);
    } else {
      const s = point.seats;
      const why = s.bound_by === "latency"
        ? `Each extra seat slows every token a little, and the ${ordinal(s.by_latency + 1)} would break the ${t} promise. Room would hold ${int(s.by_capacity)}.`
        : `Room binds: the KV pool holds ${int(s.by_capacity)} seats' memory at this prompt length, while time alone would allow ${int(s.by_latency)}.`;
      if (isBatch()) {
        setStat("hw", perSec(point.aggregate_tokens_per_sec), "tokens per second",
          `${s.max_num_seqs} seats, every one filled: a million tokens every ${spanFor(1e6, point.aggregate_tokens_per_sec)}. ${why}`);
      } else {
        setStat("hw", String(s.max_num_seqs), seatsUnit, why);
      }
    }

    // the service
    const sv = point.service;
    if (!point.seats || point.error) {
      setStat("svc", "—", "", "");
    } else if (sv === null) {
      const text = state.model !== data.measured.model
        ? `Not derivable for this model: the interference fit was measured decoding ${data.models[data.measured.model].name}, and prefill lands differently in another architecture's decode step.`
        : data.accelerators[state.accelerator].measured
          ? `Not derivable on this card yet: its run measured the interference, but the model written from it (two rows, after the fact) is not registered as a fit here.`
          : `Not derivable on this card yet: it needs a run to measure how much of each newcomer's prefill lands in the other seats' steps.`;
      setStat("svc", "?", data.accelerators[state.accelerator].measured ? "needs a fit" : "needs a run", text, null);
    } else {
      const badge = { cls: "measured", text: "measured", tip: sv.fit ? sv.fit.provenance : "" };
      if (sv.seats_shipped === 0) {
        setStat("svc", "0", seatsUnit, `Once newcomers' prefill is priced in, no seat meets the promise: the interference alone eats the budget.`, badge);
      } else if (isBatch()) {
        const text = sv.seats_shipped < point.seats.max_num_seqs
          ? `${sv.seats_shipped} seats once newcomers' prefill is priced into everyone's decode step; the hardware alone would run ${point.seats.max_num_seqs}.`
          : `Nothing is lost to newcomers here: room set the seat count, and interference would have to eat the whole budget before it moved.`;
        setStat("svc", perSec(sv.aggregate_tokens_per_sec), "tokens per second", text, badge);
      } else {
        let text = `Newcomers' prompts have to be read first, and that reading slows everyone else's tokens. ` +
          (state.hit_rate > 0 ? `${pct(state.hit_rate)} of each prompt is already cached, so less of it is read.` : `Nothing is cached, so every prompt is read in full.`);
        if (state.hit_rate < 0.8 - 1e-9) {
          const alt = R.whatIfPoint(data, Object.assign({}, state, { hit_rate: 0.8 }));
          if (alt.service && alt.service.seats_shipped > 0) text += ` With 80 % repeats: ${alt.service.seats_shipped}.`;
        }
        if (sv.note) text += ` A floor, not an estimate: run 3 measured 37.8 seats where this line said 24.3.`;
        setStat("svc", String(sv.seats_shipped), seatsUnit, text, badge);
      }
    }

    $("slo-guard").innerHTML = isBatch() ? guardLine(point) : "";

    // the first token
    const tf = point.ttft_floor, tu = point.ttft_floor_uncached;
    const target = ms(state.ttft_target);
    const asked = `ms · you asked ${target}`;
    if (tf.seconds > state.ttft_target) {
      if (state.hit_rate > 0 && tu.seconds <= state.ttft_target) {
        setStat("ttft", msNum(tu.seconds), asked,
          `Reading a ${prompt}-token prompt cold would take ${ms(tf.seconds)}, over the target; with ${pct(state.hit_rate)} cached only ${int(tu.uncached_tokens)} tokens are left to read, and that fits.`, null);
      } else {
        setStat("ttft", msNum(tf.seconds), asked,
          `Reading a ${prompt}-token prompt takes at least this long on this card: the prompt alone does not fit. Shorter prompts, or a cache serving most of them.`,
          { cls: "over", text: "too slow" });
      }
    } else {
      setStat("ttft", msNum(tf.seconds), asked, `Reading the ${prompt}-token prompt fits inside the first-word target.`, null);
    }
    $("q-ttft-small").textContent = `row 2 · TTFT p99 against its floor, ${ms(tu.seconds)}`;
    $("q-tpot-small").textContent = `row 2 · TPOT p99 against the promise, ${t}`;
  }

  // --- step 2: cost -----------------------------------------------------------------
  function renderCost(point) {
    const card = cardShort(state.accelerator);
    const rate = rateOf(state);
    const t = ms(state.tpot_target);
    const unit = "per million tokens";
    if (!point.seats || point.error) {
      setStat("cost", "—", "", `No operating point to price at ${t} on an ${card}${point.error ? `: ${point.error}` : "."}`);
      $("alt1-lab").textContent = "hardware alone"; setStat("alt1", "—", "", "");
    } else {
      const sv = point.service, hw = point.cost_per_1m_output_tokens;
      const useSvc = sv && sv.seats_shipped > 0;
      const c = useSvc ? sv.cost_per_1m_output_tokens : hw;
      const n = useSvc ? sv.seats_shipped : point.seats.max_num_seqs;
      const rateTxt = `$${rate.toFixed(2)} an hour`;
      setStat("cost", usd(c), unit, isBatch()
        ? `${n} seats sharing ${rateTxt}; a million tokens every ${spanFor(1e6, useSvc ? sv.aggregate_tokens_per_sec : point.aggregate_tokens_per_sec)}.`
        : `${n} people sharing ${rateTxt}, every token inside the ${t} promise.`);
      if (useSvc) {
        $("alt1-lab").textContent = "hardware alone";
        setStat("alt1", usd(hw), `${point.seats.max_num_seqs} ${isBatch() ? "seats" : "people at once"}`,
          `The gap is newcomers' prefill landing in everyone's decode step.`);
      } else if (sv === null) {
        $("alt1-lab").textContent = "a service could promise";
        setStat("alt1", "?", state.model === data.measured.model && data.accelerators[state.accelerator].measured ? "needs a fit" : "needs a run", state.model !== data.measured.model
          ? `This is a floor with every permitted seat filled. What a service could promise on this model needs a run: the fit on the page was measured decoding ${data.models[data.measured.model].name}.`
          : data.accelerators[state.accelerator].measured
            ? `This is a floor with every permitted seat filled. What a service could promise on this card needs an interference fit registered from its run.`
            : `This is a floor with every permitted seat filled. What a service could promise on this card needs a run.`);
      } else {
        $("alt1-lab").textContent = "hardware alone";
        setStat("alt1", usd(hw), `${point.seats.max_num_seqs} seats`, `No seat survives once interference is priced in; the figure on the left is the hardware floor.`);
      }
    }

    // the third cell: the one move that changes the picture most, computed
    // with the same arithmetic -- more repeats while they are below 80 %,
    // the other card once they are not.
    if (state.hit_rate < 0.8 - 1e-9) {
      $("alt2-lab").innerHTML = "the cheapest move";
      const d = delivered(R.whatIfPoint(data, Object.assign({}, state, { hit_rate: 0.8 })));
      if (d) {
        setStat("alt2", usd(d.cost), "with 80 % repeats",
          `Same card, same promise, ${d.seats} ${isBatch() ? "seats" : "people"}. Repeats come from your traffic (a shared system prompt, the same examples), not from hardware. <a href="#" data-set-hit="0.8">Try it</a>`);
      } else {
        setStat("alt2", "—", "", `Even with 80 % repeats nothing fits at ${t} on this card.`);
      }
    } else {
      const other = state.accelerator.startsWith("mi300x") ? "l40s-run1" : "mi300x-run1";
      const oc = data.accelerators[other];
      const alt = R.whatIfPoint(data, Object.assign({}, state, { accelerator: other, hourly_rate: null }));
      const d = delivered(alt);
      // The badge speaks for the number, not the card: a floor stays a floor
      // even when the card's coefficients were measured.
      const floorOnly = d && !d.from_service;
      const tag = floorOnly ? "floor" : (oc.measured ? "measured" : "prior");
      $("alt2-lab").innerHTML = `another card <span class="badge ${tag === "measured" ? "measured" : "prior"}">${tag}</span>`;
      if (d) {
        setStat("alt2", usd(d.cost), `on an ${cardShort(other)} at $${oc.hourly_rate.toFixed(2)} an hour`,
          (floorOnly ? `A floor with every permitted seat filled, not what a service could promise: that needs this card's interference measured. ` : ``) +
          (oc.measured ? `Coefficients ${oc.provenance}. ` : `Nobody has run this card here yet: a spec-sheet guess, and a prior can be badly wrong. `) +
          `<a href="#" data-set-card="${other}">Switch card</a>`);
      } else {
        setStat("alt2", "—", `on an ${cardShort(other)}`, `No operating point at ${t} on the other card.`);
      }
    }
  }

  // --- pictures ----------------------------------------------------------------
  function renderFigure(id, result) {
    const fig = $(id);
    fig.querySelector(".plot").innerHTML = result.svg;
    fig.querySelector(".legend").innerHTML = result.legend.map((l) =>
      `<li${l.tip ? ` data-tip="${escAttr(l.tip)}" tabindex="0"` : ""}><span class="swatch ${l.swatch}"></span><span class="lbl">${l.label}</span></li>`).join("") +
      (result.caption ? `<li class="caption">${result.caption}</li>` : "");
    const note = fig.querySelector(".hidden-note");
    note.hidden = !result.hidden;
    note.textContent = result.hidden || "";
    const finding = fig.querySelector(".finding");
    finding.hidden = !result.note;
    finding.textContent = result.note || "";
  }
  function bindTooltips() {
    const tip = $("tooltip");
    let pinnedUntil = 0;
    const show = (t, x, y) => {
      tip.textContent = t.dataset.tip; tip.hidden = false;
      const left = Math.min(x + 12, document.documentElement.clientWidth - tip.offsetWidth - 12);
      tip.style.left = Math.max(8, left) + "px"; tip.style.top = (y + 14) + "px";
    };
    document.addEventListener("mousemove", (e) => {
      if (Date.now() < pinnedUntil) return;
      const t = e.target.closest && e.target.closest("[data-tip]");
      if (!t) { tip.hidden = true; return; }
      show(t, e.pageX, e.pageY);
    });
    // A tap has no hover: keep the explanation up for a few seconds.
    document.addEventListener("click", (e) => {
      const t = e.target.closest && e.target.closest(".term, .legend li[data-tip]");
      if (!t) return;
      e.preventDefault();
      show(t, e.pageX, e.pageY); pinnedUntil = Date.now() + 4000;
      setTimeout(() => { if (Date.now() >= pinnedUntil) tip.hidden = true; }, 4100);
    });
  }

  // --- plain-language terms on hover -----------------------------------------
  // Every spelling in TERMS becomes a <span class="term" data-tip>, in text nodes
  // only and never nested; rebuilt on every render from the rendered text.
  const TERM_INDEX = (() => {
    const entries = [];
    for (const t of window.TERMS || []) for (const m of t.match) entries.push({ m, t });
    entries.sort((a, b) => b.m.length - a.m.length);
    const esc = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp("(?<![\\w-])(" + entries.map((e) => esc(e.m)).join("|") + ")(?![\\w-])", "gi");
    const byLower = new Map(entries.map((e) => [e.m.toLowerCase(), e.t]));
    return { re, byLower };
  })();
  const SKIP = new Set(["CODE", "BUTTON", "SELECT", "OPTION", "INPUT", "OUTPUT", "SVG", "A", "SCRIPT", "STYLE", "H1", "H2", "H3"]);
  const SKIP_CLASS = ["term", "rname", "legend", "num", "lab", "lede"];
  function markTerms(root) {
    for (const old of root.querySelectorAll("span.term")) old.replaceWith(document.createTextNode(old.textContent));
    root.normalize();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        for (let el = node.parentElement; el && el !== root; el = el.parentElement) {
          if (SKIP.has(el.tagName) || el.tagName === "svg") return NodeFilter.FILTER_REJECT;
          for (const c of SKIP_CLASS) if (el.classList.contains(c)) return NodeFilter.FILTER_REJECT;
        }
        return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    const seen = new WeakMap();      // block element -> Set of glossary keys already marked in it
    for (const node of nodes) {
      const text = node.nodeValue;
      TERM_INDEX.re.lastIndex = 0;
      if (!TERM_INDEX.re.test(text)) continue;
      TERM_INDEX.re.lastIndex = 0;
      const block = node.parentElement.closest("p, li, td, th, figcaption, label, .cell, .qtext, summary, .stat") || root;
      if (!seen.has(block)) seen.set(block, new Set());
      const marked = seen.get(block);
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = TERM_INDEX.re.exec(text))) {
        const t = TERM_INDEX.byLower.get(m[1].toLowerCase());
        if (!t || marked.has(t.glossary)) continue;
        marked.add(t.glossary);
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const span = document.createElement("span");
        span.className = "term"; span.textContent = m[1];
        span.dataset.tip = t.plain; span.tabIndex = 0;
        frag.appendChild(span);
        last = m.index + m[1].length;
      }
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.replaceWith(frag);
    }
  }

  // --- keep one operating point, compare with the current one -----------------
  const encodePin = () => POINT_KEYS.map((k) => k + "=" + (state[k] === null ? "" : state[k])).join(";");
  function decodePin(text) {
    const out = {};
    for (const part of text.split(";")) {
      const [k, v] = part.split("=");
      if (!POINT_KEYS.includes(k)) return null;
      if (k === "accelerator") { if (!data.accelerators[v]) return null; out[k] = v; }
      else if (k === "model") { if (!data.models[v]) return null; out[k] = v; }
      else if (k === "hourly_rate") out[k] = v === "" ? null : Number(v);
      else out[k] = Number(v);
    }
    return POINT_KEYS.every((k) => k in out) ? out : null;
  }
  function bindPin() {
    for (const b of document.querySelectorAll("[data-pin]")) {
      b.addEventListener("click", () => { state.pin = state.pin ? null : encodePin(); update(); });
    }
  }
  function describe(st) {
    return `${data.models[st.model].name} · ${cardShort(st.accelerator)} · ${int(st.prompt_tokens)} + ${int(st.output_tokens)} tokens · cache ${Math.round(st.hit_rate * 100)} % · ${st.kv_dtype_bytes === 1 ? "FP8" : "BF16"} KV · ${ms(st.tpot_target)} · $${rateOf(st).toFixed(2)}/h`;
  }
  function renderCompare(point) {
    const pinned = state.pin ? decodePin(state.pin) : null;
    for (const b of document.querySelectorAll("[data-pin]")) b.textContent = pinned ? "Forget the kept one" : "Keep this to compare";
    const boxes = document.querySelectorAll(".compare");
    if (!pinned) { for (const b of boxes) { b.hidden = true; b.innerHTML = ""; } return; }
    const a = R.whatIfPoint(data, pinned), b = point;
    const seats = (p) => p.seats && !p.error ? String(p.seats.max_num_seqs) : "—";
    const svc = (p) => p.service ? String(p.service.seats_shipped) : (p.seats && !p.error ? "not derivable" : "—");
    const cost = (p) => {
      if (!p.seats || p.error) return "—";
      const v = p.service && p.service.seats_shipped ? p.service.cost_per_1m_output_tokens : p.cost_per_1m_output_tokens;
      return usd(v) + (p.service && p.service.seats_shipped ? "" : " (hardware floor)");
    };
    const ttft = (p) => `${ms(p.ttft_floor.seconds)}${p.inputs.hit_rate > 0 ? ` / ${ms(p.ttft_floor_uncached.seconds)} cached` : ""}`;
    const same = describe(pinned) === describe(state);
    const rows = [
      ["the card can hold", seats(a), seats(b)],
      ["you can safely promise", svc(a), svc(b)],
      ["$ per million tokens", cost(a), cost(b)],
      ["first word after, at least", ttft(a), ttft(b)],
    ];
    const html = `<table><thead><tr><th></th><th>kept<br><small>${describe(pinned)}</small></th><th>now<br><small>${describe(state)}</small></th></tr></thead><tbody>` +
      rows.map(([k, x, y]) => `<tr><th>${k}</th><td>${x}</td><td class="${x === y ? "" : "diff"}">${y}</td></tr>`).join("") +
      `</tbody></table>` + (same ? `<p class="quiet small">Same point on both sides: change the card, the repeats, the KV dtype or the promise to see what moves.</p>` : "");
    for (const box of boxes) { box.hidden = false; box.innerHTML = html; }
  }

  // --- the situation's controls -------------------------------------------------
  function bindInputs() {
    const acc = $("accelerator");
    for (const [key, card] of Object.entries(data.accelerators)) {
      const o = document.createElement("option");
      o.value = key; o.textContent = card.name + (card.measured ? "" : " (prior)"); acc.appendChild(o);
    }
    acc.addEventListener("change", () => { state.accelerator = acc.value; state.hourly_rate = null; update(); });
    const mod = $("model");
    const modelKeys = Object.keys(data.models).sort(
      (a, b) => (a === data.defaults.model ? -1 : b === data.defaults.model ? 1 : a < b ? -1 : 1));
    for (const key of modelKeys) {
      const m = data.models[key];
      const o = document.createElement("option");
      o.value = key; o.textContent = m.name + (m.served_by_this_stack ? "" : " (predicted only)");
      mod.appendChild(o);
    }
    mod.addEventListener("change", () => { state.model = mod.value; update(); });
    const pair = (rangeId, numId, key, fromRange, parse) => {
      const r = $(rangeId), n = $(numId);
      r.addEventListener("input", () => { state[key] = fromRange(Number(r.value)); update(); });
      n.addEventListener("change", () => { const v = parse(n.value); if (isFinite(v)) state[key] = v; update(); });
    };
    pair("prompt-range", "prompt", "prompt_tokens", sliderToPrompt, (s) => Math.min(P_MAX, Math.max(1, Math.round(Number(s)))));
    pair("output-range", "output", "output_tokens", (v) => v, (s) => Math.min(8192, Math.max(1, Math.round(Number(s)))));
    pair("hit-range", "hit", "hit_rate", (v) => v / 100, (s) => Math.min(1, Math.max(0, Number(s) / 100)));
    pair("gmu-range", "gmu", "gpu_memory_utilization", (v) => v / 100, (s) => Math.min(0.99, Math.max(0.05, Number(s) / 100)));
    $("ttft").addEventListener("change", () => { const v = Number($("ttft").value); if (v > 0) state.ttft_target = v / 1e3; update(); });
    for (const radio of document.querySelectorAll("input[name=kv]")) {
      radio.addEventListener("change", () => { state.kv_dtype_bytes = Number(radio.value); update(); });
    }
    $("rate").addEventListener("change", () => { const v = Number($("rate").value); state.hourly_rate = v > 0 ? v : null; update(); });
    $("rate-reset").addEventListener("click", () => { state.hourly_rate = null; update(); });
  }
  function reflectInputs() {
    $("accelerator").value = state.accelerator;
    const card = data.accelerators[state.accelerator];
    const badge = $("accel-badge");
    badge.textContent = card.measured ? "measured" : "prior";
    badge.className = "badge " + (card.measured ? "measured" : "prior");
    badge.title = card.provenance;
    $("accel-prov").textContent = card.provenance;
    $("v-accelerator").textContent = cardShort(state.accelerator);
    $("model").value = state.model;
    const mdl = data.models[state.model];
    const mbadge = $("model-badge");
    mbadge.textContent = mdl.measured ? "measured" : "predicted";
    mbadge.className = "badge " + (mdl.measured ? "measured" : "prior");
    mbadge.title = mdl.served_by_this_stack
      ? "The model this stack serves and every run measured."
      : "Predicted only: this stack never serves it and no run here has measured it. The arithmetic holds because it is dense, GQA and full-attention.";
    $("model-prov").textContent = `${(mdl.kv_bytes_per_token / 1e3).toFixed(0)} kB of KV per token, ${mdl.num_kv_heads} KV heads × ${mdl.num_layers} layers. ${mdl.provenance}`;
    $("v-model").textContent = mdl.name;
    $("prompt-range").value = promptToSlider(state.prompt_tokens); $("prompt").value = state.prompt_tokens;
    $("v-prompt").textContent = int(state.prompt_tokens);
    $("output-range").value = state.output_tokens; $("output").value = state.output_tokens;
    $("v-output").textContent = int(state.output_tokens);
    $("hit-range").value = Math.round(state.hit_rate * 100); $("hit").value = Math.round(state.hit_rate * 100);
    $("v-hit").textContent = pct(state.hit_rate);
    $("gmu-range").value = Math.round(state.gpu_memory_utilization * 100); $("gmu").value = Math.round(state.gpu_memory_utilization * 100);
    $("ttft").value = Math.round(state.ttft_target * 1e3);
    for (const radio of document.querySelectorAll("input[name=kv]")) radio.checked = Number(radio.value) === state.kv_dtype_bytes;
    const rate = rateOf(state);
    $("rate").value = rate.toFixed(2);
    $("v-rate").textContent = `$${rate.toFixed(2)}`;
    $("rate-prov").textContent = state.hourly_rate === null ? card.hourly_rate_provenance : "Your figure: a contract, not physics.";
  }
  function cell(title, big, small) {
    return `<div class="cell"><div class="cell-title">${title}</div><div class="cell-big">${big}</div><div class="cell-small">${small || ""}</div></div>`;
  }
  function renderDetails(point) {
    let html = "";
    if (point.seats) {
      const s = point.seats;
      html += cell("KV pool", `${(point.kv_pool_tokens / 1e6).toFixed(2)} M tokens`, "the engine's startup log outranks this");
      html += cell("seats by room / by time", `${s.by_capacity} / ${s.by_latency}`, `max_num_seqs ${s.max_num_seqs}, bound by ${s.bound_by}` + (s.gap ? `, gap ${s.gap.toFixed(2)}×` : ""));
    }
    const tf = point.ttft_floor, tu = point.ttft_floor_uncached;
    html += cell("TTFT floor", ms(tf.seconds), `${int(state.prompt_tokens)} tokens, bound by ${tf.bound_by}; ${ms(tu.seconds)} at the ${int(tu.uncached_tokens)} uncached`);
    if (point.tpot_floor_at_max_num_seqs) {
      const d = point.tpot_floor_at_max_num_seqs;
      html += cell(`TPOT floor at ${point.seats.max_num_seqs} seats`, ms(d.seconds), `bound by ${d.bound_by}, ${d.ratio.toFixed(1)}×`);
      html += cell("hardware floor", `${int(point.aggregate_tokens_per_sec)} tok/s`, `$${point.cost_per_1m_output_tokens.toFixed(3)} / 1M output tokens`);
    }
    const sv = point.service;
    if (sv) {
      html += cell(`service at h = ${state.hit_rate.toFixed(2)}`, `${sv.seats_shipped} seats`,
        `${sv.seats_at_hit_rate.toFixed(1)} by prefill interference${sv.bound_by === "capacity" ? ", capped by the pool" : ""}` +
        (sv.seats_shipped ? `; ${int(sv.aggregate_tokens_per_sec)} tok/s, $${sv.cost_per_1m_output_tokens.toFixed(3)} / 1M` : ""));
    }
    $("outputs").innerHTML = html;
    const coef = point.coefficients;
    $("coefficients").innerHTML =
      `<span>eff_mem <b>${coef.eff_mem}</b> · mfu <b>${coef.mfu}</b> <span class="prov ${data.accelerators[state.accelerator].measured ? "measured" : "prior"}">${coef.provenance}</span></span>` +
      (sv ? `<span>interference <b>${(sv.fit.slope_s_per_seat * 1e3).toFixed(3)} ms/seat, ${(sv.fit.intercept_s * 1e3).toFixed(2)} ms</b> <span class="prov measured">${sv.fit.provenance}</span></span>` : "");
  }

  // --- step 3: trouble ------------------------------------------------------------
  function bindFlow() {
    for (const group of document.querySelectorAll(".choices")) {
      for (const b of group.querySelectorAll("button")) {
        b.addEventListener("click", () => { flow[group.dataset.q] = b.dataset.v; renderAdvisor(current); });
      }
    }
    $("changed").addEventListener("change", () => { flow.changed = $("changed").checked; renderAdvisor(current); });
    const form = $("readings");
    let html = "";
    for (const [name, spec] of Object.entries(map.inputs)) {
      if (name === "ttft_target_ms" || name === "tpot_target_ms") continue;
      const kind = spec.on_kind ? "" : `<span class="kindnote">not on kind</span>`;
      if (spec.type === "enum") {
        html += `<label class="reading"><span class="rname">${name}</span><select data-reading="${name}">` +
          spec.values.map((v) => `<option value="${v}">${v}</option>`).join("") + `</select><span class="panel">${spec.panel}</span></label>`;
      } else if (spec.type === "bool") {
        html += `<label class="reading"><span class="rname">${name}</span><input type="checkbox" data-reading="${name}"><span class="panel">${spec.panel}</span></label>`;
      } else {
        html += `<label class="reading"><span class="rname">${name}</span><input type="number" step="any" data-reading="${name}" placeholder="No data">` +
          `<span class="panel">${spec.panel} ${kind}</span></label>`;
      }
    }
    form.innerHTML = html;
    for (const el of form.querySelectorAll("[data-reading]")) {
      const name = el.dataset.reading;
      if (el.type === "checkbox") el.checked = !!readings[name];
      else if (readings[name] !== null) el.value = readings[name];
      el.addEventListener("input", () => {
        if (el.type === "checkbox") readings[name] = el.checked;
        else if (el.tagName === "SELECT") readings[name] = el.value;
        else readings[name] = el.value === "" ? null : Number(el.value);
        renderAdvisor(current);
      });
    }
    document.querySelector(".expert").addEventListener("toggle", () => renderAdvisor(current));
  }
  function reflectFlow() {
    for (const group of document.querySelectorAll(".choices")) {
      for (const b of group.querySelectorAll("button")) b.classList.toggle("active", flow[group.dataset.q] === b.dataset.v);
    }
    $("changed").checked = flow.changed;
  }

  // The flow's answers as the readings the rules evaluate. The values are
  // representative, not measured -- "a little" is 1.2x the floor -- and
  // they are only ever compared against thresholds, never shown as numbers.
  function readingsFromFlow(point) {
    const r = {
      alert: "none", track: "stable", max_num_seqs: 256, replicas: 1, max_replicas: 4, targets_down: 0,
      kv_usage: null, preemptions_per_s: null, median_itl_ms: null,
      changed_recently: flow.changed,
      waiting: flow.queue === "yes" ? 1 : flow.queue === "no" ? 0 : null,
      running: flow.seats === "yes" ? 256 : flow.seats === "no" ? 1 : null,
    };
    const ttftFloor = point.ttft_floor_uncached.seconds * 1e3;
    r.ttft_p99_ms = { far: 2.0, near: 1.2, ok: 0.95 }[flow.ttft] !== undefined
      ? ttftFloor * { far: 2.0, near: 1.2, ok: 0.95 }[flow.ttft] : null;
    // A queue standing on a replica is what QueueBeyondTTFTBudget fires on,
    // so "people are queuing up" is that alert; a late first token with no
    // queue is the burn-rate rule.
    if (r.waiting > 0) r.alert = "QueueBeyondTTFTBudget";
    else if (flow.ttft === "far" || flow.ttft === "near") r.alert = "TTFTBudgetBurning";
    const target = state.tpot_target * 1e3;
    // "near the floor" and "far above" are relative to the decode step at the
    // seats in use; without a seat count, the target itself stands in.
    r.tpot_p99_ms = { physics: target * 1.05, far: target * 1.6, ok: target * 0.9 }[flow.tpot] ?? null;
    r.__tpot_gap_override = { physics: 1.05, far: 1.6 }[flow.tpot] ?? null;
    return r;
  }

  function derived(point, base) {
    const a = data.accelerators[state.accelerator];
    const m = R.withKvDtype(data.models[state.model], state.kv_dtype_bytes);
    const ctx = state.prompt_tokens + state.output_tokens;
    const r = Object.assign({}, base);
    r.ttft_target_ms = state.ttft_target * 1e3;
    r.tpot_target_ms = state.tpot_target * 1e3;
    r.ttft_floor_ms = point.ttft_floor_uncached.seconds * 1e3;
    r.prompt_fits_budget = point.ttft_floor.seconds <= state.ttft_target;
    r.tpot_floor_at_running_ms = base.running > 0 ? R.tpotFloor(m, a, base.running, ctx).seconds * 1e3 : null;
    r.ttft_gap_ratio = base.ttft_p99_ms !== null ? base.ttft_p99_ms / r.ttft_floor_ms : null;
    r.tpot_gap_ratio = base.__tpot_gap_override ?? ((base.tpot_p99_ms !== null && r.tpot_floor_at_running_ms) ? base.tpot_p99_ms / r.tpot_floor_at_running_ms : null);
    r.itl_ratio = (base.tpot_p99_ms !== null && base.median_itl_ms) ? base.tpot_p99_ms / base.median_itl_ms : null;
    delete r.__tpot_gap_override;
    return r;
  }

  function inlineMd(text) {
    const esc = String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return esc.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
              .replace(/`([^`]+)`/g, "<code>$1</code>")
              .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:)]|$)/g, "$1<em>$2</em>");
  }
  function nodeHtml(n, cls) {
    const ev = n.evidence ? `<span class="evidence">${n.evidence}</span>` : "";
    const link = docLink(n.source);
    const src = link ? `<a href="${link}">${n.source}</a>` : `<code>${n.source}</code>`;
    return `<li class="node ${cls}">${ev}${inlineMd(n.text)} <span class="src">${src}</span></li>`;
  }

  function renderAdvisor(point) {
    reflectFlow();
    const expert = document.querySelector(".expert").open;
    const remark = () => markTerms($("advice"));
    const base = expert ? readings : readingsFromFlow(point);
    const r = derived(point, base);
    $("derived").innerHTML =
      `<span>TTFT floor at the uncached tokens <b>${r.ttft_floor_ms.toFixed(1)} ms</b></span>` +
      `<span>prompt fits the TTFT budget: <b>${r.prompt_fits_budget ? "yes" : "no"}</b></span>` +
      (r.tpot_floor_at_running_ms ? `<span>decode step at ${base.running} seats <b>${r.tpot_floor_at_running_ms.toFixed(1)} ms</b></span>` : "") +
      (r.ttft_gap_ratio !== null ? `<span>TTFT p99 / floor <b>${r.ttft_gap_ratio.toFixed(2)}×</b></span>` : "") +
      (r.tpot_gap_ratio !== null ? `<span>TPOT p99 / floor <b>${r.tpot_gap_ratio.toFixed(2)}×</b></span>` : "");
    const rep = window.Advisor.evaluate(map, r);
    let html = "";
    const answered = Object.values(flow).some((v) => v !== "dk" && v !== false);
    if (!rep.active.length) {
      html += `<p class="quiet">${answered || expert
        ? "Nothing is breaching on these answers, or nothing here can tell."
        : "Answer what you can. “Don’t know” is an answer: on <code>kind</code> the stub exports no histogram, so the first-word and per-token questions have no reading there."}</p>`;
    }
    for (const e of rep.active) {
      html += `<section class="symptom"><div class="lab">most likely</div><h3>${e.symptom.title}</h3>`;
      if (e.branch) {
        html += `<ul class="main">${nodeHtml(e.branch, "branch")}</ul>` +
          `<p class="next"><b>Read next:</b> ${e.branch.next_number}</p>` +
          (e.branch.knobs.length ? `<p class="knobs"><b>Knobs:</b> ${e.branch.knobs.map((k) => { const l = docLink(k.source); return l ? `<a href="${l}">${k.name}</a>` : `<code>${k.name}</code>`; }).join(" · ")}</p>` : "");
      } else {
        html += `<p class="quiet">${e.notEvaluable.some((x) => x.node.kind === "branch") ? "Which branch needs a reading you do not have." : "No branch matches these answers."}</p>`;
      }
      const more = e.probes.concat(e.continuations, e.traps);
      if (more.length || e.notEvaluable.length) {
        html += `<details class="more"><summary>Why this, step by step: ${more.length} node${more.length === 1 ? "" : "s"}` +
          (e.notEvaluable.length ? `, ${e.notEvaluable.length} it cannot judge from these answers` : "") + `</summary>`;
        if (e.probes.length) html += `<ul>${e.probes.map((n) => nodeHtml(n, "probe")).join("")}</ul>`;
        if (e.continuations.length) html += `<ul>${e.continuations.map((n) => nodeHtml(n, "cont")).join("")}</ul>`;
        if (e.traps.length) html += `<div class="traps"><div class="branch-head">Traps</div><ul>${e.traps.map((n) => nodeHtml(n, "trap")).join("")}</ul></div>`;
        if (e.notEvaluable.length) html += `<ul class="cannot">` + e.notEvaluable.map((x) => `<li><span class="need">needs ${x.missing.join(", ")}</span> ${inlineMd(x.node.text)}</li>`).join("") + `</ul>`;
        html += `</details>`;
      }
      html += `</section>`;
    }
    if (rep.unknown.length && (answered || expert)) {
      html += `<p class="quiet">Could not check ${rep.unknown.map((u) => `<b>${u.symptom.title.toLowerCase()}</b> (needs ${u.missing.join(", ")})`).join("; ")}: readings the <code>kind</code> stub does not export.</p>`;
    }
    html += `<p class="quiet small">Two symptoms have no dashboard probe and are read in the map itself: <em>Cost per 1M tokens too high</em> and <em>Throughput healthy, users angry</em>.</p>`;
    $("advice").innerHTML = html;
    remark();
  }

  // --- the footer: parity badge, links home ------------------------------------
  function renderSelftest() {
    const b = $("selftest");
    try {
      const rep = window.Selftest.run(data, window.GOLDEN);
      if (rep.failures.length) {
        b.className = "badge bad";
        b.textContent = `parity FAILED: ${rep.failures.length} of ${rep.rows} golden rows disagree with bench/predictions.py`;
        console.error(rep.failures.slice(0, 3));
      } else {
        b.className = "badge ok";
        b.textContent = `parity: ${rep.rows}/${rep.rows} golden rows agree with bench/predictions.py`;
      }
    } catch (e) {
      b.className = "badge bad"; b.textContent = "parity check did not run: " + e.message;
    }
    $("golden-count").textContent = String(window.GOLDEN.length);
  }
  // Every link into the repository reads the same data-repo the advisor's
  // sources use; without it they stay plain text rather than pointing nowhere.
  function renderNav() {
    for (const a of document.querySelectorAll("a[data-doc]")) {
      const href = repoLink(a.dataset.doc);
      if (href) a.href = href;
    }
    for (const a of document.querySelectorAll("a[data-repo-link]")) if (REPO_OK) a.href = REPO;
    if (REPO_OK) {
      const shown = REPO.replace(/^https?:\/\//, "").replace(/\/$/, "");
      $("source-link").innerHTML = `Source, docs and the runs: <a href="${REPO}">${shown}</a>.`;
    }
  }

  // --- the loop ----------------------------------------------------------------
  let current = null;
  function update() {
    writeHash();
    reflectPromise();
    reflectInputs();
    current = R.whatIfPoint(data, state);
    renderSlo(current);
    renderCost(current);
    renderDetails(current);
    renderFigure("fig-seats", window.Draw.seats(data, state, current, { served: true }));
    renderFigure("fig-cost", window.Draw.cost(data, state, current));
    renderAdvisor(current);
    renderCompare(current);
    markTerms(document.querySelector("main"));
  }

  function bindStickySituation() {
    const sit = document.querySelector(".situation");
    const spacer = document.createElement("div");
    spacer.className = "sit-spacer";
    sit.parentNode.insertBefore(spacer, sit);
    let pinned = false;
    const restTop = () => spacer.getBoundingClientRect().top + window.scrollY;
    function check() {
      if (!pinned && window.scrollY > restTop()) {
        spacer.style.height = sit.offsetHeight + "px";   // measured before it leaves the flow
        document.body.classList.add("pinned");
        pinned = true;
      } else if (pinned && window.scrollY <= restTop()) {
        document.body.classList.remove("pinned");
        spacer.style.height = "0px";
        pinned = false;
      }
    }
    function unpin() {
      if (!pinned) return;
      document.body.classList.remove("pinned");
      spacer.style.height = "0px";
      pinned = false;
    }
    window.addEventListener("scroll", check, { passive: true });
    window.addEventListener("resize", () => { unpin(); check(); });
    check();
  }

  readHash();
  bindSteps();
  bindPops();
  bindCopy();
  bindAnswerLinks();
  bindPromise();
  bindInputs();
  bindFlow();
  bindPin();
  bindStickySituation();
  bindTooltips();
  renderSelftest();
  renderNav();
  update();
  if (landing) $(landing).scrollIntoView({ block: "start" });
  window.addEventListener("hashchange", () => { readHash(); update(); });
})();
