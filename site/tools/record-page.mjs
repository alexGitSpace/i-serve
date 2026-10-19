/* record-page.mjs -- the frames behind docs/page.gif.
 *
 * Drives a headless Chrome over the DevTools protocol, so every frame is the
 * page answering a real click: no state is faked and no number is drawn here.
 * Node 22+ only, for the built-in WebSocket; nothing is installed.
 * The storyboard below is the tour, and site/README.md says how to run it.
 */

const PORT = process.env.PORT || 9222;
const OUT = process.env.OUT || "/tmp/page-frames";

import { writeFileSync, mkdirSync, rmSync } from "fs";

async function connect() {
  let targets;
  for (let i = 0; i < 40; i++) {
    try {
      targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      if (targets.length) break;
    } catch { /* Chrome is still starting */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  const page = targets.find((t) => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  });
  const send = (method, params = {}) => new Promise((res) => {
    const mid = ++id;
    pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  return { send, close: () => ws.close() };
}

const c = await connect();
await c.send("Page.enable");
await c.send("Runtime.enable");

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

let n = 0;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const evaluate = async (expr) =>
  (await c.send("Runtime.evaluate", { expression: expr, returnByValue: true })).result?.result?.value;

async function shot(times) {
  for (let i = 0; i < times; i++) {
    const r = await c.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(`${OUT}/f${String(n++).padStart(3, "0")}.png`, Buffer.from(r.result.data, "base64"));
  }
}

// hold repeats the frame: identical frames are free in a GIF, motion is not.
const act = async (expr, hold = 1, settle = 200) => { await evaluate(expr); await wait(settle); await shot(hold); };
const card = (v) => `(() => { const e = document.getElementById('card'); e.value='${v}'; e.dispatchEvent(new Event('change',{bubbles:true})); })()`;
const slide = (id, v) => `(() => { const e = document.getElementById('${id}'); e.value=${v}; e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); })()`;
const to = (y) => `window.scrollTo(0, ${y})`;

// 1. the first screen: the card and the runs behind it, then the cache
await act(to(0), 4);
await act(card("mi300x"), 3);
await act(`document.getElementById('v-accelerator').click()`, 9, 320);
await act(`document.body.click()`, 1);
await act(card("l40s"), 4);
for (const v of [20, 40, 60]) await act(slide("hit-range", v), 1);
await act(slide("hit-range", 80), 5);
await act(slide("hit-range", 30), 4);

// 2. the sentence collapsing into the pinned bar
for (const y of [120, 260, 400]) await act(to(y), 2, 240);
await act(to(560), 4);

// 3. the seats chart, with the knobs that move it still on screen
await act(to(610), 1);
await act(to(750), 4);
for (const v of [40, 30]) await act(slide("tpot-range", v), 2);
await act(slide("tpot-range", 20), 5);
await act(slide("tpot-range", 50), 4);
await act(card("mi300x"), 5);
await act(card("l40s"), 4);

// 4. cost: a stricter promise, priced
for (const y of [1030, 1310, 1580]) await act(to(y), 1);
await act(to(1580), 3);
await act(slide("tpot-range", 20), 5);
await act(slide("tpot-range", 50), 4);

// 5. trouble: one symptom answered, and the advice it produces
for (const y of [1840, 2140, 2390]) await act(to(y), 1);
await act(to(2390), 5);
await act(`document.querySelector('[data-q="queue"] button[data-v="yes"]').click()`, 10, 320);

console.log(`${n} frames in ${OUT}`);
c.close();
