# The site

The calculator and the advisor, as one static page for GitHub Pages. No
framework, no build step, no `npm`: a reader needs a browser, and a
contributor needs Python and, for the one check that must run JavaScript, the
`node` that GitHub's runners already carry.

| Path | Written by | What it is |
|---|---|---|
| `index.html`, `style.css` | hand | the page: one sentence for the situation, three steps under it, light theme only |
| `404.html` | hand | what Pages serves for an address that is not the page: one link back to it |
| `fonts/` | Google Fonts, latin subsets | Newsreader, IBM Plex Sans and IBM Plex Mono as woff2 with their OFL licences, so the page looks the same from `file://` with no network |
| `terms.js` | hand | the plain-language explanations shown on hover; every term is registered in `docs/GLOSSARY.md` (a test holds that) |
| `calc.js` | hand | `bench/roofline.py` and `predictions.what_if_point()`, ported in the same order of operations |
| `selftest.js` | hand | runs every row of `data/golden.json` through `calc.js` and compares — the page's badge and CI's step |
| `draw.js` | hand | the two SVG pictures |
| `advisor.js` | hand | evaluates `docs/symptom-map.json` as rules over the readings |
| `ui.js` | hand | the sentence's popovers and the sliders ↔ state ↔ URL hash ↔ render; the answers written in plain words, the glossary term small beside them; a card picked by name, its coefficients chosen under *More assumptions* |
| `data/model.{json,js}` | `python3 bench/export_site_data.py` | every constant: cards with provenance, each card's runs and coefficients in use from the run journal, the models and which one a run measured, fits, rates, presets, measured points |
| `data/golden.{json,js}` | the same | the parity grid |
| `data/symptom-map.js` | the same | a `window.SYMPTOM_MAP = …` twin of `docs/symptom-map.json` |

The `.js` twins exist because `fetch()` fails on a `file://` URL: a reader who
double-clicks `index.html` gets the same page as one on Pages.

## Run it

    python3 bench/export_site_data.py        # after any change under bench/
    node site/selftest.js                    # "golden: N/N rows agree ...", exit 0
    python3 -m http.server -d site 8000      # then open http://localhost:8000/

`open site/index.html` works too. The parity badge in the footer is the same
check the second command runs.

## What holds it together

Three guards, the same three the front-page chart has: `bench/tests/test_export_site_data.py`
asserts the committed `data/` is what the export produces; `.github/workflows/bench.yml`
regenerates it on a clean checkout, diffs, and runs `selftest.js`; and
`.github/workflows/pages.yml` refuses to deploy a page whose JavaScript
disagrees with Python. The repository URL every link into the tree is built from
is `data-repo` on `<body>` (`index.html`).

## The picture of it in the README

`docs/page.gif` is the front page's first image and the route to Pages. It is a
recording, not a generated file: nothing in CI holds it equal to the page, so
re-record it whenever the first screen changes, and carry the date in its alt
text.

    python3 -m http.server -d site 8000 &
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      --headless --disable-gpu --hide-scrollbars --remote-debugging-port=9222 \
      --window-size=1280,900 http://127.0.0.1:8000/ &
    node site/tools/record-page.mjs      # the tour, as PNGs in /tmp/page-frames
    python3 site/tools/make-gif.py       # docs/page.gif

`record-page.mjs` needs Node 22 for its built-in WebSocket and nothing else. It
drives the page over the DevTools protocol, so every frame is the page answering
a real click. `make-gif.py` needs Pillow — the one dependency in the tree, a
contributor's tool only: `bench/`, `site/` and CI still install nothing.

The storyboard inside `record-page.mjs` is what a reader is meant to try, in the
order the page answers it. Scrolling is what the file weighs: 16 of its 95
frames move the page and carry almost all of the megabyte, while a repeated
frame is free. A longer pause is cheap; one more scroll is not.
