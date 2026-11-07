#!/usr/bin/env python3
"""Pack the finished page images into one self-contained HTML file."""
import base64
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from strata_bins import output_root, probe_name  # noqa: E402

WORK = output_root()
PAGE = WORK / "figures" / "page"


def uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


layers = [uri(PAGE / f"layer_{layer:02d}.png") for layer in range(48)]
template = r'''<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column of figure plates, led by a layer stepper.
   Single light theme by choice: the figures are light PNGs with their own surface. */
:root {
  color-scheme: light;
  --page: #f5f6f8;
  --plate: #fcfcfb;
  --ink: #15181d;
  --ink-soft: #4d535c;
  --rule: #d9dde3;
  --accent: #123f75;
  --accent-wash: #dce9f9;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
body { background: var(--page); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.55; }
main { max-width: 1100px; margin: 0 auto; padding-block: 40px 72px;
       padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
       display: grid; gap: 40px; }
header, section { display: grid; gap: 14px; min-width: 0; }
h1 { font-size: clamp(1.8rem, 4.5vw, 2.5rem); line-height: 1.15; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 1.3rem; line-height: 1.3; font-weight: 600; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 68ch; }
.lede { color: var(--ink-soft); font-size: 1.05rem; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 70ch; }
.check { font-family: var(--mono); font-size: .8rem; color: var(--ink-soft); overflow-wrap: anywhere; }
figure { margin: 0; background: var(--plate); border: 1px solid var(--rule); min-width: 0; display: grid; }
figure img { display: block; width: 100%; height: auto; }
figcaption { padding: 12px 18px 14px; border-top: 1px solid var(--rule); font-size: .95rem; color: var(--ink-soft); }
.stepper { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; background: var(--plate);
           border: 1px solid var(--rule); padding: 12px 16px; }
.stepper label { font-family: var(--mono); font-size: .95rem; font-weight: 500; min-width: 7.5ch; font-variant-numeric: tabular-nums; }
.stepper input { flex: 1 1 220px; min-width: 0; accent-color: var(--accent); }
.stepper button { font: inherit; font-size: .9rem; color: var(--accent); background: var(--accent-wash); border: 0;
                  padding: 6px 12px; cursor: pointer; }
.stepper button:focus-visible, .stepper input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>

<main>
  <header>
    <h1>__HEADING__</h1>
    __INTRO__
  </header>

  <section>
    <h2>Step through the layers</h2>
    <div class="stepper">
      <label id="layerLabel" for="layerRange">Layer 47</label>
      <input id="layerRange" type="range" min="0" max="47" step="1" value="47">
      <button id="previousLayer" type="button">Previous</button>
      <button id="nextLayer" type="button">Next</button>
    </div>
    <figure>
      <img id="layerImage" src="__FIRST__" alt="Heatmap of mean signed error and standard deviation by sequence separation and final structure distance for layer 47.">
      <figcaption>Each cell shows mean signed error with its SD below. Grey cells have under 100 pairs or under 5 proteins, and their values are still printed. A dash means the bin has no pairs.</figcaption>
    </figure>
  </section>

  <section>
    <h2>What stands out</h2>
    __FINDINGS__
  </section>

  <section>
    <h2>All layers at once</h2>
    <figure>
      <img src="__MEAN__" alt="__MEAN_ALT__">
      <figcaption>Mean signed error. Colour is linear within ±2 Å and logarithmic beyond, so small biases and large ones both show.</figcaption>
    </figure>
    <figure>
      <img src="__SD__" alt="Forty-eight small heatmaps of the standard deviation of signed error, one per layer.">
      <figcaption>Standard deviation of signed error, on a linear scale shared by all panels. The scale is capped at 9 Å.</figcaption>
    </figure>__EXTRA_FIGURES__
  </section>
</main>

<script>
const layerImages = __LAYERS__;
const range = document.getElementById("layerRange");
const label = document.getElementById("layerLabel");
const image = document.getElementById("layerImage");
function showLayer(value) {
  const layer = Math.min(47, Math.max(0, value));
  range.value = String(layer);
  label.textContent = "Layer " + layer;
  image.src = layerImages[layer];
  image.alt = "Heatmap of mean signed error and standard deviation by sequence separation and final structure distance for layer " + layer + ".";
}
range.addEventListener("input", () => showLayer(Number(range.value)));
document.getElementById("previousLayer").addEventListener("click", () => showLayer(Number(range.value) - 1));
document.getElementById("nextLayer").addEventListener("click", () => showLayer(Number(range.value) + 1));
</script>
'''
page = template
for key, value in json.loads((SCRIPT_DIR / f"page_text_{probe_name()}.json").read_text()).items():
    page = page.replace(f"__{key}__", value)
if "__COMPARISON__" in page:
    page = page.replace("__COMPARISON__", uri(PAGE / "mae_difference_vs_ols.png"))
page = (page.replace("__FIRST__", layers[47]).replace("__MEAN__", uri(PAGE / "mean_signed_error.png"))
        .replace("__SD__", uri(PAGE / "sd_signed_error.png")).replace("__LAYERS__", json.dumps(layers)))
target = WORK / "native_error_strata_page.html"
target.write_text(page)
print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
