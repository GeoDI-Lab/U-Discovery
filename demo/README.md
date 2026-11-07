# U-Discovery interactive demo

Live demo: **https://geodi-lab.github.io/U-Discovery/**

This directory contains the ready-to-publish static HTML, CSS, JavaScript,
fonts, images, and prepared scientific data for the interactive research demo.
It preserves the GeoDI-branded version published at
https://zhongfuma-gis.com/u-discovery-geodi/ on 6 November 2026.
No build step, Node.js/Python application server, API keys, live LLM requests,
or model fitting service is required to publish or use it.

## Deployment

The `Deploy interactive demo` GitHub Actions workflow uploads only this
directory and publishes it at the repository's GitHub Pages URL. Changes to
`demo/` on `main` redeploy automatically; it can also be run manually from
the Actions tab. The repository's Pages build source is **GitHub Actions**.
The Python research workflows and their documentation are separate from the
published site.

Keep `index.html`, `assets/`, `images/`, and `data/` together. The compiled
deployment prefix is `/U-Discovery/`. Navigation uses URL hashes, so direct
links such as `https://geodi-lab.github.io/U-Discovery/#results` work on
static hosting. Opening `index.html` directly with a `file://` URL is not
supported because the browser loads local data with `fetch()`.

## What runs in the browser

All five stages, including the map playback, knowledge graph, archived
prompts and proposals, and fitted-equation comparisons, use the bundled
data. The demo examines recorded research results; it does not perform new
LLM inference or neural fitting. Optional OpenStreetMap background tiles
and external source links require internet access. Fonts and research data
are bundled locally.

## Provenance and updates

`snapshot-manifest.json` records each captured file's original URL, original
SHA-256, deployed SHA-256, and size. Only deployment prefixes in HTML,
JavaScript, and CSS were changed for this deployment. Scientific data,
images, and other assets retain their published bytes.

This is a compiled release snapshot, not the React/TypeScript authoring
source. To update the interface, replace this directory's static release
with a new production build using `/U-Discovery/` as its deployment base,
retain the required assets and notices, update the provenance manifest, and
push to `main`. No access to the original website is required at runtime.

Software and data follow the repository's `LICENSE`, `DATA_LICENSE.md`, and
`LICENSING.md`; third-party components retain their own licenses. GeoDI Lab
branding is preserved from the original demo. See `THIRD_PARTY_NOTICES.md`
and `licenses/` for bundled dependency notices.
