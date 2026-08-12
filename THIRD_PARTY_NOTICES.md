# Third-party review notes

This release depends on NumPy, pandas, PyTorch, Matplotlib, NetworkX, and
scikit-learn. Their licenses are not replaced by the MIT or CC BY 4.0 licenses
for this repository.

Parts of `src/udiscovery/model.py` are adapted from Graph WaveNet by Zonghan Wu
et al.:

- Source: <https://github.com/nnzhan/Graph-WaveNet>
- Upstream revision reviewed: `6b162e80c59a1d494809252eca055cff93dc66b1`
- Upstream copyright: `Copyright (c) <2019> <Zonghan Wu>`
- Affected implementation: neighborhood convolution, graph convolution, and
  the dilated gated Graph WaveNet encoder
- License: MIT; the preserved upstream notice is in
  `THIRD_PARTY_LICENSES/Graph-WaveNet-LICENSE`

The U-Discovery implementation adds distance-aware supports, deterministic
initialization, node- and graph-level embeddings, and UrbanDE-Net parameter
heads. These adaptations remain subject to the preserved upstream notice as
well as this repository's MIT license.

The GraphRAG configuration describes Microsoft GraphRAG and hosted OpenAI model
interfaces. No GraphRAG source code, model weights, source-paper corpus, or
provider credentials are redistributed here.

Quoted literature excerpts and other attributed third-party material in the
GraphRAG records remain under their original rights and are not relicensed.

This notice does not replace the license texts or constitute legal advice. See
`LICENSING.md`.
