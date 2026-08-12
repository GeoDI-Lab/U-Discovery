# Paper-to-artifact audit

This file records inconsistencies discovered while selecting the exact release artifacts. They are retained transparently rather than silently changing the paper results.

## GraphRAG prompt and graph statistics

- The exact prompt embedded in the January 14 query run differs from the prompt file modified on January 15. Both are included and identified by SHA-256 in `graphrag/provenance.json`.
- The archived index contains 108 documents, 1,100 text units, 6,994 entity rows, and 10,136 relationship rows.
- All three inspected graph exports contain 13,334 nodes and 10,136 edges. The manuscript states 14,769 nodes and 32,331 edges. No inspected archive supports the manuscript counts.

## Synthetic experiment

- Table 2 is frozen unchanged at the user's request. The exact archived state and distance arrays, metrics, reconstructions, and selected parameter-convergence runs are included.
- The historical fitting CSV contains two distance-weighted rows. The manuscript used the row labeled `Distance-weighted diffusion2` (`MSE_delta=9.7443`, `MSE=4.3209`, complexity `10`); the other row is not reported. The release keeps the 13-row legacy export for audit and provides the exact 12-row manuscript table separately.
- The unchanged Figure 4 panel was generated from a second archived realization (`data_0213`) that has the same initial state and distance matrix but a different stochastic trajectory from the Table 2 training bundle. It is isolated under `figure_sources/figure4`; its closure-residual standard deviation (`1.6812`) agrees with the declared noise scale (`1.6787`).
- Complexity bookkeeping differs between proposal equations and fitting tables for several baselines. The canonical registry records complexity by paper table rather than forcing one value.
- The historical training archive did not contain a complete ten-seed sweep for every Table 2 candidate. The release makes no new aggregate seed claim and does not rerun the experiment.

## Hennepin 2.5 km case

- The exact training distance matrix is asymmetric because column-wise min–max normalization was applied to an originally symmetric metric-distance matrix. Maximum absolute asymmetry is `0.4780116`. The exact fitted input is preserved.
- Figure 7 used the `Complexity` values from its localized result CSVs. Five of those differ from Table 4: SC POW `21→17`, SC EXP `25→21`, Classic EXP `22→24`, Distance diffusion `10→11`, and Classic POW `20→22` (Figure source → Table 4). Candidate selection and top-20 order are unchanged, but frontier x-coordinates depend on which values are used.
- The top paper narrative identifies Betweenness-augmented gravity after excluding the most complex equations; the unfiltered lowest nMSE-delta candidate is Explore–return diffusion.

## Scope decision

All 1.5 km and 2.0 km inputs, fits, scripts, geometries, and Figure 8 products are intentionally excluded. Only the 2.5 km empirical case is released.
