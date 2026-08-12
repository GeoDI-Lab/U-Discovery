# GraphRAG equation-proposal artifacts

This directory preserves the prompts and model outputs used to turn retrieved scientific priors into the equation candidates reported in the paper. It is an **output-level reproducibility bundle**: the exact recorded prompt, scientific-prior answer, and candidate JSON files are included, while the copyrighted/source corpus, GraphRAG index, embeddings, and credentials are not.

## Files

- `prompts/query_run_20260114.txt` is the exact prompt embedded in the successful January 14 query record.
- `prompts/query_current_20260115.txt` is the later prompt file found on MSI. Its hash differs from the run prompt, so it must not be presented as the prompt that generated the archived answer.
- `outputs/scientific_prior_output_20260114.md` is the exact STDOUT answer from that query.
- `prompts/propose_no_invention_20260114.txt` and `outputs/no_invention_candidates_20260114.json` preserve the nine-candidate No-Invention proposal.
- `prompts/propose_two_mode_20260119.txt` and `outputs/two_mode_candidates_20260119.json` preserve the later 10 prior-only plus 10 creative proposal.
- `candidate_mapping.json` preserves all 29 generated IDs and maps each selected manuscript row both to its archived UrbanDE-Net method and to the canonical ID in `../artifacts/candidates.json`. Release code evaluates those canonical IDs through `udiscovery.equations.evaluate`.
- `evidence_manifest.json` resolves the answer's opaque source numbers to OpenAlex works without redistributing paper text.
- `config/settings.sanitized.yaml`, `provenance.json`, and `archived_stats.json` record the sanitized configuration, environment, hashes, and audit counts.

## Reuse

Set `GRAPHRAG_API_KEY` in the process environment if adapting the sanitized settings. Do not commit a `.env` file or a provider key. Because the source corpus/index is intentionally absent and hosted LLM calls are not bitwise deterministic, this directory does not claim that rerunning GraphRAG will reproduce identical bytes.

The archived project contains 108 documents, 1,100 text units, 6,994 entity rows, and 10,136 relationship rows. All three inspected graph exports contain 13,334 nodes and 10,136 edges. These counts do not support the manuscript statement of 14,769 nodes and 32,331 edges; see `archived_stats.json` for the explicit comparison.
