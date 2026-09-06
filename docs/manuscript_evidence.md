# Manuscript evidence and reproduction

The curated [`results/manuscript_evidence/`](../results/manuscript_evidence/)
directory contains six CSV tables, capacity source provenance and two
author-supplied figures:

- `capacity_device_audit.csv` and `station_capacity.csv`: device membership and
  station capacity reconstruction.
- `experiment_scope.csv`: models, feature variants, splits and seeds represented
  in the archived experiments.
- `feature_dimensions.csv`: retained input columns and trainable parameter counts.
- `split_counts.csv`: raw and eligible observations by station, split and variant.
- `optimization_protocol.csv`: optimizer, batching and stopping settings.
- `capacity_provenance.json`: versioned source file IDs, sizes and SHA-256 hashes,
  membership rule and agreement with the frozen station labels.
- `overall_pipeline_2x.png` and `federated_architecture_2x.png`: author-supplied
  manuscript illustrations; the audit does not regenerate these exact images.

## Reproduce the full audit

```bash
uv run python scripts/prepare_data.py --download --audit-capacity
uv run python -m solarfl.eval.manuscript_assets --download
uv run python -m unittest discover -v
uv run ruff check src scripts tests
```

Both audit commands default to `reproduced-results/manuscript_evidence_audit/`,
keeping regenerated files separate from the archived results. Explicit
`--output-dir` options remain available. The manuscript audit writes only the
six CSVs listed above, `capacity_provenance.json`, `asset_provenance.json` and
`SHA256SUMS`. It generates no images, PDFs, SVGs, Markdown or LaTeX exports,
and no additional summary tables. The standalone capacity audit writes only
the two capacity CSVs and `capacity_provenance.json`.

The full audit contains input hashes in `asset_provenance.json` and output
hashes in `SHA256SUMS`; neither replaces `results/SHA256SUMS`. To refresh the
upload packet after an audit, copy only its six listed CSVs and
`capacity_provenance.json`. Keep the current author-supplied figures. The
curated directory's own `SHA256SUMS`
covers its six CSVs, provenance JSON and two figures; refresh it whenever those
files change.

The source [solar semantics](solar_semantics.md), [source manifest](source_manifest.json)
and [algorithm fragment](federated_algorithm.tex) remain in the repository for
reproducibility; their necessary content is represented in the upload packet.
