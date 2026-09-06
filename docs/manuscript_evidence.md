# Manuscript evidence and reproduction

The curated [`results/manuscript_evidence/`](../results/manuscript_evidence/)
directory contains six CSV tables, capacity source provenance and two current
author-supplied figures. It contains no LaTeX fragments, duplicate exports, or
separately generated evidence-notes file. The manuscript and Prism prompt have
not been edited in this cleanup.

## Reproduce the full audit

```bash
uv run python scripts/prepare_data.py --download --audit-capacity
uv run python -m solarfl.eval.manuscript_assets --download
uv run python -m unittest discover -v
uv run ruff check src scripts tests
```

Both audit commands now default to `reproduced-results/manuscript_evidence_audit/`,
keeping full machine-generated tables, LaTeX and diagnostic figures outside
the curated upload folder. Explicit `--output-dir` options remain available.
The pre-cleanup 50-file packet is preserved locally in
`reproduced-results/manuscript_evidence_full_archive/` (ignored by Git).

The full audit contains input hashes in `asset_provenance.json` and output
hashes in `SHA256SUMS`; neither replaces `results/SHA256SUMS`. To refresh the
upload packet after an audit, copy only its six listed CSVs and
`capacity_provenance.json`. Keep the current author-supplied figures rather
than generated historical diagrams.

The source [solar semantics](solar_semantics.md), [source manifest](source_manifest.json)
and [algorithm fragment](federated_algorithm.tex) remain in the repository for
reproducibility; their necessary content is represented in the upload packet.
