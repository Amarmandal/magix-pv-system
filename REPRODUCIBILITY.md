# Reproducing the SolarFL experiments

This guide recreates the seven client datasets from the immutable public input,
runs the frozen experiment protocol, and compares regenerated tables with the
archived results. The notebooks are supporting analysis records; the Python
modules under `src/solarfl/` are the authoritative experiment implementation.

## Scope

The reproducibility target is the set of twelve CSV files listed in
`results/SHA256SUMS`. PNG and PDF figures are derived from those tables by
`notebooks/07_results-figures.ipynb` and
`notebooks/08_test_results_figures.ipynb`.

The primary paper result is defined by D-029 through D-031 in
`docs/framing.md`: centralized MLP versus FedProx with one local epoch,
`mu = 1`, training seeds 0 through 4, and a 10,000-repetition paired
complete-day bootstrap.

## Requirements

- Git
- [`uv`](https://docs.astral.sh/uv/)
- Internet access for the first dependency synchronization and data download
- Enough CPU time for the neural-network and federated sweeps

The repository pins Python in `.python-version` and all Python packages in
`uv.lock`. Training is CPU-based; no CUDA device is selected by the code.
Floating-point results can vary slightly between operating systems and CPU
libraries, so verification uses an absolute and relative tolerance of `1e-6`.

## 1. Create the environment

```bash
git clone https://github.com/Amarmandal/magix-internship.git
cd magix-internship
uv sync --locked
uv run python -c "import solarfl"
```

## 2. Acquire and verify the data

```bash
uv run python scripts/prepare_data.py --download --audit-capacity
uv run python -c "from solarfl.data.splits import verify; verify()"
```

The preparation command downloads `hourly_pv_weather_station.csv` from
Mendeley Data V2, verifies its official SHA-256 checksum, and partitions it by
station. With `--audit-capacity`, it also acquires/verifies `devices.csv`,
`stations.csv`, and `hourly_pv_weather_inverter.csv`, independently reconstructs
the retrospective capacity denominators, and checks the frozen station labels. See `data/README.md` for the
DOI, license, exact file identifier, and the documented landing-page row-count
discrepancy.

Do not run `python -m solarfl.data.splits`: that maintenance entry point would
regenerate the frozen split manifest.

## 3. Run data-independent checks

```bash
uv run ruff check src scripts tests
uv run python -m unittest discover -v
uv run python -m solarfl.eval.test_evaluation --smoke
```

## 4. Audit the Variant-A row mask

```bash
uv run python scripts/audit_variant_row_masks.py \
  --output reproduced-results/variant_row_mask_audit.csv
```

The `past` row set requires only Variant-A predictors, the target, and explicit
target-hour daylight. The `common` row set additionally requires target-hour
reanalysis so exploratory past/perfect-weather comparisons remain paired. The
audit reports whether that additional availability requirement excludes any
otherwise eligible Variant-A timestamps.

## 5. Generate the margin-sensitivity table

```bash
uv run python -m solarfl.eval.margin_sensitivity \
  --summary results/test_evaluation_summary.csv \
  --output reproduced-results/noninferiority_margin_sensitivity.csv
```

This post-hoc appendix analysis reuses the frozen confidence-interval upper
bound and checks the decision at margins from 0 to 0.010. It does not retrain
models, rerun the bootstrap, or replace the pre-specified primary margin of
`0.005`.

## 6. Regenerate the experiment tables

Write regenerated outputs to a new directory. This preserves the archived
tables and the one-shot test guard.

```bash
uv run python -m solarfl.models.baselines --output-dir reproduced-results
uv run python -m solarfl.federated.fedavg --output-dir reproduced-results
uv run python -m solarfl.federated.fedprox --output-dir reproduced-results
uv run python -m solarfl.eval.seed_robustness --output-dir reproduced-results
uv run python -m solarfl.eval.test_evaluation --output-dir reproduced-results
```

The FedProx sweep and five-seed robustness/test evaluations are the expensive
steps. Record the machine description and elapsed time when reporting an
independent reproduction, for example with `/usr/bin/time -p` before each
command.

## 7. Compare with the archive

```bash
uv run python scripts/verify_results.py --candidate reproduced-results
```

The verifier first confirms that the committed reference tables still match
`results/SHA256SUMS`. It then compares table schemas, row order, labels, and
numeric values. A mismatch is evidence to investigate, not a reason to change
the frozen protocol or reference results.

## Expected primary result

The frozen test summary reports:

- centralized mean macro-MAE: `0.133405`
- FedProx mean macro-MAE: `0.128397`
- FedProx-minus-centralized gap: `-0.005008`
- paired 95% bootstrap interval: `[-0.009926, -0.000537]`
- non-inferiority margin: `0.005`
- decision: FedProx is non-inferior under the pre-specified rule

The post-hoc appendix sensitivity table reports the same decision for all
examined non-negative margins from `0` to `0.010`; `0.005` remains the sole
pre-specified primary margin.

## Manuscript evidence audit and assets

```bash
uv run python -m solarfl.eval.manuscript_assets --download \
  --output-dir reproduced-results/manuscript_evidence_audit
```

This descriptive command generates capacity/device audits, scope and
optimization tables, input-width and row-count checks, and reusable figures.
It does not train models or open the test for additional model selection.
See [the manuscript evidence map](docs/manuscript_evidence.md) for the output inventory,
source provenance and manuscript qualifications. These supplemental outputs
have their own checksum manifest; the twelve primary reference CSVs stay frozen.

## Reproducibility boundaries

- The source observations are published separately under CC BY 4.0 and are not
  relicensed by this repository.
- The current federated simulation exchanges unprotected model updates and
  aggregate scaling statistics. It is not a differential-privacy or secure
  aggregation implementation. The host holds all public-data arrays and pools
  validation predictors and targets for early stopping.
- ERA5 values at the prediction timestamp appear only in the explicitly named
  perfect-weather validation variant. The primary test comparison uses the
  past-only feature set at a nominal 24-hour timestamp horizon. The source
  uses interval-start labels; complete lagged aggregates are available only
  after their interval ends. Real-time dispatch and reanalysis latency are not
  simulated; see `docs/solar_semantics.md`.
