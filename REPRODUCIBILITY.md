# Reproducing the SolarFL experiments

This guide recreates the seven client datasets from the immutable public input,
runs the frozen experiment protocol, and compares regenerated tables with the
archived results. The notebooks are supporting analysis records; the Python
modules under `src/solarfl/` are the authoritative experiment implementation.

## Scope

The reproducibility target is the set of ten CSV files listed in
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
uv run python scripts/prepare_data.py --download
uv run python -c "from solarfl.data.splits import verify; verify()"
```

The preparation command downloads only
`hourly_pv_weather_station.csv` from Mendeley Data V2, verifies its official
SHA-256 checksum, and partitions it by station. See `data/README.md` for the
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

## 4. Regenerate the experiment tables

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

## 5. Compare with the archive

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

## Reproducibility boundaries

- The source observations are published separately under CC BY 4.0 and are not
  relicensed by this repository.
- The current federated simulation exchanges unprotected model updates and
  aggregate scaling statistics. It is not a differential-privacy or secure
  aggregation implementation.
- ERA5 values at the prediction timestamp appear only in the explicitly named
  perfect-weather validation variant. The primary test comparison uses the
  deployable past-weather feature set.
