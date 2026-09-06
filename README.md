# SolarFL — federated PV forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible research code for 24-hour-ahead photovoltaic capacity-factor
forecasting across seven solar stations in North Macedonia. Exploratory validation compares
same-hour-yesterday persistence, local and centralized Ridge/MLP models,
FedAvg, and FedProx. The frozen confirmatory test compares only centralized
MLP with validation-selected FedProx E1 (mu=1), using past-only inputs and
paired training seeds 0–4.

The associated manuscript is:

> Amar Kumar Mandal, Yuvraj Chaudhary, and Eftim Zdravevski. “Localized versus
> Pooled Station Data in Time-Series Forecasting of PV Power Generation: A Case
> Study in North Macedonia.”

## Main result

The primary question is whether federated learning can match centralized
training without pooling raw station observations. “Comparable” was frozen as
a non-inferiority test with a margin of **0.005 absolute macro-MAE**.

Across training seeds 0–4, centralized MLP obtained mean macro-MAE `0.133405`
and FedProx obtained `0.128397`. The observed FedProx-minus-centralized gap was
`-0.005008`, with a paired complete-day bootstrap 95% interval of
`[-0.009926, -0.000537]`. Its upper endpoint is below the pre-specified margin,
so FedProx with one local epoch and `mu = 1` is non-inferior under the frozen
rule.

Federated training here does **not** provide differential privacy, secure
aggregation, or protected parameter exchange. Client training updates use separate station arrays and exchange unprotected
model weights and aggregate feature statistics. This is an offline simulation
on public data: the host can access all arrays, and validation arrays are
pooled for early stopping. Station clients represent possible data silos, not
verified ownership or privacy boundaries.

## Reproduce the study

Install the exact locked environment, download and partition the immutable
public input, and run the fast checks:

```bash
uv sync --locked
uv run python scripts/prepare_data.py --download --audit-capacity
uv run python -c "from solarfl.data.splits import verify; verify()"
uv run python -m unittest discover -v
uv run python -m solarfl.eval.test_evaluation --smoke
```

Then follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full experiment
sequence, isolated output directory, result comparison, expected values, and
known reproducibility boundaries.

The source observations come from Mendeley Data V2, DOI
[`10.17632/4zgsckxpdy.2`](https://doi.org/10.17632/4zgsckxpdy.2), under CC BY
4.0. See [data/README.md](data/README.md) for the exact file ID, checksum, data
license, and preparation procedure.

## Experiment design

- **Target:** capacity factor at hour T.
- **Forecast horizon:** 24 hours.
- **Clients:** one PV station per client, seven total.
- **Splits:** frozen per-client chronological 70/15/15 train/validation/test.
- **Past-only inputs (Variant A):** production and weather at stored label
  T−24 plus deterministic solar geometry. See [timestamp semantics](docs/solar_semantics.md)
  for the interval-availability qualification of the nominal 24-hour horizon.
- **Perfect-weather validation variant:** additionally uses ERA5 reanalysis at
  T as an optimistic upper bound, not a deployable forecast.
- **Metrics:** daylight-only MAE, RMSE, and MAE skill against persistence.
- **Primary comparison:** centralized MLP versus full-participation FedProx,
  one local epoch per round, `mu = 1`, seeds 0–4.
- **Secondary comparison:** five local epochs studies the
  communication–computation trade-off and is not compute-matched to centralized
  training.

Model and hyperparameter choices used validation only. The test split was
opened after the D-029–D-031 protocol freeze; the final outputs are archived in
`results/test_evaluation_*.csv`.

See the [methodology overview](results/manuscript_evidence/overall_pipeline_2x.png) for the data,
feature, training, and evaluation flow.

## Manuscript evidence

[Manuscript evidence](docs/manuscript_evidence.md) maps code changes to supporting
capacity, architecture, split-count and scope assets. Regenerate them without
training or new test predictions:

```bash
uv run python -m solarfl.eval.manuscript_assets --download
```

## Repository layout

- `configs/` — feature specification, station labels, and frozen split manifest.
- `src/solarfl/` — reusable feature, model, federated-training, and evaluation code.
- `scripts/` — data preparation and result-verification commands.
- `tests/` — data-independent release and reproducibility checks.
- `notebooks/` — historical analysis and publication-figure generation.
- `docs/framing.md` — methodological decision log and frozen protocol.
- `results/` — archived validation/test tables, checksums, and figures.
- `data/README.md` — external data provenance and acquisition instructions.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff). After the first Zenodo
release, cite the version-specific Zenodo DOI in the manuscript; a DOI badge can
then be added here without changing the archived computation.

The software is released under the [MIT License](LICENSE). The input dataset
retains its separate CC BY 4.0 license.
