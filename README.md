# SolarMagix — Federated PV Forecasting

Research project evaluating whether federated learning can approach centralized
training for 24-hour-ahead photovoltaic capacity-factor forecasting without
pooling raw data from seven solar stations in North Macedonia.

## Experiment

- **Target:** capacity factor at hour T.
- **Clients:** one PV station per client (seven total).
- **Splits:** frozen per-client chronological 70/15/15 train/validation/test.
- **Operational inputs:** T−24 production and weather plus deterministic solar
  geometry.
- **Perfect-weather variant:** additionally uses ERA5 reanalysis at T as an
  optimistic upper bound, not a deployable day-ahead forecast.
- **Comparisons:** persistence, local and centralized Ridge/MLP, FedAvg, and
  FedProx.
- **Metrics:** daylight-only MAE, RMSE, and MAE skill against same-hour-yesterday
  persistence.

All published result tables currently use the validation split. The test split
is reserved for final evaluation.

## Repository layout

- `configs/` — hand-maintained feature specification, station labels, and frozen
  split manifest.
- `src/solarfl/` — reusable feature, model, federated-training, and evaluation
  code.
- `notebooks/` — numbered historical exploration; never imported by modules.
- `docs/framing.md` — methodological decision log and current protocol.
- `results/` — validation tables and figures.
- `data/` — local, gitignored source and processed data.

## Setup and checks

```bash
uv sync
uv run python -c "import solarfl"
uv run ruff check src main.py
```

The split module's default entry point would regenerate the frozen manifest, so
do not run it during routine verification. To verify the existing manifest
without rewriting it, use:

```bash
uv run python -c "from solarfl.data.splits import verify; verify()"
```

## Run experiments

```bash
uv run python -m solarfl.models.baselines
uv run python -m solarfl.federated.fedavg
uv run python -m solarfl.federated.fedprox
```

Read `CLAUDE.md` before changing the pipeline: it records invariants that keep
the frozen splits, leakage controls, and comparisons valid.
