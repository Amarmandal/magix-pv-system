# SolarMagix — Federated PV Forecasting

Research project evaluating federated learning for 24-hour-ahead photovoltaic
capacity-factor forecasting across seven solar stations in North Macedonia.

## Research questions

**RQ1 — Primary:** Can federated learning achieve forecasting performance
comparable to centralized training without pooling raw station observations?

Here, "comparable" is a non-inferiority question: the final protocol must define
the largest practically acceptable federated-versus-centralized performance gap
before the test split is evaluated. The current work focuses on RQ1 using
federated learning without a formal differential-privacy guarantee.

**RQ2 — Additional direction:** How does adding differential privacy to
federated training affect the privacy–utility trade-off under explicitly stated
privacy budgets?

Federated training avoids central collection of raw station observations, but
the current implementation exchanges unprotected model updates and aggregate
feature statistics. It therefore does not by itself guarantee privacy or secure
parameter exchange.

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

### Primary and secondary federated comparisons

- **Primary RQ1 comparison:** centralized MLP versus FedAvg/FedProx with one
  local epoch per communication round (`E = 1`). One full-participation E=1
  round processes every client's training data once, making its data exposure
  approximately comparable to one centralized epoch.
- **Secondary experiment:** `E = 5` studies the communication–computation
  trade-off created by doing more local training before aggregation. These runs
  are not compute-matched to centralized training, so they do not support the
  primary non-inferiority claim.
- The fair like-for-like E=5 algorithm comparison is FedProx E=5 versus FedAvg
  E=5. Centralized results may be shown alongside them as context.
- E=5 must not be called more communication-efficient unless round histories
  demonstrate that it reaches a predefined performance level in fewer rounds.

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
