"""One-shot final test evaluation for the primary RQ1 comparison.

The protocol is frozen in D-029--D-031: centralized MLP versus FedProx E=1,
mu=1, training seeds 0--4, paired complete-day bootstrap, 10,000 repetitions,
and a two-sided 95% percentile interval. Test results are written only after
all fits and bootstrap calculations finish successfully.

Use ``--smoke`` to exercise and time the bootstrap on synthetic data without
loading any project dataset. Run the real evaluation without arguments once.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solarfl.data.features import MODEL_MATRIX, ROOT, build_features
from solarfl.data.splits import client_ids
from solarfl.eval.metrics import mae, rmse, skill
from solarfl.federated.fedprox import fit_fedprox
from solarfl.labels.capacity import station_labels
from solarfl.models.mlp import fit_mlp

SEEDS = (0, 1, 2, 3, 4)
FEDPROX_MU = 1.0
LOCAL_EPOCHS = 1
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95
NONINFERIORITY_MARGIN = 0.005
COLS = [c for c in MODEL_MATRIX if not c.startswith("weather_future_")]

DETAIL_PATH = ROOT / "results/test_evaluation_detail.csv"
BY_SEED_PATH = ROOT / "results/test_evaluation_by_seed.csv"
BOOTSTRAP_PATH = ROOT / "results/test_evaluation_bootstrap.csv"
SUMMARY_PATH = ROOT / "results/test_evaluation_summary.csv"


def _clip(prediction: np.ndarray) -> np.ndarray:
    return np.clip(prediction, 0.0, 1.0)


def _bootstrap_macro_gaps(
    errors: pd.DataFrame,
    repetitions: int,
    random_seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Return seed-averaged paired macro-MAE gaps from day resamples.

    ``errors`` has one row per client/timestamp/seed and retains the two
    methods' absolute errors on that same observation. Aggregating before
    resampling is exact for MAE and avoids copying hourly rows 10,000 times.
    A multinomial row records how often every date was drawn; it is equivalent
    to drawing ``n_dates`` dates independently with replacement.
    """
    required = {
        "calendar_day", "client", "seed", "central_abs_error",
        "fedprox_abs_error",
    }
    missing = required - set(errors.columns)
    if missing:
        raise ValueError(f"bootstrap errors table is missing {sorted(missing)}")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")

    daily = (
        errors.groupby(["calendar_day", "seed", "client"], sort=True)
        .agg(
            central_sum=("central_abs_error", "sum"),
            fedprox_sum=("fedprox_abs_error", "sum"),
            count=("central_abs_error", "size"),
        )
    )
    dates = np.sort(errors["calendar_day"].unique())
    seeds = np.sort(errors["seed"].unique())
    clients = np.sort(errors["client"].unique())
    full_index = pd.MultiIndex.from_product(
        [dates, seeds, clients], names=["calendar_day", "seed", "client"]
    )
    daily = daily.reindex(full_index, fill_value=0)
    shape = (len(dates), len(seeds), len(clients))
    central_sum = daily["central_sum"].to_numpy().reshape(shape)
    fedprox_sum = daily["fedprox_sum"].to_numpy().reshape(shape)
    counts = daily["count"].to_numpy().reshape(shape)

    rng = np.random.default_rng(random_seed)
    # Each row contains the number of times each complete date was selected.
    weights = rng.multinomial(
        len(dates), np.full(len(dates), 1.0 / len(dates)), size=repetitions
    )
    denominators = np.einsum("rd,dsc->rsc", weights, counts)
    if np.any(denominators == 0):
        raise ValueError(
            "a bootstrap resample omitted every observation for at least one "
            "seed/client; the complete-client macro-MAE is undefined"
        )

    central_mae = (
        np.einsum("rd,dsc->rsc", weights, central_sum) / denominators
    )
    fedprox_mae = (
        np.einsum("rd,dsc->rsc", weights, fedprox_sum) / denominators
    )
    # client mean -> macro-MAE; seed mean -> D-030 primary estimand
    return (fedprox_mae - central_mae).mean(axis=2).mean(axis=1)


def smoke_benchmark() -> pd.DataFrame:
    """Validate and time the bootstrap kernel using synthetic hourly errors."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=49, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, range(7), SEEDS, range(10)],
        names=["calendar_day", "client", "seed", "hour"],
    )
    synthetic = index.to_frame(index=False)
    synthetic["central_abs_error"] = rng.uniform(0.0, 0.25, len(synthetic))
    synthetic["fedprox_abs_error"] = np.clip(
        synthetic["central_abs_error"]
        + rng.normal(0.003, 0.02, len(synthetic)),
        0.0,
        None,
    )

    timings = []
    for repetitions in (100, 1_000):
        start = time.perf_counter()
        gaps = _bootstrap_macro_gaps(synthetic, repetitions, random_seed=0)
        elapsed = time.perf_counter() - start
        timings.append({
            "repetitions": repetitions,
            "seconds": elapsed,
            "milliseconds_per_repetition": 1_000 * elapsed / repetitions,
            "mean_gap": gaps.mean(),
        })

    first = _bootstrap_macro_gaps(synthetic, 100, random_seed=0)
    second = _bootstrap_macro_gaps(synthetic, 100, random_seed=0)
    if not np.array_equal(first, second):
        raise AssertionError("bootstrap seed does not reproduce identical draws")

    equal = synthetic.copy()
    equal["fedprox_abs_error"] = equal["central_abs_error"]
    if not np.allclose(_bootstrap_macro_gaps(equal, 100), 0.0):
        raise AssertionError("equal paired errors must produce zero gaps")

    return pd.DataFrame(timings)


def _load_development_data() -> dict:
    """Load train/validation only; test is deliberately loaded later."""
    return {
        sid: {split: build_features(sid, split=split)
              for split in ("train", "val")}
        for sid in client_ids()
    }


def _load_test_data(client_order: list[str]) -> dict:
    """Open the frozen test partitions only after all model fits are complete."""
    return {
        sid: build_features(sid, split="test", return_timestamps=True)
        for sid in client_order
    }


def run(
    output_dir: Path = ROOT / "results",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_path = output_dir / DETAIL_PATH.name
    by_seed_path = output_dir / BY_SEED_PATH.name
    bootstrap_path = output_dir / BOOTSTRAP_PATH.name
    summary_path = output_dir / SUMMARY_PATH.name
    output_paths = (detail_path, by_seed_path, bootstrap_path, summary_path)
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to repeat the one-shot test evaluation because output "
            f"already exists: {[str(path) for path in existing]}"
        )

    development = _load_development_data()
    clients = list(development)
    labels = station_labels()
    train_X = [development[sid]["train"][0][COLS].values for sid in clients]
    train_y = [development[sid]["train"][1].values for sid in clients]
    train_X_pool = np.vstack(train_X)
    train_y_pool = np.concatenate(train_y)
    val_X_pool = np.vstack(
        [development[sid]["val"][0][COLS].values for sid in clients]
    )
    val_y_pool = np.concatenate(
        [development[sid]["val"][1].values for sid in clients]
    )

    fits = {}
    for seed in SEEDS:
        print(f"seed {seed}: fitting centralized MLP", flush=True)
        central = fit_mlp(
            train_X_pool, train_y_pool, val_X_pool, val_y_pool, seed=seed
        )
        print(f"seed {seed}: fitting FedProx E=1 mu={FEDPROX_MU:g}", flush=True)
        fedprox = fit_fedprox(
            train_X,
            train_y,
            val_X_pool,
            val_y_pool,
            seed=seed,
            prox_mu=FEDPROX_MU,
            local_epochs=LOCAL_EPOCHS,
            verbose=True,
        )
        fits[seed] = {"central_mlp": central, "fedprox_e1": fedprox}

    print("all models frozen; opening test partitions once", flush=True)
    test = _load_test_data(clients)
    detail_rows = []
    error_frames = []

    for seed in SEEDS:
        for sid in clients:
            X, y, timestamps = test[sid]
            values = X[COLS].values
            actual = y.to_numpy()
            persistence = X["history_capacity_factor"].to_numpy()
            central_pred = _clip(fits[seed]["central_mlp"].predict(values))
            fedprox_pred = _clip(fits[seed]["fedprox_e1"].predict(values))

            for method, prediction in (
                ("central_mlp", central_pred), ("fedprox_e1", fedprox_pred)
            ):
                detail_rows.append({
                    "seed": seed,
                    "client": labels[sid]["label"],
                    "method": method,
                    "n_test": len(actual),
                    "mae": mae(actual, prediction),
                    "rmse": rmse(actual, prediction),
                    "skill": skill(actual, prediction, persistence),
                })

            error_frames.append(pd.DataFrame({
                "calendar_day": timestamps.dt.normalize().to_numpy(),
                "client": labels[sid]["label"],
                "seed": seed,
                "central_abs_error": np.abs(actual - central_pred),
                "fedprox_abs_error": np.abs(actual - fedprox_pred),
            }))

    detail = pd.DataFrame(detail_rows).sort_values(["seed", "method", "client"])
    by_seed_rows = []
    for seed, group in detail.groupby("seed", sort=True):
        macro = group.groupby("method")["mae"].mean()
        by_seed_rows.append({
            "seed": seed,
            "central_macro_mae": macro["central_mlp"],
            "fedprox_macro_mae": macro["fedprox_e1"],
            "macro_gap_fedprox_minus_central": (
                macro["fedprox_e1"] - macro["central_mlp"]
            ),
        })
    by_seed = pd.DataFrame(by_seed_rows)

    errors = pd.concat(error_frames, ignore_index=True)
    gaps = _bootstrap_macro_gaps(
        errors, BOOTSTRAP_REPETITIONS, random_seed=BOOTSTRAP_SEED
    )
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
    passes = bool(upper < NONINFERIORITY_MARGIN)
    bootstrap = pd.DataFrame({
        "repetition": np.arange(1, BOOTSTRAP_REPETITIONS + 1),
        "macro_gap_fedprox_minus_central": gaps,
    })
    summary = pd.DataFrame([{
        "n_calendar_days": errors["calendar_day"].nunique(),
        "n_clients": errors["client"].nunique(),
        "n_training_seeds": len(SEEDS),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "central_macro_mae_mean": by_seed["central_macro_mae"].mean(),
        "fedprox_macro_mae_mean": by_seed["fedprox_macro_mae"].mean(),
        "observed_macro_gap": by_seed[
            "macro_gap_fedprox_minus_central"
        ].mean(),
        "ci_lower": lower,
        "ci_upper": upper,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_passes": passes,
    }])

    # Delay every write until the complete evaluation has succeeded, so the
    # output guard cannot mistake a partial run for a completed one.
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    by_seed.to_csv(by_seed_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail, by_seed, bootstrap, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="benchmark bootstrap mechanics on synthetic data only",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.smoke:
        timings = smoke_benchmark()
        print(timings.to_string(index=False))
        return

    _, by_seed, _, summary = run(args.output_dir)
    print("\n=== Test macro-MAE by training seed ===")
    print(by_seed.to_string(index=False))
    print("\n=== Frozen non-inferiority result ===")
    print(summary.to_string(index=False))
    print(f"\nsummary -> {args.output_dir / SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
