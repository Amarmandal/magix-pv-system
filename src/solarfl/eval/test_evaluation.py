"""One-shot final test evaluation for the primary RQ1 comparison.

The primary protocol is frozen in D-029--D-031: centralized MLP versus FedProx
E=1, mu=1, training seeds 0--4, a paired union-calendar day bootstrap, 10,000
repetitions, and a two-sided 95% percentile interval. Review-motivated
common-overlap and client-stratified calendar bootstraps are reported alongside
that primary analysis. Test results are written only after all fits and
bootstrap calculations finish successfully.

Use ``--smoke`` to exercise and time the bootstrap on synthetic data without
loading any project dataset. Run the real evaluation without arguments once.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solarfl.data.features import PAST_MODEL_MATRIX, ROOT, build_features
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
COLS = PAST_MODEL_MATRIX

DETAIL_PATH = ROOT / "results/test_evaluation_detail.csv"
BY_SEED_PATH = ROOT / "results/test_evaluation_by_seed.csv"
BOOTSTRAP_PATH = ROOT / "results/test_evaluation_bootstrap.csv"
SUMMARY_PATH = ROOT / "results/test_evaluation_summary.csv"
PAIRED_ERRORS_PATH = ROOT / "results/test_evaluation_paired_errors.csv"
CALENDAR_BOOTSTRAP_PATH = (
    ROOT / "results/test_evaluation_calendar_bootstrap.csv"
)
CALENDAR_SENSITIVITY_PATH = (
    ROOT / "results/test_evaluation_calendar_sensitivity.csv"
)

UNION_CALENDAR = "union_calendar"
COMMON_OVERLAP = "common_overlap"
CLIENT_STRATIFIED = "client_stratified"

_ERROR_COLUMNS = {
    "calendar_day",
    "client",
    "seed",
    "central_abs_error",
    "fedprox_abs_error",
}


def _clip(prediction: np.ndarray) -> np.ndarray:
    return np.clip(prediction, 0.0, 1.0)


def _bootstrap_macro_gaps(
    errors: pd.DataFrame,
    repetitions: int,
    random_seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Return paired macro-MAE gaps from union-calendar day resamples.

    ``errors`` has one row per client/timestamp/seed and retains the two
    methods' absolute errors on that same observation. Aggregating before
    resampling is exact for MAE and avoids copying hourly rows 10,000 times.
    A multinomial row records how often every date was drawn; it is equivalent
    to drawing ``n_dates`` union-calendar dates independently with replacement.

    A client without observations on a selected date contributes a zero count,
    not a zero error. Its number of contributing date occurrences can therefore
    vary between repetitions. The same date weights are used for both methods,
    all clients, and all training seeds to retain pairing and shared-date
    weather dependence.
    """
    missing = _ERROR_COLUMNS - set(errors.columns)
    if missing:
        raise ValueError(f"bootstrap errors table is missing {sorted(missing)}")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if errors.empty:
        raise ValueError("bootstrap errors table must not be empty")

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


def _common_calendar_days(errors: pd.DataFrame) -> pd.Index:
    """Return dates represented by every client in ``errors``."""
    missing = {"calendar_day", "client"} - set(errors.columns)
    if missing:
        raise ValueError(f"bootstrap errors table is missing {sorted(missing)}")
    if errors.empty:
        raise ValueError("bootstrap errors table must not be empty")

    represented_clients = (
        errors[["calendar_day", "client"]]
        .drop_duplicates()
        .groupby("calendar_day", sort=True)["client"]
        .nunique()
    )
    common = represented_clients[
        represented_clients == errors["client"].nunique()
    ].index
    if common.empty:
        raise ValueError("clients have no common calendar dates")
    return common


def _bootstrap_common_overlap_gaps(
    errors: pd.DataFrame,
    repetitions: int,
    random_seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Bootstrap dates shared by every client using synchronized weights."""
    common = _common_calendar_days(errors)
    overlap = errors[errors["calendar_day"].isin(common)]
    return _bootstrap_macro_gaps(overlap, repetitions, random_seed)


def _bootstrap_client_stratified_gaps(
    errors: pd.DataFrame,
    repetitions: int,
    random_seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Bootstrap each client's own dates while fixing its number of draws.

    A client with ``n_client_dates`` contributes exactly that many sampled date
    occurrences in every repetition. Date weights are shared across methods and
    seeds within a client, but clients are sampled independently. This preserves
    method/seed pairing and client calendar sizes at the cost of not preserving
    cross-client weather dependence on shared dates.
    """
    missing = _ERROR_COLUMNS - set(errors.columns)
    if missing:
        raise ValueError(f"bootstrap errors table is missing {sorted(missing)}")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if errors.empty:
        raise ValueError("bootstrap errors table must not be empty")

    clients = np.sort(errors["client"].unique())
    seeds = np.sort(errors["seed"].unique())
    rng = np.random.default_rng(random_seed)
    client_gaps = []

    for client in clients:
        client_errors = errors[errors["client"] == client]
        daily = (
            client_errors.groupby(["calendar_day", "seed"], sort=True)
            .agg(
                central_sum=("central_abs_error", "sum"),
                fedprox_sum=("fedprox_abs_error", "sum"),
                count=("central_abs_error", "size"),
            )
        )
        dates = np.sort(client_errors["calendar_day"].unique())
        full_index = pd.MultiIndex.from_product(
            [dates, seeds], names=["calendar_day", "seed"]
        )
        daily = daily.reindex(full_index, fill_value=0)
        shape = (len(dates), len(seeds))
        central_sum = daily["central_sum"].to_numpy().reshape(shape)
        fedprox_sum = daily["fedprox_sum"].to_numpy().reshape(shape)
        counts = daily["count"].to_numpy().reshape(shape)

        weights = rng.multinomial(
            len(dates),
            np.full(len(dates), 1.0 / len(dates)),
            size=repetitions,
        )
        denominators = np.einsum("rd,ds->rs", weights, counts)
        if np.any(denominators == 0):
            raise ValueError(
                f"a client-stratified resample omitted every observation for "
                f"client {client!r} and at least one seed"
            )

        central_mae = (
            np.einsum("rd,ds->rs", weights, central_sum) / denominators
        )
        fedprox_mae = (
            np.einsum("rd,ds->rs", weights, fedprox_sum) / denominators
        )
        client_gaps.append(fedprox_mae - central_mae)

    # client mean -> macro-MAE; seed mean -> D-030 primary estimand
    return np.stack(client_gaps, axis=2).mean(axis=2).mean(axis=1)


def _observed_macro_metrics(errors: pd.DataFrame) -> tuple[float, float, float]:
    """Return seed-averaged central, FedProx, and paired macro-MAE gap."""
    missing = _ERROR_COLUMNS - set(errors.columns)
    if missing:
        raise ValueError(f"bootstrap errors table is missing {sorted(missing)}")
    if errors.empty:
        raise ValueError("bootstrap errors table must not be empty")

    client_seed = errors.groupby(["seed", "client"], sort=True).agg(
        central_mae=("central_abs_error", "mean"),
        fedprox_mae=("fedprox_abs_error", "mean"),
    )
    central = float(client_seed["central_mae"].groupby("seed").mean().mean())
    fedprox = float(client_seed["fedprox_mae"].groupby("seed").mean().mean())
    return central, fedprox, fedprox - central


def _calendar_sensitivity_tables(
    errors: pd.DataFrame,
    repetitions: int,
    random_seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
    noninferiority_margin: float = NONINFERIORITY_MARGIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bootstrap draws and summaries for three calendar designs."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")

    common_days = _common_calendar_days(errors)
    overlap = errors[errors["calendar_day"].isin(common_days)]
    gap_samples = {
        UNION_CALENDAR: _bootstrap_macro_gaps(
            errors, repetitions, random_seed
        ),
        COMMON_OVERLAP: _bootstrap_common_overlap_gaps(
            errors, repetitions, random_seed
        ),
        CLIENT_STRATIFIED: _bootstrap_client_stratified_gaps(
            errors, repetitions, random_seed
        ),
    }
    analysis_errors = {
        UNION_CALENDAR: errors,
        COMMON_OVERLAP: overlap,
        CLIENT_STRATIFIED: errors,
    }
    calendar_scope = {
        UNION_CALENDAR: "all_available_client_dates",
        COMMON_OVERLAP: "dates_shared_by_all_clients",
        CLIENT_STRATIFIED: "all_available_client_dates",
    }
    shared_date_weights = {
        UNION_CALENDAR: True,
        COMMON_OVERLAP: True,
        CLIENT_STRATIFIED: False,
    }
    fixed_draws_per_client = {
        UNION_CALENDAR: False,
        COMMON_OVERLAP: True,
        CLIENT_STRATIFIED: True,
    }
    alpha = 1.0 - confidence_level
    distribution_frames = []
    summary_rows = []

    for design in (UNION_CALENDAR, COMMON_OVERLAP, CLIENT_STRATIFIED):
        design_errors = analysis_errors[design]
        gaps = gap_samples[design]
        lower, upper = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
        central, fedprox, observed_gap = _observed_macro_metrics(design_errors)
        client_days = (
            design_errors[["client", "calendar_day"]]
            .drop_duplicates()
            .groupby("client", sort=True)["calendar_day"]
            .nunique()
        )
        day_counts = ";".join(
            f"{client}:{count}" for client, count in client_days.items()
        )
        if design == UNION_CALENDAR:
            draws = str(errors["calendar_day"].nunique())
        elif design == COMMON_OVERLAP:
            draws = str(len(common_days))
        else:
            draws = day_counts

        distribution_frames.append(pd.DataFrame({
            "bootstrap_design": design,
            "repetition": np.arange(1, repetitions + 1),
            "macro_gap_fedprox_minus_central": gaps,
        }))
        summary_rows.append({
            "bootstrap_design": design,
            "calendar_scope": calendar_scope[design],
            "shared_date_weights_across_clients": shared_date_weights[design],
            "fixed_draws_per_client": fixed_draws_per_client[design],
            "draws_per_repetition": draws,
            "client_calendar_days": day_counts,
            "n_union_calendar_days": errors["calendar_day"].nunique(),
            "n_common_calendar_days": len(common_days),
            "n_clients": errors["client"].nunique(),
            "n_training_seeds": errors["seed"].nunique(),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": random_seed,
            "confidence_level": confidence_level,
            "central_macro_mae_mean": central,
            "fedprox_macro_mae_mean": fedprox,
            "observed_macro_gap": observed_gap,
            "ci_lower": lower,
            "ci_upper": upper,
            "noninferiority_margin": noninferiority_margin,
            "noninferiority_passes": bool(upper < noninferiority_margin),
        })

    return pd.concat(distribution_frames, ignore_index=True), pd.DataFrame(
        summary_rows
    )


def smoke_benchmark() -> pd.DataFrame:
    """Validate and time all calendar bootstraps on unequal test calendars."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=49, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, range(7), SEEDS, range(10)],
        names=["calendar_day", "client", "seed", "hour"],
    )
    synthetic = index.to_frame(index=False)
    first_day_by_client = {
        0: dates[0],
        1: dates[-16],
        2: dates[-17],
        3: dates[-17],
        4: dates[-17],
        5: dates[-17],
        6: dates[-18],
    }
    client_first_days = synthetic["client"].map(first_day_by_client)
    synthetic = synthetic[
        synthetic["calendar_day"] >= client_first_days
    ].reset_index(drop=True)
    synthetic["central_abs_error"] = rng.uniform(0.0, 0.25, len(synthetic))
    synthetic["fedprox_abs_error"] = np.clip(
        synthetic["central_abs_error"]
        + rng.normal(0.003, 0.02, len(synthetic)),
        0.0,
        None,
    )

    methods = {
        UNION_CALENDAR: _bootstrap_macro_gaps,
        COMMON_OVERLAP: _bootstrap_common_overlap_gaps,
        CLIENT_STRATIFIED: _bootstrap_client_stratified_gaps,
    }
    timings = []
    for design, method in methods.items():
        for repetitions in (100, 1_000):
            start = time.perf_counter()
            gaps = method(synthetic, repetitions, random_seed=0)
            elapsed = time.perf_counter() - start
            timings.append({
                "bootstrap_design": design,
                "repetitions": repetitions,
                "seconds": elapsed,
                "milliseconds_per_repetition": (
                    1_000 * elapsed / repetitions
                ),
                "mean_gap": gaps.mean(),
            })

        first = method(synthetic, 100, random_seed=0)
        second = method(synthetic, 100, random_seed=0)
        if not np.array_equal(first, second):
            raise AssertionError(
                f"{design} bootstrap seed does not reproduce identical draws"
            )

    equal = synthetic.copy()
    equal["fedprox_abs_error"] = equal["central_abs_error"]
    for design, method in methods.items():
        if not np.allclose(method(equal, 100), 0.0):
            raise AssertionError(
                f"{design} bootstrap must preserve zero paired gaps"
            )

    if len(_common_calendar_days(synthetic)) != 16:
        raise AssertionError("synthetic common calendar must contain 16 days")

    return pd.DataFrame(timings)


def _load_development_data() -> dict:
    """Load train/validation only; test is deliberately loaded later."""
    return {
        sid: {split: build_features(sid, split=split, row_set="past")
              for split in ("train", "val")}
        for sid in client_ids()
    }


def _load_test_data(client_order: list[str]) -> dict:
    """Open the frozen test partitions only after all model fits are complete."""
    return {
        sid: build_features(
            sid, split="test", row_set="past", return_timestamps=True
        )
        for sid in client_order
    }


def run(
    output_dir: Path = ROOT / "results",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_path = output_dir / DETAIL_PATH.name
    by_seed_path = output_dir / BY_SEED_PATH.name
    bootstrap_path = output_dir / BOOTSTRAP_PATH.name
    summary_path = output_dir / SUMMARY_PATH.name
    paired_errors_path = output_dir / PAIRED_ERRORS_PATH.name
    calendar_bootstrap_path = output_dir / CALENDAR_BOOTSTRAP_PATH.name
    calendar_sensitivity_path = output_dir / CALENDAR_SENSITIVITY_PATH.name
    output_paths = (
        detail_path,
        by_seed_path,
        bootstrap_path,
        summary_path,
        paired_errors_path,
        calendar_bootstrap_path,
        calendar_sensitivity_path,
    )
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
                "timestamp": timestamps.to_numpy(),
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

    errors = (
        pd.concat(error_frames, ignore_index=True)
        .sort_values(["seed", "client", "timestamp"])
        .reset_index(drop=True)
    )
    calendar_bootstrap, calendar_sensitivity = _calendar_sensitivity_tables(
        errors,
        BOOTSTRAP_REPETITIONS,
        random_seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
        noninferiority_margin=NONINFERIORITY_MARGIN,
    )
    bootstrap = (
        calendar_bootstrap[
            calendar_bootstrap["bootstrap_design"] == UNION_CALENDAR
        ]
        .drop(columns="bootstrap_design")
        .reset_index(drop=True)
    )
    primary_sensitivity = calendar_sensitivity.iloc[0]
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
        "ci_lower": primary_sensitivity["ci_lower"],
        "ci_upper": primary_sensitivity["ci_upper"],
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_passes": primary_sensitivity[
            "noninferiority_passes"
        ],
    }])

    # Delay every write until the complete evaluation has succeeded, so the
    # output guard cannot mistake a partial run for a completed one.
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    by_seed.to_csv(by_seed_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    summary.to_csv(summary_path, index=False)
    errors.to_csv(paired_errors_path, index=False)
    calendar_bootstrap.to_csv(calendar_bootstrap_path, index=False)
    calendar_sensitivity.to_csv(calendar_sensitivity_path, index=False)
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
    sensitivity = pd.read_csv(
        args.output_dir / CALENDAR_SENSITIVITY_PATH.name
    )
    print("\n=== Calendar-bootstrap sensitivity ===")
    print(sensitivity.to_string(index=False))
    print(f"\nsummary -> {args.output_dir / SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
