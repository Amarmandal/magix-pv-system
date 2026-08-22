"""Validation-only seed robustness check for the primary RQ1 comparison.

Runs the past-weather centralized MLP, FedAvg E=1, and FedProx E=1 with the
validation-selected mu=1 across five fixed seeds. Test data is never loaded.

Run: uv run python -m solarfl.eval.seed_robustness
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarfl.data.features import MODEL_MATRIX, ROOT, build_features
from solarfl.data.splits import client_ids
from solarfl.eval.metrics import mae, rmse, skill
from solarfl.federated.fedavg import fit_fedavg
from solarfl.federated.fedprox import fit_fedprox
from solarfl.labels.capacity import station_labels
from solarfl.models.mlp import fit_mlp

SEEDS = (0, 1, 2, 3, 4)
FEDPROX_MU = 1.0  # selected for past/E=1 on validation in D-025
COLS = [c for c in MODEL_MATRIX if not c.startswith("weather_future_")]

DETAIL_PATH = ROOT / "results/seed_robustness_val.csv"
BY_SEED_PATH = ROOT / "results/seed_robustness_by_seed_val.csv"
SUMMARY_PATH = ROOT / "results/seed_robustness_summary_val.csv"


def _clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0.0, 1.0)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = {
        sid: {split: build_features(sid, split=split)
              for split in ("train", "val")}
        for sid in client_ids()
    }
    labels = station_labels()

    Xt = [d["train"][0][COLS].values for d in data.values()]
    yt = [d["train"][1].values for d in data.values()]
    Xv_pool = np.vstack([d["val"][0][COLS].values for d in data.values()])
    yv_pool = np.concatenate([d["val"][1].values for d in data.values()])
    Xt_pool = np.vstack(Xt)
    yt_pool = np.concatenate(yt)

    detail_rows = []
    by_seed_rows = []

    for seed in SEEDS:
        print(f"seed {seed}: centralized", flush=True)
        fits = {
            "central_mlp": fit_mlp(
                Xt_pool, yt_pool, Xv_pool, yv_pool, seed=seed
            ),
        }
        print(f"seed {seed}: fedavg E=1", flush=True)
        fits["fedavg_e1"] = fit_fedavg(
            Xt, yt, Xv_pool, yv_pool, seed=seed, local_epochs=1
        )
        print(f"seed {seed}: fedprox E=1 mu={FEDPROX_MU:g}", flush=True)
        fits["fedprox_e1"] = fit_fedprox(
            Xt, yt, Xv_pool, yv_pool, seed=seed,
            local_epochs=1, prox_mu=FEDPROX_MU,
        )

        seed_metrics = {}
        for method, fit in fits.items():
            client_maes = []
            absolute_error_sum = 0.0
            n_total = 0

            for sid, d in data.items():
                Xv, yv = d["val"][0][COLS].values, d["val"][1].values
                y_persist = d["val"][0]["history_capacity_factor"].values
                yhat = _clip(fit.predict(Xv))
                client_mae = mae(yv, yhat)
                client_maes.append(client_mae)
                absolute_error_sum += float(np.abs(yv - yhat).sum())
                n_total += len(yv)

                detail_rows.append({
                    "seed": seed,
                    "client": labels[sid]["label"],
                    "variant": "past",
                    "method": method,
                    "n_val": len(yv),
                    "mae": round(client_mae, 6),
                    "rmse": round(rmse(yv, yhat), 6),
                    "skill": round(skill(yv, yhat, y_persist), 6),
                })

            seed_metrics[method] = {
                "macro_mae": float(np.mean(client_maes)),
                "micro_mae": absolute_error_sum / n_total,
            }

        central_macro = seed_metrics["central_mlp"]["macro_mae"]
        for method, values in seed_metrics.items():
            by_seed_rows.append({
                "seed": seed,
                "method": method,
                "macro_mae": round(values["macro_mae"], 6),
                "micro_mae": round(values["micro_mae"], 6),
                "macro_gap_vs_central": round(
                    values["macro_mae"] - central_macro, 6
                ),
            })

    detail = pd.DataFrame(detail_rows).sort_values(["seed", "method", "client"])
    by_seed = pd.DataFrame(by_seed_rows).sort_values(["seed", "method"])

    summary_rows = []
    for method, group in by_seed.groupby("method", sort=False):
        gaps = group["macro_gap_vs_central"]
        summary_rows.append({
            "method": method,
            "n_seeds": len(group),
            "macro_mae_mean": round(group["macro_mae"].mean(), 6),
            "macro_mae_std": round(group["macro_mae"].std(ddof=1), 6),
            "macro_gap_vs_central_mean": round(gaps.mean(), 6),
            "macro_gap_vs_central_std": round(gaps.std(ddof=1), 6),
            "seeds_better_than_central": int((gaps < 0).sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("macro_mae_mean")

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    by_seed.to_csv(BY_SEED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    return detail, by_seed, summary


if __name__ == "__main__":
    _, by_seed, summary = run()
    print("\n=== Macro MAE and gap versus centralized, by seed ===")
    print(by_seed.round(6).to_string(index=False))
    print("\n=== Five-seed validation summary ===")
    print(summary.round(6).to_string(index=False))
    print(f"\ndetail -> {DETAIL_PATH}")
    print(f"by seed -> {BY_SEED_PATH}")
    print(f"summary -> {SUMMARY_PATH}")
