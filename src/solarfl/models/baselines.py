"""Baseline experiment: persistence vs ridge vs MLP, local vs centralized.

Produces the two endpoints the federated stage will be judged against (D-013):
  - local:       one model per client, trained only on that client's train split
  - centralized: one model trained on all clients' pooled train splits

Every result is reported per client on that client's VAL split — test stays
untouched until the protocol freeze. Two feature variants per D-014:
  - past:    weather observed at T-24 only (deployable)
  - perfect: adds reanalysis weather at T (perfect-forecast upper bound)

Run: uv run python -m solarfl.models.baselines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solarfl.data.features import MODEL_MATRIX, ROOT, build_features
from solarfl.data.splits import client_ids
from solarfl.eval.metrics import mae, rmse, skill
from solarfl.labels.capacity import station_labels
from solarfl.models.mlp import fit_mlp

SEED = 0
RESULTS_PATH = ROOT / "results/baselines_val.csv"

VARIANTS = {
    "past": [c for c in MODEL_MATRIX if not c.startswith("weather_future_")],
    "perfect": MODEL_MATRIX,
}


def _load_all() -> dict[str, dict[str, tuple[pd.DataFrame, pd.Series]]]:
    return {
        sid: {s: build_features(sid, split=s) for s in ("train", "val")}
        for sid in client_ids()
    }


def _fit_ridge(X: np.ndarray, y: np.ndarray):
    # standardize inside the pipeline so alpha penalizes comparable scales;
    # raw radiation (~1000) next to sin/cos (~1) would make alpha meaningless
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def _clip(pred: np.ndarray) -> np.ndarray:
    # capacity factor is physically bounded; models shouldn't be graded on
    # predictions outside what the target definition allows (D-010)
    return np.clip(pred, 0.0, 1.0)


def run(output_dir: Path = ROOT / "results") -> pd.DataFrame:
    output_path = output_dir / RESULTS_PATH.name
    data = _load_all()
    labels = station_labels()
    rows = []

    for variant, cols in VARIANTS.items():
        # ---- fit ----------------------------------------------------------
        local_ridge, local_mlp = {}, {}
        for sid, d in data.items():
            Xt, yt = d["train"][0][cols].values, d["train"][1].values
            Xv, yv = d["val"][0][cols].values, d["val"][1].values
            local_ridge[sid] = _fit_ridge(Xt, yt)
            local_mlp[sid] = fit_mlp(Xt, yt, Xv, yv, seed=SEED)

        Xt_pool = np.vstack([d["train"][0][cols].values for d in data.values()])
        yt_pool = np.concatenate([d["train"][1].values for d in data.values()])
        Xv_pool = np.vstack([d["val"][0][cols].values for d in data.values()])
        yv_pool = np.concatenate([d["val"][1].values for d in data.values()])
        central_ridge = _fit_ridge(Xt_pool, yt_pool)
        # centralized early-stopping watches the pooled val set — per-client
        # val would hand the "centralized" model 7 private stopping criteria
        central_mlp = fit_mlp(Xt_pool, yt_pool, Xv_pool, yv_pool, seed=SEED)

        # ---- evaluate per client on its val split -------------------------
        for sid, d in data.items():
            Xv, yv = d["val"][0][cols].values, d["val"][1].values
            y_persist = d["val"][0]["history_capacity_factor"].values

            preds = {
                ("persistence", "-"): y_persist,
                ("ridge", "local"): _clip(local_ridge[sid].predict(Xv)),
                ("ridge", "central"): _clip(central_ridge.predict(Xv)),
                ("mlp", "local"): _clip(local_mlp[sid].predict(Xv)),
                ("mlp", "central"): _clip(central_mlp.predict(Xv)),
            }
            for (model, regime), yhat in preds.items():
                rows.append({
                    "client": labels[sid]["label"],
                    "variant": variant,
                    "model": model,
                    "regime": regime,
                    "n_val": len(yv),
                    "mae": round(mae(yv, yhat), 4),
                    "rmse": round(rmse(yv, yhat), 4),
                    "skill": round(skill(yv, yhat, y_persist), 4),
                })

    results = pd.DataFrame(rows).sort_values(["variant", "client", "model", "regime"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    results = run(args.output_dir)
    for variant in VARIANTS:
        sub = results[results["variant"] == variant]
        pivot = sub.pivot_table(index="client", columns=["model", "regime"],
                                values="mae", sort=False)
        print(f"\n=== MAE on val, variant = {variant} "
              f"(daylight hours only; skill vs same-hour-yesterday) ===")
        print(pivot.round(4).to_string())
        pivot_s = sub.pivot_table(index="client", columns=["model", "regime"],
                                  values="skill", sort=False)
        print("\nskill (>0 beats persistence):")
        print(pivot_s.round(3).to_string())
    print(f"\nfull table -> {args.output_dir / RESULTS_PATH.name}")
