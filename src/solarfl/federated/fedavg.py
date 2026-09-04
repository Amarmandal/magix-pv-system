"""FedAvg: 7 clients train locally, a server averages their weights.

Hand-rolled rather than Flower so every step is inspectable: one round is
broadcast -> local epochs on each client's own train split -> weighted average
of the returned state_dicts (McMahan et al. 2017). Nothing leaves a client
except model weights and, once at startup, per-feature aggregate stats for the
global scaler — never raw rows.

Frozen to match the baselines (D-018): same MLP(64, 32), Adam 1e-3, MSE,
batch 256, seed 0, early stopping on pooled val MAE with patience 25 and
best-weight restore. Results land in results/fedavg_val.csv with the same
columns as baselines_val.csv so the tables join cleanly.

Run: uv run python -m solarfl.federated.fedavg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from solarfl.data.features import MODEL_MATRIX, ROOT, build_features
from solarfl.data.splits import client_ids
from solarfl.eval.metrics import mae, rmse, skill
from solarfl.labels.capacity import station_labels
from solarfl.models.mlp import MLP, FittedMLP

SEED = 0
RESULTS_PATH = ROOT / "results/fedavg_val.csv"

VARIANTS = {
    "past": [c for c in MODEL_MATRIX if not c.startswith("weather_future_")],
    "perfect": MODEL_MATRIX,
}


def _global_scaler(X_by_client: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Global train mu/sigma from per-client aggregates only.

    Each client contributes (sum, sum of squares, count) — enough to
    reconstruct the pooled mean/std mathematically, without any raw row
    crossing the client boundary. The result is equivalent to centralized
    train-only scaling, apart from negligible floating-point differences caused
    by calculation order and when values are cast to float32. Preprocessing is
    therefore aligned without claiming bit-for-bit identical scaler arrays.
    """
    agg = [(x.sum(axis=0), (x**2).sum(axis=0), len(x)) for x in X_by_client]
    n = sum(a[2] for a in agg)
    mu = sum(a[0] for a in agg) / n
    var = sum(a[1] for a in agg) / n - mu**2
    sigma = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    sigma[sigma == 0] = 1.0  # constant columns, same guard as fit_mlp
    return mu.astype(np.float32), sigma


def _average(states: list[dict], weights: list[int]) -> dict:
    """FedAvg aggregation: per-parameter average weighted by sample count."""
    total = sum(weights)
    return {
        k: sum(s[k] * (w / total) for s, w in zip(states, weights))
        for k in states[0]
    }


def _local_update(
    global_state: dict,
    Zt: torch.Tensor,
    yt: torch.Tensor,
    rng: np.random.Generator,
    epochs: int,
    lr: float,
    batch_size: int,
    n_features: int,
) -> dict:
    """One client's round: load global weights, train E epochs, return weights.

    The optimizer is created fresh each round — FedAvg clients are stateless;
    carrying Adam moments across rounds would mix them with weights the
    server has since replaced.
    """
    model = MLP(n_features=n_features)
    model.load_state_dict(global_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Zt)),
                                  max(1, len(Zt) // batch_size)):
            opt.zero_grad()
            loss = loss_fn(model(Zt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return {k: v.clone() for k, v in model.state_dict().items()}


def fit_fedavg(
    X_by_client: list[np.ndarray],
    y_by_client: list[np.ndarray],
    X_val_pool: np.ndarray,
    y_val_pool: np.ndarray,
    seed: int = 0,
    local_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_rounds: int = 300,
    patience: int = 25,
) -> FittedMLP:
    """Full-participation FedAvg over the given clients.

    With local_epochs=1 a round does the same gradient work as one centralized
    epoch, so max_rounds/patience mirror fit_mlp's epoch budget and the
    early-stopping comparison is like-for-like. Stopping watches the POOLED
    val MAE for the same reason the centralized baseline does (D-018).
    """
    torch.manual_seed(seed)
    mu, sigma = _global_scaler(X_by_client)

    Z = [torch.from_numpy((np.asarray(x, dtype=np.float32) - mu) / sigma)
         for x in X_by_client]
    y = [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in y_by_client]
    n_features = X_by_client[0].shape[1]
    n_train = [len(v) for v in y]
    # one rng per client so adding/removing a client never reshuffles the others
    rngs = [np.random.default_rng(seed + i) for i in range(len(Z))]

    model = MLP(n_features=n_features)
    global_state = {k: v.clone() for k, v in model.state_dict().items()}
    fitted = FittedMLP(model, mu, sigma)

    best_mae = np.inf
    best_state = None
    since_best = 0

    for _ in range(max_rounds):
        local_states = [
            _local_update(global_state, Zt, yt, rng, local_epochs, lr,
                          batch_size, n_features)
            for Zt, yt, rng in zip(Z, y, rngs)
        ]
        global_state = _average(local_states, n_train)
        model.load_state_dict(global_state)

        val_mae = float(np.mean(np.abs(fitted.predict(X_val_pool) - y_val_pool)))
        if val_mae < best_mae:
            best_mae, since_best = val_mae, 0
            best_state = {k: v.clone() for k, v in global_state.items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    return fitted


def _clip(pred: np.ndarray) -> np.ndarray:
    # same physical-range clip as the baselines (D-017)
    return np.clip(pred, 0.0, 1.0)


def run(output_dir: Path = ROOT / "results") -> pd.DataFrame:
    output_path = output_dir / RESULTS_PATH.name
    data = {
        sid: {s: build_features(sid, split=s) for s in ("train", "val")}
        for sid in client_ids()
    }
    labels = station_labels()
    rows = []

    for variant, cols in VARIANTS.items():
        Xt = [d["train"][0][cols].values for d in data.values()]
        yt = [d["train"][1].values for d in data.values()]
        Xv_pool = np.vstack([d["val"][0][cols].values for d in data.values()])
        yv_pool = np.concatenate([d["val"][1].values for d in data.values()])

        fedavg = fit_fedavg(Xt, yt, Xv_pool, yv_pool, seed=SEED)

        for sid, d in data.items():
            Xv, yv = d["val"][0][cols].values, d["val"][1].values
            y_persist = d["val"][0]["history_capacity_factor"].values
            yhat = _clip(fedavg.predict(Xv))
            rows.append({
                "client": labels[sid]["label"],
                "variant": variant,
                "model": "mlp",
                "regime": "fedavg",
                "n_val": len(yv),
                "mae": round(mae(yv, yhat), 4),
                "rmse": round(rmse(yv, yhat), 4),
                "skill": round(skill(yv, yhat, y_persist), 4),
            })

    results = pd.DataFrame(rows).sort_values(["variant", "client"])
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
        print(f"\n=== FedAvg MAE on val, variant = {variant} "
              f"(daylight hours only; skill vs same-hour-yesterday) ===")
        print(sub.set_index("client")[["n_val", "mae", "rmse", "skill"]]
              .round(4).to_string())
    print(f"\nfull table -> {args.output_dir / RESULTS_PATH.name}")
