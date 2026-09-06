"""FedProx: FedAvg with a proximal term against non-IID client drift.

Clone of fedavg.py (kept separate so the frozen FedAvg run stays untouched)
with one algorithmic change: during local training each client minimizes
MSE + (mu/2)·||w − w_global||² (Li et al. 2020). The penalty pulls local
weights back toward the model the server broadcast that round, so clients
with different data distributions can't drift as far apart before averaging.
Server-side aggregation is unchanged. With prox_mu=0 the update is exact
FedAvg — that arm doubles as a check that this clone reproduces
results/fedavg_val.csv.

Two experiment axes on top of the frozen D-018 setup:
- local_epochs ∈ {1, 5}: validation explores the effect of extra local passes.
  The proximal penalty can affect either setting; it is zero at broadcast
  initialization but can become nonzero after the first minibatch update.
  FedAvg is also evaluated at E=5 to compare the methods at the same E.
- mu ∈ {0.001, 0.01, 0.1, 1}: the FedProx paper's grid, selected per
  (variant, E) by pooled val MAE — the same selection signal as D-018.

Everything else (MLP(64, 32), Adam 1e-3, MSE, batch 256, seed 0, early
stopping on pooled val MAE, patience 25, best-weight restore, global scaler
from per-client aggregates) is identical to fedavg.py. Results land in
results/fedprox_val.csv with the baseline columns plus local_epochs and mu.

Run: uv run python -m solarfl.federated.fedprox
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from solarfl.data.features import (
    MODEL_MATRIX,
    PAST_MODEL_MATRIX,
    ROOT,
    build_features,
)
from solarfl.data.splits import client_ids
from solarfl.eval.metrics import mae, rmse, skill
from solarfl.labels.capacity import station_labels
from solarfl.models.mlp import MLP, FittedMLP

SEED = 0
RESULTS_PATH = ROOT / "results/fedprox_val.csv"

VARIANTS = {
    "past": PAST_MODEL_MATRIX,
    "perfect": MODEL_MATRIX,
}

MU_GRID = (0.001, 0.01, 0.1, 1.0)  # FedProx paper's sweep (Li et al. 2020)
LOCAL_EPOCHS = (1, 5)


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
    """FedAvg aggregation: per-parameter average weighted by sample count.

    FedProx changes only the local objective — the server still averages.
    """
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
    prox_mu: float,
) -> dict:
    """One client's round: load global weights, train E epochs, return weights.

    The optimizer is created fresh each round — clients are stateless;
    carrying Adam moments across rounds would mix them with weights the
    server has since replaced.

    The proximal anchor is the broadcast model, detached so gradients flow
    only through the local copy, never the reference point. prox_mu=0 skips
    the term entirely and is bit-for-bit the FedAvg update.
    """
    model = MLP(n_features=n_features)
    model.load_state_dict(global_state)
    anchor = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Zt)),
                                  max(1, len(Zt) // batch_size)):
            opt.zero_grad()
            # Local MSE
            loss = loss_fn(model(Zt[idx]), yt[idx])
            # It means that there is no penalty; that means fedavg and fedprox behaves identically
            if prox_mu > 0.0:
                prox = sum((p - a).pow(2).sum()
                           for p, a in zip(model.parameters(), anchor))
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            opt.step()
    return {k: v.clone() for k, v in model.state_dict().items()}


def fit_fedprox(
    X_by_client: list[np.ndarray],
    y_by_client: list[np.ndarray],
    X_val_pool: np.ndarray,
    y_val_pool: np.ndarray,
    seed: int = 0,
    prox_mu: float = 0.0,
    local_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_rounds: int = 300,
    patience: int = 25,
    verbose: bool = False,
) -> FittedMLP:
    """Full-participation FedProx over the given clients (FedAvg at prox_mu=0).

    At E=1 each round visits every training row once, approximately matching
    one centralized epoch's data exposure. Optimizer steps and moments are not
    matched: clients reset Adam each round and their models are averaged.
    E=5 makes five local passes per round and is an exploratory validation
    comparison. Stopping uses pooled validation MAE and restores the best
    round; the simulation pools validation predictors and targets centrally.
    ``seed`` is supplied by the runner: 0 for the broad validation sweep and
    each paired seed 0--4 for the frozen centralized-versus-FedProx test.
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
    t0 = time.perf_counter()

    rnd = 0
    for rnd in range(1, max_rounds + 1):
        local_states = [
            _local_update(global_state, Zt, yt, rng, local_epochs, lr,
                          batch_size, n_features, prox_mu)
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

        if verbose:
            # \r rewrites one status line in place instead of scrolling
            # 300 lines past the terminal; flush because there's no newline
            print(f"\r  round {rnd:3d}/{max_rounds}  "
                  f"val_mae={val_mae:.4f}  best={best_mae:.4f}  "
                  f"since_best={since_best:2d}/{patience}",
                  end="", flush=True)

        if since_best >= patience:
            break

    if verbose:
        print(f"\n  stopped after {rnd} rounds, best val_mae={best_mae:.4f}, "
              f"{time.perf_counter() - t0:.0f}s")

    model.load_state_dict(best_state)
    return fitted


def _clip(pred: np.ndarray) -> np.ndarray:
    # same physical-range clip as the baselines (D-017)
    return np.clip(pred, 0.0, 1.0)


def run(output_dir: Path = ROOT / "results") -> pd.DataFrame:
    output_path = output_dir / RESULTS_PATH.name
    data = {
        sid: {
            s: build_features(sid, split=s, row_set="common")
            for s in ("train", "val")
        }
        for sid in client_ids()
    }
    labels = station_labels()
    rows = []
    n_fits = len(VARIANTS) * len(LOCAL_EPOCHS) * (1 + len(MU_GRID))
    fit_no = 0

    def _header(variant: str, local_epochs: int, prox_mu: float) -> None:
        nonlocal fit_no
        fit_no += 1
        algo = "fedavg" if prox_mu == 0.0 else f"fedprox mu={prox_mu}"
        print(f"[{fit_no}/{n_fits}] variant={variant} E={local_epochs} {algo}",
              flush=True)

    for variant, cols in VARIANTS.items():
        Xt = [d["train"][0][cols].values for d in data.values()]
        yt = [d["train"][1].values for d in data.values()]
        Xv_pool = np.vstack([d["val"][0][cols].values for d in data.values()])
        yv_pool = np.concatenate([d["val"][1].values for d in data.values()])

        for local_epochs in LOCAL_EPOCHS:
            # FedAvg arm (prox_mu=0): at E=1 it must reproduce
            # results/fedavg_val.csv; at E=5 it's the drift-prone control
            # FedProx is judged against.
            _header(variant, local_epochs, 0.0)
            candidates = {0.0: fit_fedprox(Xt, yt, Xv_pool, yv_pool, seed=SEED,
                                           prox_mu=0.0,
                                           local_epochs=local_epochs,
                                           verbose=True)}
            # mu is selected on clipped pooled val MAE — the same quantity we
            # report, so selection and headline metric can't disagree.
            prox_fits = {}
            for prox_mu in MU_GRID:
                _header(variant, local_epochs, prox_mu)
                fit = fit_fedprox(Xt, yt, Xv_pool, yv_pool, seed=SEED,
                                  prox_mu=prox_mu, local_epochs=local_epochs,
                                  verbose=True)
                pooled_mae = float(np.mean(np.abs(
                    _clip(fit.predict(Xv_pool)) - yv_pool)))
                prox_fits[prox_mu] = (pooled_mae, fit)
            best_mu = min(prox_fits, key=lambda m: prox_fits[m][0])
            candidates[best_mu] = prox_fits[best_mu][1]

            for prox_mu, fit in candidates.items():
                regime = "fedavg" if prox_mu == 0.0 else "fedprox"
                for sid, d in data.items():
                    Xv, yv = d["val"][0][cols].values, d["val"][1].values
                    y_persist = d["val"][0]["history_capacity_factor"].values
                    yhat = _clip(fit.predict(Xv))
                    rows.append({
                        "client": labels[sid]["label"],
                        "variant": variant,
                        "model": "mlp",
                        "regime": regime,
                        "local_epochs": local_epochs,
                        "mu": prox_mu,
                        "n_val": len(yv),
                        "mae": round(mae(yv, yhat), 4),
                        "rmse": round(rmse(yv, yhat), 4),
                        "skill": round(skill(yv, yhat, y_persist), 4),
                    })

    results = pd.DataFrame(rows).sort_values(
        ["variant", "local_epochs", "regime", "client"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    results = run(args.output_dir)
    for variant in VARIANTS:
        for local_epochs in LOCAL_EPOCHS:
            sub = results[(results["variant"] == variant)
                          & (results["local_epochs"] == local_epochs)]
            best_mu = sub.loc[sub["regime"] == "fedprox", "mu"].iloc[0]
            print(f"\n=== FedAvg vs FedProx (best mu={best_mu}) MAE on val, "
                  f"variant = {variant}, E = {local_epochs} ===")
            print(sub.set_index(["regime", "client"])
                  [["n_val", "mae", "rmse", "skill"]].round(4).to_string())
    print(f"\nfull table -> {args.output_dir / RESULTS_PATH.name}")
