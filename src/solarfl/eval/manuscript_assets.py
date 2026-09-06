"""Generate manuscript evidence from frozen data/specs and archived results.

This command performs descriptive audits only: it never fits models, selects
hyperparameters, or evaluates new test predictions. Outputs are the six archived
evidence CSVs, input/capacity provenance and checksums.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from solarfl.data.features import (
    CLIENTS_DIR,
    MODEL_MATRIX,
    PAST_MODEL_MATRIX,
    ROOT,
    SPLITS_PATH,
    build_features,
)
from solarfl.eval.test_evaluation import FEDPROX_MU, LOCAL_EPOCHS, SEEDS
from solarfl.federated.fedavg import fit_fedavg
from solarfl.federated.fedprox import fit_fedprox
from solarfl.labels.capacity import station_labels
from solarfl.labels.capacity_audit import run as audit_capacity
from solarfl.labels.capacity_audit import sha256
from solarfl.models.mlp import MLP, fit_mlp


def feature_and_split_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count actual columns and raw/eligible rows, without fitting any model."""
    labels = station_labels()
    manifest = json.loads(SPLITS_PATH.read_text())["clients"]
    rows, dimensions = [], []
    pooled: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for sid in sorted(labels, key=lambda key: labels[key]["label"]):
        raw = pd.read_csv(CLIENTS_DIR / f"{sid}.csv", parse_dates=["measured_ts"])
        boundaries = manifest[sid]
        for split in ("train", "val", "test"):
            for row_set, variant, names in (
                ("past", "A", PAST_MODEL_MATRIX),
                ("common", "B", MODEL_MATRIX),
            ):
                X, _, ts = build_features(
                    sid, split, row_set=row_set, return_timestamps=True
                )
                X = X[names]
                start, end = (
                    pd.Timestamp(boundaries["train_end"]),
                    pd.Timestamp(boundaries["val_end"]),
                )
                raw_mask = {
                    "train": raw.measured_ts < start,
                    "val": (raw.measured_ts >= start) & (raw.measured_ts < end),
                    "test": raw.measured_ts >= end,
                }[split]
                raw_ts = raw.loc[raw_mask, "measured_ts"]
                rows.append(
                    {
                        "client": labels[sid]["label"],
                        "split": split,
                        "variant": variant,
                        "row_set": row_set,
                        "raw_rows": len(raw_ts),
                        "eligible_rows": len(X),
                        "excluded_rows": len(raw_ts) - len(X),
                        "raw_start_utc": raw_ts.min(),
                        "raw_end_utc": raw_ts.max(),
                        "eligible_start_utc": ts.min(),
                        "eligible_end_utc": ts.max(),
                        "purpose": "descriptive row audit; no model evaluation",
                    }
                )
                pooled.setdefault((variant, split), []).append(X)
    for (variant, split), matrices in pooled.items():
        X = pd.concat(matrices, ignore_index=True)
        constants = X.columns[X.nunique(dropna=False) <= 1].tolist()
        # Model construction consumes a Torch seed; isolate it from callers.
        with torch.random.fork_rng():
            model = MLP(X.shape[1])
        dimensions.append(
            {
                "variant": variant,
                "split": split,
                "eligible_rows": len(X),
                "input_width": X.shape[1],
                "nonconstant_columns": X.shape[1] - len(constants),
                "constant_columns": ";".join(constants),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "architecture": f"{X.shape[1]} -> 64 ReLU -> 32 ReLU -> 1 linear",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(dimensions)


def experiment_scope(results_dir: Path) -> pd.DataFrame:
    """Inventory archived comparisons, with their actual split and seed scope."""
    rows = []
    for filename in ("baselines_val.csv", "fedavg_val.csv", "fedprox_val.csv"):
        frame = pd.read_csv(results_dir / filename)
        keys = [
            c
            for c in ("variant", "model", "regime", "local_epochs", "mu")
            if c in frame
        ]
        for values, group in frame.groupby(keys, dropna=False, sort=True):
            settings = dict(zip(keys, values))
            rows.append(
                {
                    "source": filename,
                    "split": "validation",
                    "role": "exploratory",
                    "seeds": "0",
                    **settings,
                    "clients": group.client.nunique(),
                    "macro_mae": group.mae.mean(),
                }
            )
    frame = pd.read_csv(results_dir / "seed_robustness_val.csv")
    for method, group in frame.groupby("method"):
        rows.append(
            {
                "source": "seed_robustness_val.csv",
                "split": "validation",
                "role": "validation seed robustness",
                "seeds": ";".join(map(str, sorted(group.seed.unique()))),
                "variant": "past",
                "model": "mlp",
                "regime": method,
                "clients": group.client.nunique(),
                "macro_mae": group.groupby("seed").mae.mean().mean(),
            }
        )
    test = pd.read_csv(results_dir / "test_evaluation_by_seed.csv")
    if tuple(sorted(test.seed.unique())) != SEEDS or len(test) != len(SEEDS):
        raise ValueError("archived test seeds differ from frozen protocol")
    for method, col in (
        ("central_mlp", "central_macro_mae"),
        ("fedprox_e1", "fedprox_macro_mae"),
    ):
        rows.append(
            {
                "source": "test_evaluation_by_seed.csv",
                "split": "test",
                "role": "confirmatory",
                "seeds": ";".join(map(str, SEEDS)),
                "variant": "past",
                "model": "mlp",
                "regime": method,
                "local_epochs": LOCAL_EPOCHS if method == "fedprox_e1" else np.nan,
                "mu": FEDPROX_MU if method == "fedprox_e1" else np.nan,
                "clients": len(station_labels()),
                "macro_mae": test[col].mean(),
            }
        )
    return pd.DataFrame(rows)


def optimization_protocol() -> pd.DataFrame:
    rows = []
    for name, function in (
        ("centralized MLP", fit_mlp),
        ("FedAvg", fit_fedavg),
        ("FedProx", fit_fedprox),
    ):
        defaults = {
            k: p.default for k, p in inspect.signature(function).parameters.items()
        }
        central = name == "centralized MLP"
        rows.append(
            {
                "regime": name,
                "optimizer": "Adam",
                "learning_rate": defaults["lr"],
                "nominal_batch_size": defaults["batch_size"],
                "budget": defaults["max_epochs" if central else "max_rounds"],
                "budget_unit": "epoch" if central else "round",
                "patience": defaults["patience"],
                "optimizer_state": "retained across epochs"
                if central
                else "fresh per client per round",
                "aggregation": "none"
                if central
                else "training-row-count weighted model average",
                "stopping": "pooled validation MAE; strict improvement; restore best weights",
                "seed_input": "s; torch.manual_seed(s)",
                "shuffle_rng": "default_rng(s)"
                if central
                else "default_rng(s+i); persistent stream per ordered client",
                "training_objective": "MSE + mu/2 squared distance to broadcast weights"
                if name == "FedProx"
                else "MSE",
            }
        )
    return pd.DataFrame(rows)


def run(
    output_dir: Path, raw_dir: Path, results_dir: Path, *, download: bool = False
) -> None:
    if output_dir.resolve() == results_dir.resolve():
        raise ValueError(
            "use a separate review output directory to preserve archived results"
        )
    # A source mismatch must fail before any manuscript evidence is emitted.
    audit_capacity(raw_dir, output_dir, download=download)
    counts, dimensions = feature_and_split_audit()
    scope = experiment_scope(results_dir)
    tables = {
        "split_counts": counts,
        "feature_dimensions": dimensions,
        "experiment_scope": scope,
        "optimization_protocol": optimization_protocol(),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    inputs = [
        SPLITS_PATH,
        ROOT / "configs/features.yaml",
        ROOT / "configs/station_labels.json",
    ]
    inputs += sorted(CLIENTS_DIR.glob("*.csv"))
    inputs += [results_dir / name for name in scope.source.unique()]
    inputs += sorted((ROOT / "src/solarfl").rglob("*.py"))
    inputs += [ROOT / "docs/source_manifest.json"]
    provenance = {
        "activity": "descriptive audit; no training or new test predictions",
        "inputs": {
            str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path): sha256(path)
            for path in inputs
        },
    }
    (output_dir / "asset_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    # Hash this run's outputs only, not unrelated files in a reused directory.
    filenames = [
        "capacity_device_audit.csv",
        "station_capacity.csv",
        "capacity_provenance.json",
        "asset_provenance.json",
        *(f"{name}.csv" for name in tables),
    ]
    outputs = [output_dir / name for name in sorted(filenames)]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in outputs)
    )
    print(f"Evidence CSVs and provenance written to {output_dir}")
    print(dimensions.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduced-results/manuscript_evidence_audit",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.raw_dir, args.results_dir, download=args.download)


if __name__ == "__main__":
    main()
