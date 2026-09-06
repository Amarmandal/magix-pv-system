"""Generate manuscript evidence from frozen data/specs and archived results.

This command performs descriptive audits only: it never fits models, selects
hyperparameters, or evaluates new test predictions. Tables are emitted as CSV,
Markdown and LaTeX; figures as PNG, PDF and SVG for manuscript reuse.
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


def write_table(frame: pd.DataFrame, destination: Path) -> None:
    """Write portable tables without adding a Markdown rendering dependency."""
    frame.to_csv(destination.with_suffix(".csv"), index=False)
    display = frame.fillna("").map(
        lambda value: str(value).replace("|", "/").replace("\n", " ")
    )
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
    frame.to_latex(
        destination.with_suffix(".tex"), index=False, escape=True, float_format="%.6f"
    )


def feature_and_split_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count actual columns and raw/eligible rows, without fitting any model."""
    labels = station_labels()
    manifest = json.loads(SPLITS_PATH.read_text())["clients"]
    rows, dimensions, columns = [], [], []
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
        for position, name in enumerate(X.columns, 1):
            group = next(
                (
                    g
                    for g in ("history", "weather_past", "weather_future")
                    if name.startswith(g + "_")
                ),
                "geometry",
            )
            columns.append(
                {
                    "variant": variant,
                    "split": split,
                    "position": position,
                    "feature": name,
                    "group": group,
                    "unique_values": X[name].nunique(dropna=False),
                    "constant": name in constants,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(dimensions), pd.DataFrame(columns)


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


def privacy_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "partition",
                "actual_behavior": "one public-data station per analytical client in one process",
                "limitation": "possible data silo; ownership and confidentiality requirements not established",
            },
            {
                "stage": "training",
                "actual_behavior": "federated updates computed on separate client training arrays",
                "limitation": "all arrays are accessible to the offline simulation host",
            },
            {
                "stage": "scaling",
                "actual_behavior": "per-feature sums, sums of squares and counts pooled",
                "limitation": "aggregate statistics are unprotected",
            },
            {
                "stage": "aggregation",
                "actual_behavior": "client model weights returned each round",
                "limitation": "unprotected updates; no secure aggregation or differential privacy",
            },
            {
                "stage": "validation",
                "actual_behavior": "validation predictors and targets concatenated in the simulation",
                "limitation": "data locality applies to the training partition; validation is centrally accessible",
            },
            {
                "stage": "evaluation",
                "actual_behavior": "per-client predictions, targets and paired errors accessible to evaluator",
                "limitation": "predictive utility assessed; no attack evaluation or formal privacy guarantee",
            },
        ]
    )


def make_figures(
    output_dir: Path, dimensions: pd.DataFrame, counts: pd.DataFrame
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def save(fig, name):
        for suffix in ("png", "pdf", "svg"):
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if suffix == "pdf"
                else ({"Date": None} if suffix == "svg" else {})
            )
            fig.savefig(
                output_dir / f"{name}.{suffix}",
                dpi=180,
                bbox_inches="tight",
                metadata=metadata,
            )
        plt.close(fig)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "solarfl-manuscript-evidence",
        }
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set(xlim=(0, 12), ylim=(0, 8))
        ax.axis("off")

        def box(x, y, w, h, text, color="#e8eef5"):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.06",
                    facecolor=color,
                    edgecolor="#31536d",
                )
            )
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

        def arrow(start, end):
            ax.add_patch(
                FancyArrowPatch(
                    start, end, arrowstyle="-|>", mutation_scale=13, color="#31536d"
                )
            )

        box(
            2.2,
            6.7,
            7.6,
            0.85,
            "Public Mendeley V2 data · seven station clients\nVerified inverter membership and capacity denominators",
        )
        box(
            2.2,
            5.15,
            7.6,
            0.95,
            "Frozen chronological splits · hourly timestamp lookup at T−24\nA: 12 input columns · B: 17 columns with target-hour ERA5 reanalysis",
        )
        arrow((6, 6.7), (6, 6.1))
        box(
            0.25,
            2.55,
            5.3,
            1.65,
            "Exploratory validation\nPersistence, local / centralized Ridge and MLP\nFedAvg / FedProx · A and perfect-weather B\nBroad sweep: seed 0 · E = 1 and E = 5",
            "#fff1d6",
        )
        box(
            6.45,
            2.55,
            5.3,
            1.65,
            "Frozen confirmatory test\nCentralized MLP vs selected FedProx only\nVariant A · E = 1 · mu = 1\nPaired training seeds s = 0, 1, 2, 3, 4",
            "#def0e6",
        )
        arrow((4, 5.15), (2.9, 4.2))
        arrow((8, 5.15), (9.1, 4.2))
        arrow((5.55, 3.35), (6.45, 3.35))
        ax.text(
            6,
            4.45,
            "Select on validation; freeze before test",
            ha="center",
            fontsize=10,
        )
        box(
            0.25,
            0.55,
            5.3,
            1.15,
            "Validation-only robustness: A, seeds 0–4\nCentralized / FedAvg E1 / FedProx E1\nNo confirmatory local or perfect-weather claim",
            "#fff1d6",
        )
        box(
            6.45,
            0.55,
            5.3,
            1.15,
            "Paired macro-MAE gap: FedProx − centralized\nUnion-calendar day bootstrap · margin 0.005\nCalendar / margin sensitivity labeled post-hoc",
            "#def0e6",
        )
        arrow((2.9, 2.55), (2.9, 1.7))
        arrow((9.1, 2.55), (9.1, 1.7))
        ax.set_title("Experiment scope and evaluation flow", fontsize=16, pad=15)
        save(fig, "methodology_scope")

        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.axis("off")
        ax.set_title(
            "Shared MLP architecture — actual input width is retained",
            fontsize=14,
            pad=15,
        )
        for y, variant in ((0.72, "A"), (0.29, "B")):
            row = dimensions.query("variant == @variant and split == 'train'").iloc[0]
            label = f"Variant {variant}: {row.input_width} inputs\n{row.nonconstant_columns} nonconstant columns"
            ax.text(
                0.02,
                y,
                label,
                transform=ax.transAxes,
                va="center",
                bbox={"facecolor": "#e8eef5", "edgecolor": "#31536d", "pad": 10},
            )
            for x, text in (
                (0.45, "64\nReLU"),
                (0.68, "32\nReLU"),
                (0.89, "1\nLinear"),
            ):
                ax.text(
                    x,
                    y,
                    text,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "#def0e6", "edgecolor": "#31536d", "pad": 10},
                )
            for x0, x1 in ((0.31, 0.40), (0.50, 0.63), (0.73, 0.84)):
                ax.annotate(
                    "",
                    xy=(x1, y),
                    xytext=(x0, y),
                    xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->"},
                )
            ax.text(
                0.43,
                y - 0.16,
                f"{row.trainable_parameters:,} trainable parameters",
                transform=ax.transAxes,
            )
        ax.text(
            0.02,
            0.01,
            "is_daylight remains in both matrices; constant-column count does not establish matrix rank.",
            transform=ax.transAxes,
            fontsize=10,
        )
        save(fig, "feature_architecture")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
        for ax, split in zip(axes, ("train", "val", "test")):
            sub = counts.query("variant == 'A' and split == @split")
            x = np.arange(len(sub))
            ax.bar(x - 0.19, sub.raw_rows, 0.38, label="Raw rows", color="#aab9c8")
            ax.bar(
                x + 0.19,
                sub.eligible_rows,
                0.38,
                label="Model-eligible rows",
                color="#28776c",
            )
            ax.set_xticks(x, sub.client)
            ax.set_title(f"{split}: {sub.eligible_rows.sum():,} retained")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Observations")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Variant A: raw split counts and retained model observations", fontsize=14
        )
        fig.tight_layout()
        save(fig, "split_retention")


def run(
    output_dir: Path, raw_dir: Path, results_dir: Path, *, download: bool = False
) -> None:
    if output_dir.resolve() == results_dir.resolve():
        raise ValueError(
            "use a separate review output directory to preserve archived results"
        )
    # A source mismatch must fail before any manuscript evidence is emitted.
    capacities = audit_capacity(raw_dir, output_dir, download=download)
    counts, dimensions, columns = feature_and_split_audit()
    scope = experiment_scope(results_dir)
    totals = (
        counts.groupby(["variant", "split"], sort=False)[
            ["raw_rows", "eligible_rows", "excluded_rows"]
        ]
        .sum()
        .reset_index()
    )
    tables = {
        "station_capacity": capacities,
        "split_counts": counts,
        "split_totals": totals,
        "feature_dimensions": dimensions,
        "feature_columns": columns,
        "experiment_scope": scope,
        "optimization_protocol": optimization_protocol(),
        "data_locality_scope": privacy_scope(),
    }
    for name, frame in tables.items():
        write_table(frame, output_dir / name)
    compact_capacity = capacities[
        [
            "client",
            "rated_inverters",
            "retained_inverters",
            "excluded_rated_inverters",
            "all_rated_power_kw",
            "excluded_rated_power_kw",
            "capacity_kw",
        ]
    ].copy()
    compact_capacity.columns = [
        "Client",
        "Rated n",
        "Retained n",
        "Excluded n",
        "All kW",
        "Excluded kW",
        "Final kW",
    ]
    write_table(compact_capacity, output_dir / "manuscript_capacity")
    compact_dimensions = dimensions.loc[
        dimensions.split == "train",
        [
            "variant",
            "input_width",
            "nonconstant_columns",
            "trainable_parameters",
        ],
    ].copy()
    compact_dimensions.columns = [
        "Variant",
        "Input width",
        "Nonconstant columns",
        "Parameters",
    ]
    write_table(compact_dimensions, output_dir / "manuscript_architecture")
    compact_counts = totals.loc[
        totals.variant == "A",
        [
            "split",
            "raw_rows",
            "eligible_rows",
            "excluded_rows",
        ],
    ].copy()
    compact_counts.columns = ["Split", "Raw rows", "Eligible rows", "Excluded rows"]
    write_table(compact_counts, output_dir / "manuscript_split_totals")
    # Show the perfect-weather contrast at its actual evidence level; these
    # macro means derive from the rounded seed-0 client scores in the archive.
    base = pd.read_csv(results_dir / "baselines_val.csv")
    contrasts = []
    for regime in ("central", "local"):
        sub = base.loc[(base.model == "mlp") & (base.regime == regime)]
        means = sub.groupby("variant").mae.mean()
        contrasts.append(
            {
                "Regime": regime,
                "Split": "validation",
                "Seed": 0,
                "A MAE": means["past"],
                "B MAE": means["perfect"],
                "B minus A": means["perfect"] - means["past"],
            }
        )
    federated = pd.read_csv(results_dir / "fedprox_val.csv")
    for regime in ("fedavg", "fedprox"):
        sub = federated.loc[
            (federated.regime == regime) & (federated.local_epochs == 1)
        ]
        means = sub.groupby("variant").mae.mean()
        contrasts.append(
            {
                "Regime": f"{regime} E1",
                "Split": "validation",
                "Seed": 0,
                "A MAE": means["past"],
                "B MAE": means["perfect"],
                "B minus A": means["perfect"] - means["past"],
            }
        )
    write_table(pd.DataFrame(contrasts), output_dir / "manuscript_perfect_weather")
    make_figures(output_dir, dimensions, counts)
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
    outputs = sorted(
        p for p in output_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in outputs)
    )
    print(f"Review assets written to {output_dir}")
    print(totals.to_string(index=False))
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
