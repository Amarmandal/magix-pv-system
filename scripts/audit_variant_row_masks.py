"""Compare operational and common-mask row sets without training models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from solarfl.data.features import PAST_MODEL_MATRIX, ROOT, build_features
from solarfl.data.splits import client_ids
from solarfl.labels.capacity import station_labels

DEFAULT_OUTPUT = ROOT / "results/variant_row_mask_audit.csv"


def audit() -> pd.DataFrame:
    """Report whether future-feature availability removes operational rows."""
    labels = station_labels()
    rows = []
    ordered_clients = sorted(client_ids(), key=lambda sid: labels[sid]["label"])
    for sid in ordered_clients:
        for split in ("train", "val", "test"):
            past_X, past_y, past_ts = build_features(
                sid, split=split, row_set="past", return_timestamps=True
            )
            common_X, common_y, common_ts = build_features(
                sid, split=split, row_set="common", return_timestamps=True
            )
            past_index = pd.Index(past_ts)
            common_index = pd.Index(common_ts)
            rows.append({
                "client": labels[sid]["label"],
                "split": split,
                "past_rows": len(past_index),
                "common_rows": len(common_index),
                "past_only_rows": len(past_index.difference(common_index)),
                "common_only_rows": len(common_index.difference(past_index)),
                "identical_timestamps": past_index.equals(common_index),
                "identical_past_inputs": past_X[PAST_MODEL_MATRIX].equals(
                    common_X[PAST_MODEL_MATRIX]
                ),
                "identical_targets": past_y.equals(common_y),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\naudit -> {args.output}")


if __name__ == "__main__":
    main()
