"""Generate a post-hoc sensitivity table for non-inferiority margins."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from solarfl.data.features import ROOT

MARGINS = (0.0, 0.0025, 0.005, 0.0075, 0.01)
SUMMARY_PATH = ROOT / "results/test_evaluation_summary.csv"
OUTPUT_PATH = ROOT / "results/noninferiority_margin_sensitivity.csv"


def build_margin_sensitivity(
    summary: pd.DataFrame,
    margins: Iterable[float] = MARGINS,
) -> pd.DataFrame:
    """Apply the frozen CI upper bound to alternative reporting margins."""
    required = {"ci_upper", "noninferiority_margin"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")
    if len(summary) != 1:
        raise ValueError("summary must contain exactly one primary-result row")

    ci_upper = float(summary.iloc[0]["ci_upper"])
    primary_margin = float(summary.iloc[0]["noninferiority_margin"])
    rows = []
    for margin in margins:
        margin = float(margin)
        if margin < 0:
            raise ValueError("non-inferiority margins must be non-negative")
        passes = ci_upper < margin
        rows.append({
            "margin": margin,
            "capacity_factor_percentage_points": 100 * margin,
            "ci_upper": ci_upper,
            "noninferiority_passes": passes,
            "role": (
                "pre-specified primary"
                if math.isclose(margin, primary_margin)
                else "post-hoc sensitivity"
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    result = build_margin_sensitivity(pd.read_csv(args.summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nsensitivity table -> {args.output}")


if __name__ == "__main__":
    main()
