"""Feature builder: one PV client CSV -> (X, y) for day-ahead capacity-factor forecasting.

Column set and semantics come from configs/features.yaml (D-014 for the
weather_past/weather_future split, D-012 for the lag >= 24 requirement).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd
import yaml

from solarfl.labels.capacity import add_capacity_factor, station_labels

ROOT = Path(__file__).resolve().parents[3]
CLIENTS_DIR = ROOT / "data/processed/client"
SPLITS_PATH = ROOT / "configs/splits.json"  # the frozen manifest (hand-maintained, D-003)
SPEC_PATH = ROOT / "configs/features.yaml"  # the spec, not a suggestion

LAG = pd.Timedelta(hours=24)  # H = 24 (D-012) -- every lag feature must use lag >= 24

# The model_matrix groups this builder knows how to materialise, mapped to the
# prefix each group's columns carry in X. Iteration order fixes column order in
# X, so it must stay stable -- a fitted model indexes by position.
_GROUP_PREFIX = {
    "history": "history_",
    "geometry": "",
    "weather_past": "weather_past_",
    "weather_future": "weather_future_",
}


def _load_matrix(groups: dict[str, list[str]]) -> list[str]:
    """Flatten the yaml's model_matrix groups into prefixed column names.

    An unknown group is an error rather than a no-op: silently ignoring a group
    someone added to the yaml is exactly the spec/code drift that reading the
    yaml is meant to prevent.
    """
    unknown = sorted(set(groups) - set(_GROUP_PREFIX))
    if unknown:
        raise ValueError(
            f"configs/features.yaml declares model_matrix group(s) {unknown}, "
            f"which this builder cannot construct. Known groups: "
            f"{sorted(_GROUP_PREFIX)}."
        )
    return [
        f"{prefix}{col}"
        for group, prefix in _GROUP_PREFIX.items()
        for col in groups.get(group, [])
    ]


# features.yaml is the spec (CLAUDE.md), so the column set is read from it rather
# than restated here -- a hardcoded copy is a second source of truth that drifts
# silently the first time the yaml is edited.
_SPEC = yaml.safe_load(SPEC_PATH.read_text())
_GROUPS = _SPEC["model_matrix"]

TARGET = _SPEC["target"]
HISTORY_COLS = _GROUPS.get("history", [])
GEOMETRY_COLS = _GROUPS.get("geometry", [])
WEATHER_PAST_COLS = _GROUPS.get("weather_past", [])
# kept separate from weather_past even though the two lists are identical today:
# the yaml states them separately, and D-014 is the claim that they *could* differ
WEATHER_FUTURE_COLS = _GROUPS.get("weather_future", [])
MODEL_MATRIX = _load_matrix(_GROUPS)
PAST_MODEL_MATRIX = [
    col for col in MODEL_MATRIX if not col.startswith("weather_future_")
]
RowSet = Literal["past", "common"]


def _select_eligible_rows(df: pd.DataFrame, row_set: RowSet) -> pd.DataFrame:
    """Apply the declared daylight and feature-availability row policy.

    ``past`` represents the operational feature set and therefore cannot let
    target-hour reanalysis availability decide which rows survive. ``common``
    deliberately requires both variants so validation comparisons stay paired.
    Daylight is enforced directly rather than indirectly through target-hour
    ``kt``, which is undefined when top-of-atmosphere radiation is <= 10 W/m2.
    """
    if row_set == "past":
        required = PAST_MODEL_MATRIX + [TARGET]
    elif row_set == "common":
        required = MODEL_MATRIX + [TARGET]
    else:
        raise ValueError(
            f"unknown row_set: {row_set!r}; expected 'past' or 'common'"
        )

    daylight = df["is_daylight"].eq(1.0)
    return df.loc[daylight].dropna(subset=required)


def _lag_lookup(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Values of `cols` at T-24 for every row's T, aligned by timestamp.

    Row position can't be used for the shift -- the hourly grid has holes
    (06_timeseries-eda), so "24 rows back" and "24 hours back" disagree.
    reindex against measured_ts - LAG looks up the correct timestamp directly
    and returns NaN where that timestamp has no row, rather than silently
    grabbing the wrong hour.
    """
    by_ts = df.set_index("measured_ts")[cols]
    return by_ts.reindex(df["measured_ts"] - LAG).set_axis(df.index)


def _split_mask(ts: pd.Series, station_id: str, split: str) -> pd.Series:
    """Boolean mask assigning rows to a frozen split, verified against the manifest.

    Boundaries are the half-open timestamp intervals from configs/splits.json
    (same convention as splits._assign): [.., train_end) [train_end, val_end)
    [val_end, ..]. Counts are checked against the manifest so that a changed
    source file fails loudly instead of silently shifting the evaluation set.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split!r}")

    entry = json.loads(SPLITS_PATH.read_text())["clients"][station_id]
    if len(ts) != entry["n_total"]:
        raise ValueError(
            f"{station_id}: {len(ts)} rows, manifest says {entry['n_total']}. "
            "The source data changed — the frozen splits no longer apply."
        )

    train_end = pd.Timestamp(entry["train_end"])
    val_end = pd.Timestamp(entry["val_end"])
    mask = {
        "train": ts < train_end,
        "val": (ts >= train_end) & (ts < val_end),
        "test": ts >= val_end,
    }[split]

    if int(mask.sum()) != entry["counts"][split]:
        raise ValueError(
            f"{station_id}/{split}: mask selects {int(mask.sum())} rows, "
            f"manifest says {entry['counts'][split]}."
        )
    return mask


@overload
def build_features(
    station_id: str,
    split: str | None = None,
    *,
    row_set: RowSet,
    return_timestamps: Literal[False] = False,
) -> tuple[pd.DataFrame, pd.Series]: ...


@overload
def build_features(
    station_id: str,
    split: str | None = None,
    *,
    row_set: RowSet,
    return_timestamps: Literal[True],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]: ...


def build_features(
    station_id: str,
    split: str | None = None,
    *,
    row_set: RowSet,
    return_timestamps: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build an aligned model matrix and target for one client.

    ``row_set='past'`` filters on operational predictors only, whereas
    ``row_set='common'`` also requires target-hour reanalysis for paired
    past/perfect-weather comparisons. Evaluation code that needs temporal
    grouping can request ``(X, y, timestamps)``. Timestamps are taken after all
    filtering, so row ``i`` refers to the same observation in each object.
    """
    labels = station_labels()
    capacity_kw = labels[station_id]["capacity_kw"]

    df = pd.read_csv(CLIENTS_DIR / f"{station_id}.csv", parse_dates=["measured_ts"])
    df = add_capacity_factor(df, capacity_kw)
    df = df.sort_values("measured_ts").reset_index(drop=True)

    # geometry -- TOA is terrestrial_radiation; E0 is the earth-sun distance
    # correction, so toa / (solar_constant * E0) isolates cos(zenith)
    doy = df["measured_ts"].dt.dayofyear
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    df["cos_zenith"] = (df["terrestrial_radiation"] / (1361 * e0)).clip(0, 1)

    df["kt"] = np.where(
        df["terrestrial_radiation"] > 10,
        df["shortwave_radiation"] / df["terrestrial_radiation"],
        np.nan,
    )
    df["is_daylight"] = (df["terrestrial_radiation"] > 10).astype(float)

    hour = df["measured_ts"].dt.hour + df["measured_ts"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Every base column the yaml names must exist by now, before we prefix any of
    # them. Checking here rather than at selection turns "the yaml asks for a
    # column this builder never derives" into a named error instead of a KeyError
    # from inside a loop. Code conforms to the yaml, so this failing means the
    # code is behind the spec.
    base = [
        TARGET,
        *HISTORY_COLS,
        *GEOMETRY_COLS,
        *WEATHER_PAST_COLS,
        *WEATHER_FUTURE_COLS,
    ]
    missing = [c for c in dict.fromkeys(base) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{station_id}: configs/features.yaml names {missing}, which this "
            "builder does not construct — either a derivation is missing from "
            "build_features or the column is absent from the source CSV."
        )

    # history: lag >= 24 (D-012)
    hist = _lag_lookup(df, HISTORY_COLS)
    for c in HISTORY_COLS:
        df[f"history_{c}"] = hist[c]

    # weather_past: observed at T-24, available at forecast time (D-014)
    past = _lag_lookup(df, WEATHER_PAST_COLS)
    for c in WEATHER_PAST_COLS:
        df[f"weather_past_{c}"] = past[c]

    # weather_future: reanalysis at T itself -- perfect-forecast assumption (D-014)
    for c in WEATHER_FUTURE_COLS:
        df[f"weather_future_{c}"] = df[c]

    # split filter comes BEFORE dropna: lag lookups were already computed on the
    # full timeline above, so a val/test row keeps its T-24 value even when that
    # timestamp falls in an earlier split — that's past information, not leakage
    if split is not None:
        df = df[_split_mask(df["measured_ts"], station_id, split)]

    df = _select_eligible_rows(df, row_set)

    X = df[MODEL_MATRIX].reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)
    if return_timestamps:
        timestamps = df["measured_ts"].reset_index(drop=True)
        return X, y, timestamps
    return X, y
