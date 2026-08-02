"""Feature builder: one PV client CSV -> (X, y) for day-ahead capacity-factor forecasting.

Column set and semantics come from configs/features.yaml (D-014 for the
weather_past/weather_future split, D-012 for the lag >= 24 requirement).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarfl.labels.capacity import add_capacity_factor, station_labels

ROOT = Path(__file__).resolve().parents[3]
CLIENTS_DIR = ROOT / "data/processed/client"
SPLITS_PATH = ROOT / "configs/splits.json"  # the frozen manifest (hand-maintained, D-003)

LAG = pd.Timedelta(hours=24)  # H = 24 (D-012) -- every lag feature must use lag >= 24

WEATHER_COLS = [
    "temperature_2m",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "kt",
]

MODEL_MATRIX = (
    ["history_capacity_factor"]
    + ["cos_zenith", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_daylight"]
    + [f"weather_past_{c}" for c in WEATHER_COLS]
    + [f"weather_future_{c}" for c in WEATHER_COLS]
)


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


def build_features(
    station_id: str, split: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
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

    # history: capacity_factor lag >= 24 (D-012)
    df["history_capacity_factor"] = _lag_lookup(df, ["capacity_factor"])["capacity_factor"]

    # weather_past: observed at T-24, available at forecast time (D-014)
    past = _lag_lookup(df, WEATHER_COLS)
    for c in WEATHER_COLS:
        df[f"weather_past_{c}"] = past[c]

    # weather_future: reanalysis at T itself -- perfect-forecast assumption (D-014)
    for c in WEATHER_COLS:
        df[f"weather_future_{c}"] = df[c]

    # split filter comes BEFORE dropna: lag lookups were already computed on the
    # full timeline above, so a val/test row keeps its T-24 value even when that
    # timestamp falls in an earlier split — that's past information, not leakage
    if split is not None:
        df = df[_split_mask(df["measured_ts"], station_id, split)]

    required = MODEL_MATRIX + ["capacity_factor"]
    df = df.dropna(subset=required)

    X = df[MODEL_MATRIX].reset_index(drop=True)
    y = df["capacity_factor"].reset_index(drop=True)
    return X, y
