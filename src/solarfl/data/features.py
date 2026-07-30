"""Feature builder: one PV client CSV -> (X, y) for day-ahead capacity-factor forecasting.

Column set and semantics come from configs/features.yaml (D-014 for the
weather_past/weather_future split, D-012 for the lag >= 24 requirement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from solarfl.labels.capacity import add_capacity_factor, station_labels

ROOT = Path(__file__).resolve().parents[3]
CLIENTS_DIR = ROOT / "data/processed/client"

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


def build_features(station_id: str) -> tuple[pd.DataFrame, pd.Series]:
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

    required = MODEL_MATRIX + ["capacity_factor"]
    df = df.dropna(subset=required)

    X = df[MODEL_MATRIX].reset_index(drop=True)
    y = df["capacity_factor"].reset_index(drop=True)
    return X, y
