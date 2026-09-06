"""Target variable: capacity factor.

capacity_factor = hourly energy (kWh) / (station rated power (kW) * 1 hour)

The source labels UTC intervals [T, T+1 hour) by their start T. Denominators
sum metadata max_power for rated devices present in the full hourly inverter
fact table (D-009), not verified commissioned capacity. Reconstruct and verify
them with ``python -m solarfl.labels.capacity_audit``.

Values above 1.0 are impossible - an inverter cannot beat its own rating for
a full hour. Those are blanked to NaN, not deleted, because splits.json checks
row counts and deleting rows would break it.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LABELS_PATH = ROOT / "configs/station_labels.json"


def station_labels(path: Path = LABELS_PATH) -> dict:
    return json.loads(path.read_text())


def add_capacity_factor(df: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    cf = df["total_produced_energy"] / capacity_kw
    df["capacity_factor"] = cf.where(cf <= 1.0)
    return df
