"""Per-client temporal train/val/test splits.

Each of the 7 stations is a federated client and is split on its OWN timeline:
first 70% of its observations -> train, next 15% -> val, last 15% -> test.

Two properties this module guarantees:
  1. Deterministic. No shuffling, no seed. Same input -> same split, always.
  2. Frozen. Cut points are written to splits.json and committed to git. Row
     counts are stored alongside as a checksum, so if the source data ever
     changes the loader fails loudly instead of silently comparing runs that
     were evaluated on different test sets.

Split boundaries are computed on the UNFILTERED timeline. Filtering (night
hours, sensor-noise rows) happens inside each split afterwards, so changing a
filter can never move a boundary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CLIENTS_DIR = ROOT / "data/processed/client"
SPLITS_PATH = ROOT / "data/processed/splits.json"
RAW_FACT = ROOT / "data/raw"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test is the remainder — never specified directly, so the fractions can't drift apart

KEY = "station_hash_id"
TS = "measured_ts"



def client_ids(clients_dir: Path = CLIENTS_DIR) -> list[str]:
    return sorted(p.stem for p in clients_dir.glob("*.csv"))


def load_client(station_id: str, clients_dir: Path = CLIENTS_DIR) -> pd.DataFrame:
    """Load exactly one client. Deliberately loads one — never a list."""
    return pd.read_csv(clients_dir / f"{station_id}.csv")


# --------------------------------------------------------------------------
# 2. Compute and freeze splits
# --------------------------------------------------------------------------

def _cut_points(ts: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the two boundary timestamps by ROW POSITION, not by calendar span.

    Coverage is uneven (the paper reports ~21% inter-day missingness on every
    device). A calendar-based cut would hand training a span that is mostly
    gaps; a position-based cut guarantees the proportion of real observations.
    """
    ts_sorted = ts.reset_index(drop=True)
    n = len(ts_sorted)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return ts_sorted.iloc[i_train], ts_sorted.iloc[i_val]


def _assign(ts: pd.Series, train_end, val_end) -> pd.Series:
    """Half-open intervals: [.., train_end) [train_end, val_end) [val_end, ..].

    Boundaries are timestamps rather than row indices so that duplicate
    timestamps, if any exist, land wholly on one side instead of straddling.
    """
    ts = pd.to_datetime(ts)
    return pd.Series(
        pd.cut(ts, bins=[ts.min(), train_end, val_end, ts.max() + pd.Timedelta("1s")],
               labels=["train", "val", "test"], right=False, include_lowest=True),
        index=ts.index,
    )


def freeze_splits(clients_dir: Path = CLIENTS_DIR,
                  out_path: Path = SPLITS_PATH) -> dict:
    """Compute the cut points once and write them to disk."""
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(RAW_FACT),
        "ratios": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                   "test": round(1 - TRAIN_FRAC - VAL_FRAC, 4)},
        "method": "per-client temporal, cut by row position",
        "clients": {},
    }


    for sid in client_ids(clients_dir):
        df = load_client(sid, clients_dir)
        df[TS] = pd.to_datetime(df[TS])
        train_end, val_end = _cut_points(df[TS])
        part = _assign(df[TS], train_end, val_end)

        entry = {
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
            "n_total": len(df),
            "counts": {k: int((part == k).sum()) for k in ("train", "val", "test")},
            "ranges": {},
        }
        for k in ("train", "val", "test"):
            sub = df.loc[part == k, TS]
            entry["ranges"][k] = [sub.min().isoformat(), sub.max().isoformat()]
        manifest["clients"][sid] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------
# 3. Load and verify
# --------------------------------------------------------------------------

def load_split(station_id: str, which: str,
               clients_dir: Path = CLIENTS_DIR,
               splits_path: Path = SPLITS_PATH) -> pd.DataFrame:
    """Load one split of one client, verifying it against the frozen manifest."""
    assert which in ("train", "val", "test"), f"unknown split: {which}"

    manifest = json.loads(splits_path.read_text())
    entry = manifest["clients"][station_id]
    train_end = pd.Timestamp(entry["train_end"])
    val_end = pd.Timestamp(entry["val_end"])

    df = load_client(station_id, clients_dir)
    part = _assign(df[TS], train_end, val_end)
    out = df.loc[part == which].reset_index(drop=True)

    expected = entry["counts"][which]
    if len(out) != expected:
        raise ValueError(
            f"{station_id}/{which}: got {len(out)} rows, manifest says {expected}. "
            "The source data changed — earlier results are no longer comparable."
        )
    return out


def verify(splits_path: Path = SPLITS_PATH, clients_dir: Path = CLIENTS_DIR) -> None:
    """Three assertions that make 'frozen' mean something."""
    manifest = json.loads(splits_path.read_text())

    for sid in manifest["clients"]:
        tr = load_split(sid, "train", clients_dir, splits_path)
        va = load_split(sid, "val", clients_dir, splits_path)
        te = load_split(sid, "test", clients_dir, splits_path)

        # 1. every split is usable
        for name, part in (("train", tr), ("val", va), ("test", te)):
            assert len(part) > 0, f"{sid}: {name} split is empty"

        # 2. strict chronological order — no future leaking into the past
        assert tr[TS].max() < va[TS].min(), f"{sid}: train overlaps val"
        assert va[TS].max() < te[TS].min(), f"{sid}: val overlaps test"

        # 3. nothing lost or double-counted
        total = manifest["clients"][sid]["n_total"]
        assert len(tr) + len(va) + len(te) == total, f"{sid}: rows lost in split"

    print(f"verified {len(manifest['clients'])} clients")


def summary(splits_path: Path = SPLITS_PATH) -> pd.DataFrame:
    """Human-readable table of what was frozen. Paste this into docs/framing.md."""
    manifest = json.loads(splits_path.read_text())
    rows = []
    for sid, e in manifest["clients"].items():
        rows.append({
            "station": sid[:8],
            "n_total": e["n_total"],
            "train": e["counts"]["train"],
            "val": e["counts"]["val"],
            "test": e["counts"]["test"],
            "data_start": e["ranges"]["train"][0][:10],
            "train_end": e["train_end"][:10],
            "val_end": e["val_end"][:10],
            "data_end": e["ranges"]["test"][1][:10],
        })
    return pd.DataFrame(rows).sort_values("n_total", ascending=False)


if __name__ == "__main__":
    freeze_splits()
    verify()
    print()
    print(summary().to_string(index=False))
