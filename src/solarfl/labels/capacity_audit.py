"""Reconstruct D-009 from versioned metadata and observed inverter membership.

No model is trained and no frozen configuration is rewritten. 'Producing' in
the historical notebook means present in the full hourly inverter fact table,
including zero-energy observations; it is not a commissioning classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = ROOT / "docs/source_manifest.json"
REQUIRED_FILES = ("devices.csv", "stations.csv", "hourly_pv_weather_inverter.csv")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_sources(raw_dir: Path, *, download: bool = False) -> dict:
    """Verify every capacity input; download absent files only, fail on drift."""
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for name in REQUIRED_FILES:
        entry = manifest["files"][name]
        path = raw_dir / name
        if not path.exists() and download:
            raw_dir.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".csv.part")
            try:
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "SolarFL capacity audit"}
                )
                with urllib.request.urlopen(request) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(partial) != entry["sha256"]:
                    raise ValueError(f"download checksum mismatch: {name}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; use --download to acquire V2 inputs"
            )
        if (
            path.stat().st_size != entry["size_bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"source checksum/size mismatch: {path}")
    return manifest


def derive_capacities(
    devices: pd.DataFrame, stations: pd.DataFrame, observations: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return independently derived labels, a device ledger, and station totals.

    Membership is assessed over ALL supplied observations, reproducing the
    notebook's retrospective denominator. Capacity ties use the station hash
    as a deterministic secondary key, matching the frozen S1--S7 mapping.
    """
    for name, frame, required in (
        (
            "devices",
            devices,
            {"device_hash_id", "station_hash_id", "max_power", "device_model"},
        ),
        ("stations", stations, {"station_hash_id"}),
        ("observations", observations, {"device_hash_id", "measured_ts"}),
    ):
        if required - set(frame):
            raise ValueError(f"{name}: missing columns {sorted(required - set(frame))}")
    for frame, key in ((devices, "device_hash_id"), (stations, "station_hash_id")):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"null or duplicate {key}")
    if devices["station_hash_id"].isna().any():
        raise ValueError("device with missing station")
    if not set(devices.station_hash_id) <= set(stations.station_hash_id):
        raise ValueError("device references unknown station")
    if (
        observations.empty
        or observations[["device_hash_id", "measured_ts"]].isna().any().any()
    ):
        raise ValueError("empty observations or missing device/timestamp")
    if not set(observations.device_hash_id) <= set(devices.device_hash_id):
        raise ValueError("observations reference unknown device")
    if observations.duplicated(["device_hash_id", "measured_ts"]).any():
        raise ValueError("duplicate device/timestamp observations")

    ledger = devices[
        ["station_hash_id", "device_hash_id", "device_model", "max_power"]
    ].copy()
    ledger["max_power"] = pd.to_numeric(ledger.max_power, errors="raise")
    rated = ledger.max_power.notna()
    if not (
        np.isfinite(ledger.loc[rated, "max_power"])
        & (ledger.loc[rated, "max_power"] > 0)
    ).all():
        raise ValueError("rated power must be finite and positive")
    obs = observations.copy()
    obs["measured_ts"] = pd.to_datetime(obs.measured_ts, errors="raise")
    membership = obs.groupby("device_hash_id").agg(
        observation_rows=("measured_ts", "size"),
        first_observation=("measured_ts", "min"),
        last_observation=("measured_ts", "max"),
    )
    ledger = ledger.join(membership, on="device_hash_id", validate="one_to_one")
    ledger["observation_rows"] = ledger.observation_rows.fillna(0).astype(int)
    present = ledger.observation_rows > 0
    if (present & ~rated).any():
        raise ValueError("observed device has no rated power")
    ledger["retained"] = rated & present
    ledger["reason"] = np.select(
        [~rated, ~present],
        ["no_rated_power_metadata", "absent_from_full_hourly_fact"],
        default="rated_and_observed_in_full_hourly_fact",
    )
    ledger["included_capacity_kw"] = ledger.max_power.where(ledger.retained, 0.0)

    rows = []
    for sid in sorted(stations.station_hash_id):
        group = ledger.loc[ledger.station_hash_id == sid]
        is_rated = group.max_power.notna()
        rows.append(
            {
                "station_hash_id": sid,
                "metadata_devices": len(group),
                "rated_inverters": int(is_rated.sum()),
                "retained_inverters": int(group.retained.sum()),
                "excluded_rated_inverters": int((is_rated & ~group.retained).sum()),
                "unrated_devices": int((~is_rated).sum()),
                "all_rated_power_kw": float(group.max_power.sum()),
                "excluded_rated_power_kw": float(
                    group.loc[is_rated & ~group.retained, "max_power"].sum()
                ),
                "capacity_kw": float(group.included_capacity_kw.sum()),
                "exclusion_rule": "unrated metadata or absent from full hourly inverter fact",
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(["capacity_kw", "station_hash_id"])
        .reset_index(drop=True)
    )
    if (table.capacity_kw <= 0).any():
        raise ValueError("station without positive retained capacity")
    table.insert(0, "client", [f"S{i + 1}" for i in range(len(table))])
    labels = {
        r.station_hash_id: {"label": r.client, "capacity_kw": r.capacity_kw}
        for r in table.itertuples()
    }
    ledger.insert(
        0, "client", ledger.station_hash_id.map(lambda sid: labels[sid]["label"])
    )
    ledger = ledger.sort_values(["client", "device_hash_id"]).reset_index(drop=True)
    return labels, ledger, table


def run(raw_dir: Path, output_dir: Path, *, download: bool = False) -> pd.DataFrame:
    manifest = verified_sources(raw_dir, download=download)
    observations = pd.read_csv(
        raw_dir / REQUIRED_FILES[2], usecols=["device_hash_id", "measured_ts"]
    )
    labels, ledger, table = derive_capacities(
        pd.read_csv(raw_dir / "devices.csv"),
        pd.read_csv(raw_dir / "stations.csv"),
        observations,
    )
    frozen_path = ROOT / "configs/station_labels.json"
    if labels != json.loads(frozen_path.read_text()):
        raise ValueError("reconstructed labels differ from frozen station_labels.json")
    if output_dir.resolve() == frozen_path.parent.resolve():
        raise ValueError("audit output must not be the frozen configs directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_dir / "capacity_device_audit.csv", index=False)
    table.to_csv(output_dir / "station_capacity.csv", index=False)
    provenance = {
        "source_manifest": manifest,
        "frozen_labels_sha256": sha256(frozen_path),
        "matches_frozen_labels": True,
        "membership_rule": "rated device ID occurs at least once; energy sign not consulted",
        "membership_scope": "full released hourly inverter fact, not training-only",
        "observation_start_utc": observations.measured_ts.min(),
        "observation_end_utc": observations.measured_ts.max(),
        "commissioning_status_verified": False,
    }
    (output_dir / "capacity_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduced-results/manuscript_evidence_audit",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(
        run(args.raw_dir, args.output_dir, download=args.download).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
