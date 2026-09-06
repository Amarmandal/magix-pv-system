"""Download and deterministically partition the archived station-level data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/raw/hourly_pv_weather_station.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/client"
DEFAULT_MANIFEST = ROOT / "configs/splits.json"

SOURCE_URL = (
    "https://data.mendeley.com/public-files/datasets/4zgsckxpdy/files/"
    "fac7b49d-96d8-4a4a-8d9b-4aa008683339/file_downloaded"
)
SOURCE_SHA256 = "fd504ae8ec39fad64f3b08adb546d2158c8f94f567bca03752616eb5555ddbd5"
KEY = "station_hash_id"
TIMESTAMP = "measured_ts"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(destination: Path) -> None:
    """Download the immutable Mendeley V2 file and verify it before promotion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "SolarFL/1.0 reproducibility script"},
    )
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        observed = sha256_file(partial)
        if observed != SOURCE_SHA256:
            raise ValueError(
                "downloaded source checksum differs from Mendeley V2: "
                f"expected {SOURCE_SHA256}, got {observed}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def partition_frame(frame: pd.DataFrame, manifest: dict) -> dict[str, pd.DataFrame]:
    """Validate and split the source frame using the frozen client manifest."""
    missing = {KEY, TIMESTAMP} - set(frame.columns)
    if missing:
        raise ValueError(f"source CSV is missing required columns: {sorted(missing)}")
    if frame.duplicated([KEY, TIMESTAMP]).any():
        raise ValueError("source CSV contains duplicate station/timestamp rows")

    clients = manifest["clients"]
    observed_ids = set(frame[KEY].astype(str).unique())
    expected_ids = set(clients)
    if observed_ids != expected_ids:
        raise ValueError(
            "source station IDs differ from configs/splits.json: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )

    partitions = {}
    for station_id in sorted(clients):
        part = frame.loc[frame[KEY].astype(str) == station_id]
        part = part.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True)
        expected_rows = clients[station_id]["n_total"]
        if len(part) != expected_rows:
            raise ValueError(
                f"{station_id}: source has {len(part)} rows; "
                f"frozen manifest requires {expected_rows}"
            )
        partitions[station_id] = part
    return partitions


def prepare(
    source: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> list[tuple[Path, str]]:
    """Verify the source and create or verify all client CSVs."""
    observed = sha256_file(source)
    if observed != SOURCE_SHA256:
        raise ValueError(
            "source checksum differs from the archived Mendeley V2 file: "
            f"expected {SOURCE_SHA256}, got {observed}"
        )

    frame = pd.read_csv(source)
    manifest = json.loads(manifest_path.read_text())
    partitions = partition_frame(frame, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    for station_id, part in partitions.items():
        destination = output_dir / f"{station_id}.csv"
        payload = part.to_csv(index=False)
        if destination.exists():
            current = destination.read_text()
            if current == payload:
                outcomes.append((destination, "verified"))
                continue
            if not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite different client data: {destination}; "
                    "inspect it or rerun with --overwrite"
                )
        destination.write_text(payload)
        outcomes.append((destination, "written"))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the exact Mendeley V2 source when --source is absent",
    )
    parser.add_argument(
        "--audit-capacity",
        action="store_true",
        help="verify/download capacity sources and independently check frozen labels",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace client CSVs that differ from the verified source partition",
    )
    args = parser.parse_args()

    if args.audit_capacity:
        from solarfl.labels.capacity_audit import run as audit_capacity

        audit_capacity(
            args.source.parent,
            ROOT / "reproduced-results/manuscript_evidence_audit",
            download=args.download,
        )

    if not args.source.exists():
        if not args.download:
            parser.error(f"source file does not exist: {args.source}")
        print(f"downloading Mendeley V2 source -> {args.source}", flush=True)
        download_source(args.source)

    outcomes = prepare(
        args.source,
        args.output_dir,
        args.manifest,
        overwrite=args.overwrite,
    )
    for path, outcome in outcomes:
        print(f"{outcome}: {path}")
    print(f"prepared {len(outcomes)} frozen clients")


if __name__ == "__main__":
    main()
