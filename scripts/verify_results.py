"""Compare regenerated CSV results with the frozen archival tables."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "results"
DEFAULT_CANDIDATE = ROOT / "reproduced-results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_reference_checksums(reference_dir: Path) -> None:
    """Fail if any committed result differs from its archival checksum."""
    checksum_path = reference_dir / "SHA256SUMS"
    for line in checksum_path.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = reference_dir / filename.strip()
        observed = _sha256(path)
        if observed != expected:
            raise AssertionError(
                f"frozen result checksum mismatch for {path}: "
                f"expected {expected}, got {observed}"
            )


def compare_csv(reference: Path, candidate: Path, tolerance: float) -> None:
    """Compare schema, ordering, labels, and numeric values within tolerance."""
    expected = pd.read_csv(reference)
    observed = pd.read_csv(candidate)
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=tolerance,
        atol=tolerance,
        obj=candidate.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    verify_reference_checksums(args.reference)
    filenames = [
        line.split(maxsplit=1)[1].strip()
        for line in (args.reference / "SHA256SUMS").read_text().splitlines()
    ]
    missing = [name for name in filenames if not (args.candidate / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"candidate directory is missing result tables: {missing}"
        )

    for filename in filenames:
        compare_csv(
            args.reference / filename,
            args.candidate / filename,
            args.tolerance,
        )
        print(f"matches: {filename}")
    print(f"verified {len(filenames)} regenerated result tables")


if __name__ == "__main__":
    main()
