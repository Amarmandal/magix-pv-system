"""Fast checks for release metadata and frozen reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.prepare_data import partition_frame
from scripts.verify_results import verify_reference_checksums
from solarfl.eval.test_evaluation import _bootstrap_macro_gaps

ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactTests(unittest.TestCase):
    def test_release_metadata_parses(self) -> None:
        citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
        zenodo = json.loads((ROOT / ".zenodo.json").read_text())
        self.assertEqual(citation["version"], "1.0.0")
        self.assertEqual(citation["license"], "MIT")
        self.assertEqual(len(citation["authors"]), 2)
        self.assertEqual(zenodo["license"], "MIT")
        self.assertEqual(len(zenodo["creators"]), 2)

    def test_frozen_result_checksums(self) -> None:
        verify_reference_checksums(ROOT / "results")

    def test_partition_uses_manifest_ids_counts_and_timestamp_order(self) -> None:
        frame = pd.DataFrame(
            {
                "station_hash_id": ["b", "a", "b", "a"],
                "measured_ts": [
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-01",
                    "2026-01-01",
                ],
                "value": [4, 2, 3, 1],
            }
        )
        manifest = {"clients": {"a": {"n_total": 2}, "b": {"n_total": 2}}}
        parts = partition_frame(frame, manifest)
        self.assertEqual(list(parts), ["a", "b"])
        self.assertEqual(parts["a"]["value"].tolist(), [1, 2])
        self.assertEqual(parts["b"]["value"].tolist(), [3, 4])

    def test_bootstrap_is_reproducible(self) -> None:
        rows = []
        for day in pd.date_range("2026-01-01", periods=4):
            for seed in (0, 1):
                for client in ("S1", "S2"):
                    rows.append(
                        {
                            "calendar_day": day,
                            "client": client,
                            "seed": seed,
                            "central_abs_error": 0.2,
                            "fedprox_abs_error": 0.1,
                        }
                    )
        errors = pd.DataFrame(rows)
        first = _bootstrap_macro_gaps(errors, 20, random_seed=7)
        second = _bootstrap_macro_gaps(errors, 20, random_seed=7)
        first_hash = hashlib.sha256(first.tobytes()).digest()
        second_hash = hashlib.sha256(second.tobytes()).digest()
        self.assertEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
