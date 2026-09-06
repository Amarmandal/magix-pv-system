"""Guard capacity provenance and the architecture/seed/optimizer disclosures."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from solarfl.data.features import MODEL_MATRIX, PAST_MODEL_MATRIX, ROOT, build_features
from solarfl.eval.manuscript_assets import experiment_scope
from solarfl.federated.fedprox import fit_fedprox
from solarfl.labels.capacity_audit import derive_capacities, verified_sources
from solarfl.models.mlp import MLP, fit_mlp


class CapacityAuditTests(unittest.TestCase):
    def setUp(self):
        self.devices = pd.DataFrame(
            {
                "device_hash_id": ["a1", "a2", "logger", "b1"],
                "station_hash_id": ["a", "a", "a", "b"],
                "max_power": [44.0, 55.0, np.nan, 44.0],
                "device_model": ["inverter", "inverter", "logger", "inverter"],
            }
        )
        self.stations = pd.DataFrame({"station_hash_id": ["b", "a"]})
        # Zero-energy membership must survive: positive-output filtering would
        # silently substitute a different rule for the historical denominator.
        self.obs = pd.DataFrame(
            {
                "device_hash_id": ["a1", "b1"],
                "measured_ts": ["2025-01-01", "2025-05-31"],
                "total_produced_energy": [0.0, 0.0],
            }
        )

    def test_membership_exclusions_and_deterministic_capacity_ties(self):
        labels, ledger, totals = derive_capacities(
            self.devices, self.stations, self.obs
        )
        self.assertEqual(
            labels,
            {
                "a": {"label": "S1", "capacity_kw": 44.0},
                "b": {"label": "S2", "capacity_kw": 44.0},
            },
        )
        reasons = ledger.set_index("device_hash_id").reason
        self.assertEqual(reasons["a2"], "absent_from_full_hourly_fact")
        self.assertEqual(reasons["logger"], "no_rated_power_metadata")
        self.assertEqual(totals.excluded_rated_power_kw.sum(), 55.0)
        shuffled = derive_capacities(
            self.devices.iloc[::-1], self.stations.iloc[::-1], self.obs.iloc[::-1]
        )
        self.assertEqual(labels, shuffled[0])
        pd.testing.assert_frame_equal(ledger, shuffled[1])

    def test_bad_metadata_and_membership_fail_loudly(self):
        for bad_devices, bad_obs in (
            (pd.concat([self.devices, self.devices.iloc[:1]]), self.obs),
            (self.devices.assign(max_power=[44, -1, np.nan, 44]), self.obs),
            (self.devices.assign(max_power=[np.inf, 55, np.nan, 44]), self.obs),
            (self.devices, self.obs.assign(device_hash_id=["unknown", "b1"])),
            (self.devices, pd.concat([self.obs, self.obs.iloc[:1]])),
            (self.devices, self.obs.assign(device_hash_id=["logger", "b1"])),
        ):
            with self.subTest(devices=len(bad_devices), observations=len(bad_obs)):
                with self.assertRaises(ValueError):
                    derive_capacities(bad_devices, self.stations, bad_obs)

    def test_corrupt_existing_source_is_not_downloaded_or_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            source = raw / "devices.csv"
            source.write_text("corrupted input")
            with patch("urllib.request.urlopen") as download:
                with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                    verified_sources(raw, download=True)
                download.assert_not_called()
            self.assertEqual(source.read_text(), "corrupted input")

    def test_source_manifest_has_official_ids_and_hashes(self):
        manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
        self.assertEqual(manifest["dataset_doi"], "10.17632/4zgsckxpdy.2")
        self.assertEqual(len(manifest["files"]), 4)
        for entry in manifest["files"].values():
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertIn(entry["file_id"], entry["url"])


class TrainingDisclosureTests(unittest.TestCase):
    def test_solar_features_use_ghi_not_gti_and_keep_hourly_label(self):
        raw = pd.DataFrame(
            {
                "measured_ts": ["2025-06-01 12:00:00", "2025-06-02 12:00:00"],
                "total_produced_energy": [5.0, 6.0],
                "terrestrial_radiation": [1000.0, 1000.0],
                "shortwave_radiation": [500.0, 600.0],
                "global_tilted_irradiance": [700.0, 900.0],
                "temperature_2m": [20.0, 21.0],
                "direct_radiation": [400.0, 450.0],
                "diffuse_radiation": [100.0, 150.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw.to_csv(Path(tmp) / "a.csv", index=False)
            with (
                patch("solarfl.data.features.CLIENTS_DIR", Path(tmp)),
                patch(
                    "solarfl.data.features.station_labels",
                    return_value={"a": {"label": "S1", "capacity_kw": 10}},
                ),
            ):
                X, y, ts = build_features("a", row_set="common", return_timestamps=True)
        self.assertEqual(len(X), 1)
        self.assertEqual(X.weather_future_kt.iloc[0], 0.6)
        self.assertEqual(X.weather_past_kt.iloc[0], 0.5)
        self.assertEqual(X.weather_future_global_tilted_irradiance.iloc[0], 900.0)
        self.assertEqual(ts.iloc[0], pd.Timestamp("2025-06-02 12:00:00"))
        self.assertEqual(y.iloc[0], 0.6)
        doy = ts.iloc[0].dayofyear
        self.assertAlmostEqual(
            X.cos_zenith.iloc[0],
            1000 / (1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))),
        )

    def test_width_keeps_daylight_and_parameter_counts(self):
        for cols, width, parameters in (
            (PAST_MODEL_MATRIX, 12, 2945),
            (MODEL_MATRIX, 17, 3265),
        ):
            self.assertEqual(len(cols), width)
            self.assertIn("is_daylight", cols)
            model = MLP(len(cols))
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)

    def test_confirmatory_scope_has_only_two_models_and_past_features(self):
        scope = experiment_scope(ROOT / "results")
        test = scope.loc[scope.role == "confirmatory"]
        self.assertEqual(set(test.regime), {"central_mlp", "fedprox_e1"})
        self.assertEqual(set(test.variant), {"past"})
        self.assertEqual(set(test.seeds), {"0;1;2;3;4"})
        self.assertTrue(
            scope.loc[scope.variant == "perfect", "split"].eq("validation").all()
        )

    def test_seed_reproducibility_and_optimizer_lifetimes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 12)).astype(np.float32)
        y = rng.uniform(size=20).astype(np.float32)

        def fit_central(seed):
            return fit_mlp(X, y, X, y, seed=seed, max_epochs=2, patience=3)

        def fit_federated(seed):
            return fit_fedprox(
                [X[:10], X[10:]],
                [y[:10], y[10:]],
                X,
                y,
                seed=seed,
                prox_mu=1,
                local_epochs=1,
                max_rounds=2,
                patience=3,
            )

        real_adam = torch.optim.Adam
        for fit, expected_optimizers in ((fit_central, 1), (fit_federated, 4)):
            optimizers = []

            def tracked_adam(*args, **kwargs):
                optimizer = real_adam(*args, **kwargs)
                self.assertEqual(len(optimizer.state), 0)
                optimizers.append(optimizer)
                return optimizer

            with patch("torch.optim.Adam", side_effect=tracked_adam):
                first = fit(3)
            self.assertEqual(len(optimizers), expected_optimizers)
            self.assertTrue(all(len(optimizer.state) > 0 for optimizer in optimizers))
            np.testing.assert_array_equal(first.predict(X), fit(3).predict(X))
            self.assertFalse(np.array_equal(first.predict(X), fit(4).predict(X)))


if __name__ == "__main__":
    unittest.main()
