"""Unit tests for feature-row eligibility policies."""

from __future__ import annotations

import unittest

import pandas as pd

from solarfl.data.features import MODEL_MATRIX, TARGET, _select_eligible_rows


class FeatureRowSelectionTests(unittest.TestCase):
    def test_past_ignores_missing_target_hour_reanalysis(self) -> None:
        frame = pd.DataFrame({col: [1.0] for col in MODEL_MATRIX})
        frame[TARGET] = 0.5
        frame["weather_future_temperature_2m"] = float("nan")

        past = _select_eligible_rows(frame, "past")
        common = _select_eligible_rows(frame, "common")

        self.assertEqual(len(past), 1)
        self.assertEqual(len(common), 0)

    def test_past_enforces_target_hour_daylight_directly(self) -> None:
        frame = pd.DataFrame({col: [1.0] for col in MODEL_MATRIX})
        frame[TARGET] = 0.0
        frame["is_daylight"] = 0.0
        frame["weather_future_kt"] = float("nan")

        past = _select_eligible_rows(frame, "past")

        self.assertEqual(len(past), 0)

    def test_unknown_row_set_is_rejected(self) -> None:
        frame = pd.DataFrame({col: [1.0] for col in MODEL_MATRIX})
        frame[TARGET] = 0.5

        with self.assertRaisesRegex(ValueError, "unknown row_set"):
            _select_eligible_rows(frame, "other")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
