"""Tests for the non-inferiority margin sensitivity table."""

from __future__ import annotations

import unittest

import pandas as pd

from solarfl.eval.margin_sensitivity import build_margin_sensitivity


class MarginSensitivityTests(unittest.TestCase):
    def test_uses_strict_upper_bound_rule_and_marks_primary_margin(self) -> None:
        summary = pd.DataFrame([{
            "ci_upper": 0.0025,
            "noninferiority_margin": 0.005,
        }])

        result = build_margin_sensitivity(summary, margins=(0.0025, 0.005))

        self.assertEqual(result["noninferiority_passes"].tolist(), [False, True])
        self.assertEqual(
            result["role"].tolist(),
            ["post-hoc sensitivity", "pre-specified primary"],
        )

    def test_rejects_negative_margin(self) -> None:
        summary = pd.DataFrame([{
            "ci_upper": -0.001,
            "noninferiority_margin": 0.005,
        }])

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            build_margin_sensitivity(summary, margins=(-0.001,))


if __name__ == "__main__":
    unittest.main()
