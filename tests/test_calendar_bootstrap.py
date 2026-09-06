"""Tests for bootstrap behavior under unequal client test calendars."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from solarfl.eval.test_evaluation import (
    CLIENT_STRATIFIED,
    COMMON_OVERLAP,
    UNION_CALENDAR,
    _bootstrap_client_stratified_gaps,
    _calendar_sensitivity_tables,
    _common_calendar_days,
)


def _unequal_calendar_errors(equal_methods: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=49, freq="D")
    calendars = {"S1": dates, "S2": dates[-16:]}
    rows = []
    for client, client_dates in calendars.items():
        for seed in (0, 1):
            for day_number, day in enumerate(client_dates, start=1):
                central_error = 0.10 + 0.01 * seed
                if equal_methods:
                    fedprox_error = central_error
                else:
                    client_offset = 0.02 if client == "S1" else -0.01
                    fedprox_error = (
                        central_error + client_offset + 0.001 * day_number
                    )
                rows.append({
                    "calendar_day": day,
                    "client": client,
                    "seed": seed,
                    "central_abs_error": central_error,
                    "fedprox_abs_error": fedprox_error,
                })
    return pd.DataFrame(rows)


class CalendarBootstrapTests(unittest.TestCase):
    def test_common_calendar_is_the_client_intersection(self) -> None:
        common = _common_calendar_days(_unequal_calendar_errors())
        expected = pd.date_range("2026-02-03", periods=16, freq="D")
        self.assertEqual(common.tolist(), expected.tolist())

    def test_client_stratified_uses_each_clients_fixed_number_of_draws(
        self,
    ) -> None:
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        rows = []
        for day, gap in zip(dates, (1.0, 2.0, 3.0), strict=True):
            rows.append({
                "calendar_day": day,
                "client": "S1",
                "seed": 0,
                "central_abs_error": 0.0,
                "fedprox_abs_error": gap,
            })
        for day, gap in zip(dates[-2:], (10.0, 20.0), strict=True):
            rows.append({
                "calendar_day": day,
                "client": "S2",
                "seed": 0,
                "central_abs_error": 0.0,
                "fedprox_abs_error": gap,
            })
        errors = pd.DataFrame(rows)

        observed = _bootstrap_client_stratified_gaps(
            errors, repetitions=30, random_seed=7
        )

        rng = np.random.default_rng(7)
        s1_weights = rng.multinomial(3, np.full(3, 1 / 3), size=30)
        s2_weights = rng.multinomial(2, np.full(2, 1 / 2), size=30)
        s1_gap = s1_weights @ np.array([1.0, 2.0, 3.0]) / 3
        s2_gap = s2_weights @ np.array([10.0, 20.0]) / 2
        expected = (s1_gap + s2_gap) / 2

        np.testing.assert_allclose(observed, expected)
        np.testing.assert_array_equal(s1_weights.sum(axis=1), 3)
        np.testing.assert_array_equal(s2_weights.sum(axis=1), 2)

    def test_all_designs_preserve_zero_paired_gap(self) -> None:
        distributions, summaries = _calendar_sensitivity_tables(
            _unequal_calendar_errors(equal_methods=True),
            repetitions=40,
            random_seed=11,
        )
        self.assertTrue(
            np.allclose(
                distributions["macro_gap_fedprox_minus_central"], 0.0
            )
        )
        self.assertTrue(np.allclose(summaries["observed_macro_gap"], 0.0))

    def test_sensitivity_table_documents_each_resampling_design(self) -> None:
        first_draws, first_summary = _calendar_sensitivity_tables(
            _unequal_calendar_errors(), repetitions=40, random_seed=5
        )
        second_draws, second_summary = _calendar_sensitivity_tables(
            _unequal_calendar_errors(), repetitions=40, random_seed=5
        )

        pd.testing.assert_frame_equal(first_draws, second_draws)
        pd.testing.assert_frame_equal(first_summary, second_summary)
        self.assertEqual(
            first_summary["bootstrap_design"].tolist(),
            [UNION_CALENDAR, COMMON_OVERLAP, CLIENT_STRATIFIED],
        )
        self.assertEqual(
            first_summary["n_union_calendar_days"].tolist(), [49] * 3
        )
        self.assertEqual(
            first_summary["n_common_calendar_days"].tolist(), [16] * 3
        )
        self.assertEqual(
            first_summary["fixed_draws_per_client"].tolist(),
            [False, True, True],
        )
        self.assertEqual(
            first_summary["shared_date_weights_across_clients"].tolist(),
            [True, True, False],
        )
        self.assertEqual(
            first_summary["draws_per_repetition"].tolist(),
            ["49", "16", "S1:49;S2:16"],
        )
        gap_by_design = first_summary.set_index("bootstrap_design")[
            "observed_macro_gap"
        ]
        self.assertAlmostEqual(gap_by_design[UNION_CALENDAR], 0.02175)
        self.assertAlmostEqual(gap_by_design[COMMON_OVERLAP], 0.03)
        self.assertAlmostEqual(gap_by_design[CLIENT_STRATIFIED], 0.02175)
        self.assertEqual(
            len(first_draws),
            3 * 40,
        )


if __name__ == "__main__":
    unittest.main()
