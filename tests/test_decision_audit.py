import unittest

import pandas as pd

from src.decision_audit import (decile_tie_expectation, profile_support,
                                select_monthly, selection_diagnostics, temporal_target_check)


class DecisionAuditTests(unittest.TestCase):
    def test_temporal_direction_uses_t_plus_one_and_two_with_exact_months(self):
        frame = pd.DataFrame({"hcp_id": [1, 1, 1, 2, 2, 2],
                              "period": ["2025-03", "2025-01", "2025-02", "2025-01", "2025-02", "2025-04"],
                              "nbrx_last_month": [4., 999., 5., 0., 5., 8.],
                              "responded_strict": [0, 1, 0, 1, 0, 0],
                              "responded_loose": [0, 1, 0, 1, 0, 0]})
        summary, _, _, checked = temporal_target_check(frame)
        self.assertEqual(summary["checked_rows"], 1)
        self.assertEqual(checked.iloc[0]["direction"], "down")
        self.assertEqual(checked.iloc[0]["conditional_nbrx_change"], -1.)
        self.assertEqual(checked.iloc[0]["csv_line"], 3)
        self.assertEqual(checked.iloc[0]["responded_strict"], 1)  # Never replace the label.

    def test_selection_rounding_ties_and_repeated_hcps(self):
        frame = pd.DataFrame({"hcp_id": [1, 2, 3, 1, 2, 3], "period": ["2025-01"] * 3 + ["2025-02"] * 3,
                              "responded_strict": [1, 0, 0, 0, 1, 0], "fixed": [.8, .8, .1, .8, .8, .1]})
        selected = select_monthly(frame, models=["fixed"], capacities=[.5])
        shuffled = select_monthly(frame.sample(frac=1, random_state=7), models=["fixed"], capacities=[.5])
        self.assertEqual(list(zip(selected.period, selected.hcp_id)), list(zip(shuffled.period, shuffled.hcp_id)))
        summary, _, turnover, _, _ = selection_diagnostics(selected)
        self.assertEqual(summary.iloc[0]["selection_slots"], 4)
        self.assertEqual(summary.iloc[0]["unique_selected_hcps"], 2)
        self.assertEqual(summary.iloc[0]["repeat_slots_after_first_selection"], 2)
        self.assertEqual(turnover.iloc[0]["retained"], 2)

    def test_random_boundary_ties_have_exact_expectation(self):
        frame = pd.DataFrame({"hcp_id": [1, 2, 3, 4], "period": ["2025-01"] * 4,
                              "responded_strict": [1, 0, 0, 1], "decile": [1., 1., 1., .1]})
        result = decile_tie_expectation(frame, capacities=[.5]).iloc[0]
        self.assertEqual(result["slots_from_boundary_ties"], 2)
        self.assertAlmostEqual(result["expected_positives_random_ties"], 2 / 3)
        self.assertAlmostEqual(result["expected_precision_random_ties"], 1 / 3)

    def test_profile_support_checks_joint_combinations_and_counts_new_hcps(self):
        train = pd.DataFrame({"hcp_id": [1, 2], "period": ["2025-10"] * 2,
                              "decile": [1, 2], "specialty": ["A", "B"], "formulary_status": ["X", "Y"],
                              "responded_strict": [0, 1]})
        score = pd.DataFrame({"hcp_id": [1, 3], "period": ["2026-01"] * 2,
                              "decile": [1, 1], "specialty": ["A", "B"], "formulary_status": ["X", "Y"]})
        _, summary, _ = profile_support(train, score)
        summary = summary[summary.reference == "all_supplied_training"].set_index("cohort")
        self.assertEqual(summary.loc["known_hcp", "unseen_profile_rows"], 0)
        self.assertEqual(summary.loc["new_hcp", "unseen_profile_rows"], 1)
        self.assertEqual(summary.loc["new_hcp", "minimum_profile_historical_rows"], 0)


if __name__ == "__main__":
    unittest.main()
