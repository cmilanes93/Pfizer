import unittest
import numpy as np
import pandas as pd

from src.metrics import monthly_metrics, weighted_precision
from src.modeling import tie_keys, split_window, fit_logistic, predict, validate_submission


class ValidationTests(unittest.TestCase):
    def test_capacity_is_applied_within_month(self):
        frame = pd.DataFrame({"hcp_id": [1, 2, 1, 2], "period": ["2025-01"] * 2 + ["2025-02"] * 2,
                              "responded_strict": [1, 0, 0, 1]})
        table = monthly_metrics(frame, "responded_strict", [.9, .8, .2, .1], "test", [.5])
        self.assertEqual(table.selected.tolist(), [1, 1])
        self.assertEqual(table.precision.tolist(), [1., 0.])

    def test_ties_do_not_depend_on_row_order_or_labels(self):
        frame = pd.DataFrame({"hcp_id": [4, 2, 1], "period": ["2025-01"] * 3, "responded_strict": [1, 0, 1]})
        original = pd.Series(tie_keys(frame), index=frame.hcp_id).sort_index()
        shuffled = frame.sample(frac=1, random_state=7).assign(responded_strict=0)
        alternate = pd.Series(tie_keys(shuffled), index=shuffled.hcp_id).sort_index()
        pd.testing.assert_series_equal(original, alternate)

    def test_next_period_labels_require_the_gap(self):
        frame = pd.DataFrame({"period": ["2025-05", "2025-06", "2025-07"]})
        window = {"fit_through": "2025-05", "start": "2025-07", "end": "2025-07"}
        fitting, evaluation = split_window(frame, window)
        self.assertEqual(fitting.period.tolist(), ["2025-05"])
        with self.assertRaises(ValueError):
            split_window(frame, dict(window, fit_through="2025-06"))

    def test_outcomes_and_operational_flags_cannot_enter_main_model(self):
        frame = pd.DataFrame({"decile": [1, 2, 3, 4], "responded_strict": [0, 0, 1, 1],
                              "responded_loose": [0, 1, 1, 1], "followup_call_logged_flag": [0, 0, 1, 1]})
        for forbidden in ["responded_strict", "responded_loose", "followup_call_logged_flag"]:
            with self.assertRaises(ValueError):
                fit_logistic(frame, "responded_strict", ["decile", forbidden])
        with self.assertRaises(ValueError):
            fit_logistic(frame, "responded_strict", ["responded_loose"], diagnostic=True)

    def test_scaler_is_fitted_on_training_and_unseen_categories_are_supported(self):
        fitting = pd.DataFrame({"decile": [1., 2., 3., 4.], "specialty": ["A", "B", "A", "B"],
                                "responded_strict": [0, 0, 1, 1]})
        future = pd.DataFrame({"decile": [10.], "specialty": ["NEW"]})
        model = fit_logistic(fitting, "responded_strict", ["decile", "specialty"])
        scaler = model.named_steps["preprocessing"].named_transformers_["numeric"]
        self.assertEqual(scaler.mean_[0], 2.5)
        self.assertTrue(np.isfinite(predict(model, future)).all())
        self.assertEqual(scaler.mean_[0], 2.5)

    def test_bootstrap_weighted_ranking_matches_explicit_repeated_rows(self):
        labels = np.array([1, 0, 1, 0])
        weights = np.array([2, 0, 3, 1])
        order = np.array([3, 0, 1, 2])
        expanded = np.repeat(labels[order], weights[order])
        k = int(np.ceil(.5 * len(expanded)))
        self.assertEqual(weighted_precision(labels, weights, order, .5), expanded[:k].mean())

    def test_submission_rejects_shuffled_ids_and_invalid_scores(self):
        template = pd.DataFrame({"row_id": [10, 11], "score": [.5, .5]})
        validate_submission(template.copy(), template)
        for invalid in [template.iloc[::-1].reset_index(drop=True), template.assign(score=[np.nan, .5]),
                        template.assign(score=[1.1, .5])]:
            with self.assertRaises(ValueError):
                validate_submission(invalid, template)


if __name__ == "__main__":
    unittest.main()
