from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.challenger import (PROVENANCE_FILES, ROOT, VECTOR, capacity_curve, choose_spec,
                            config, digest, fingerprints, fit_boosting, read_selection,
                            score_final, specifications, write_json)
from src.metrics import monthly_metrics
from src.modeling import predict


class ChallengerTests(unittest.TestCase):
    def vectors(self, core, extended, boosting, baseline=(.1, .2, .15, .12)):
        return pd.DataFrame([
            {"model": name, **dict(zip(VECTOR, values))}
            for name, values in [("decile", baseline), ("core_logistic", core),
                                 ("extended_logistic", extended), ("extended_boosting", boosting)]])

    def test_high_ap_cannot_offset_losing_to_decile_on_a_capacity_metric(self):
        settings = config()
        vectors = self.vectors((.11, .21, .16, .13), (.30, .19, .28, .20), (.35, .18, .30, .22))
        decision = choose_spec(vectors, specifications(settings), settings["comparison_epsilon"])
        self.assertEqual(decision["admissible"], ["core_logistic"])
        self.assertEqual(decision["selected"]["name"], "core_logistic")

    def test_dominated_simpler_model_is_removed_and_tradeoffs_use_simplicity(self):
        specs = specifications(config())
        # Extended logistic dominates the core. Boosting trades higher AP for
        # lower P@5%, so both extended algorithms survive; logistic comes first.
        vectors = self.vectors((.11, .21, .16, .13), (.20, .30, .25, .20), (.22, .29, .26, .21))
        decision = choose_spec(vectors, specs)
        self.assertNotIn("core_logistic", decision["nondominated"])
        self.assertEqual(decision["selected"]["name"], "extended_logistic")
        # If core also trades off instead of being dominated, fewer fields wins.
        vectors.loc[vectors.model.eq("core_logistic"), "precision_05"] = .31
        self.assertEqual(choose_spec(vectors, specs)["selected"]["name"], "core_logistic")

    def test_numeric_tolerance_is_not_a_practical_one_point_margin(self):
        specs = specifications(config())
        baseline = np.array([.1, .2, .15, .12])
        near = tuple(baseline + 5e-13)
        self.assertEqual(choose_spec(self.vectors(near, near, near), specs)["selected"]["name"], "decile")
        vectors = self.vectors(tuple(baseline + .001), tuple(baseline + .0011), tuple(baseline + .0012))
        self.assertEqual(choose_spec(vectors, specs)["selected"]["name"], "extended_boosting")

    def test_boosting_preprocessing_is_training_only_and_quarantined_inputs_fail(self):
        spec = specifications(config())["extended_boosting"].copy()
        spec["features"] = ["decile", "specialty"]
        fitting = pd.DataFrame({"decile": [1, 2, 3, 4, 5, 6, 7, 8],
                                "specialty": ["A", "B"] * 4, "responded_strict": [0, 0, 1, 0, 1, 0, 1, 1]})
        model = fit_boosting(fitting, "responded_strict", spec)
        encoder = model.named_steps["preprocessing"].named_transformers_["category"]
        categories = encoder.categories_[0].copy()
        self.assertTrue(np.isfinite(predict(model, pd.DataFrame({"decile": [10], "specialty": ["NEW"]}))).all())
        np.testing.assert_array_equal(encoder.categories_[0], categories)
        self.assertIs(model.named_steps["classifier"].early_stopping, False)
        self.assertIsNone(model.named_steps["classifier"].categorical_features)
        transformed = model.named_steps["preprocessing"].transform(fitting)
        np.testing.assert_array_equal(transformed[:, 0], fitting.decile)
        for invalid in ["speaker_program_attended_12m", "responded_strict", "followup_call_logged_flag"]:
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                fit_boosting(fitting, "responded_strict", dict(spec, features=["decile", invalid]))

    def test_curve_counts_match_capacity_metrics_and_full_capacity_is_prevalence(self):
        settings = config()
        frame = pd.DataFrame({"hcp_id": range(1, 12), "period": ["2025-07"] * 6 + ["2025-08"] * 5,
                              "responded_strict": [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1]})
        scores = np.linspace(.95, .05, len(frame))
        curve = capacity_curve(frame, "responded_strict", scores, "sample", settings)
        metrics = monthly_metrics(frame, "responded_strict", scores, "sample", [.05, .1, .2])
        common = curve.loc[curve.capacity.isin([.05, .1, .2])].reset_index(drop=True)
        for field in ["selected", "true_positives_selected", "precision", "recall", "lift"]:
            np.testing.assert_allclose(common[field], metrics[field], rtol=0, atol=1e-12)
        full = curve.loc[curve.capacity.eq(1)]
        np.testing.assert_allclose(full.precision, full.prevalence, rtol=0, atol=1e-12)
        np.testing.assert_allclose(full.recall, 1)

    def test_january_scoring_excludes_december_labels_even_with_june_inputs(self):
        settings = config()
        training = pd.DataFrame({"period": ["2025-10", "2025-11", "2025-12"]})
        scoring = pd.DataFrame({"period": ["2026-01", "2026-06"]})
        selection = {"selected": specifications(settings)["extended_logistic"]}
        with patch("src.challenger.score_spec", side_effect=RuntimeError("Fitting intercepted")) as fit:
            with self.assertRaisesRegex(RuntimeError, "Fitting intercepted"):
                score_final(training, scoring, None, settings, selection)
            self.assertEqual(fit.call_args.args[0].period.tolist(), ["2025-10", "2025-11"])
        with patch("src.challenger.score_spec") as fit:
            with self.assertRaisesRegex(ValueError, "unavailable labels"):
                score_final(training, scoring, None,
                            dict(settings, scoring_fit_through="2025-12"), selection)
            fit.assert_not_called()

    def test_provenance_and_metric_vector_edits_invalidate_the_selection(self):
        settings = config()
        vectors = self.vectors((.11, .21, .16, .13), (.20, .30, .25, .20), (.22, .29, .26, .21))
        with tempfile.TemporaryDirectory() as folder:
            root, report = Path(folder), Path(folder) / "reports/challenger"
            report.mkdir(parents=True)
            for relative in PROVENANCE_FILES.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixed source")
            (root / "config/revision.json").write_bytes((ROOT / "config/revision.json").read_bytes())
            freeze = {"fingerprints": fingerprints(root)}
            write_json(report / "implementation_freeze.json", freeze)
            vectors.to_csv(report / "development_metric_vectors.csv", index=False)
            decision = choose_spec(vectors, specifications(settings, root))
            selection = {"decision": decision, "selected": decision["selected"], "fingerprints": fingerprints(root),
                         "implementation_freeze_sha256": digest(report / "implementation_freeze.json"),
                         "development_vectors_sha256": digest(report / "development_metric_vectors.csv")}
            write_json(report / "selection.json", selection)
            self.assertEqual(read_selection(settings, report, root)["selected"]["name"], "extended_logistic")
            original_specification = selection["selected"]
            selection["selected"] = specifications(settings, root)["core_logistic"]
            write_json(report / "selection.json", selection)
            with self.assertRaisesRegex(ValueError, "recorded selection rule"):
                read_selection(settings, report, root)
            selection["selected"] = original_specification
            write_json(report / "selection.json", selection)
            training_path = root / "data/raw/hcp_engagement_train.csv"
            original_training = training_path.read_bytes()
            training_path.write_text("Different training data")
            with self.assertRaisesRegex(ValueError, "Source or configuration changed"):
                read_selection(settings, report, root)
            training_path.write_bytes(original_training)
            vectors.loc[vectors.model.eq("extended_boosting"), "precision_05"] = .50
            vectors.to_csv(report / "development_metric_vectors.csv", index=False)
            with self.assertRaisesRegex(ValueError, "vectors changed"):
                read_selection(settings, report, root)


if __name__ == "__main__":
    unittest.main()
