import unittest

import pandas as pd

from src.feature_audit import ATTENDANCE, check_speaker_history


class SpeakerHistoryTests(unittest.TestCase):
    def test_only_consecutive_months_within_one_hcp_form_a_pattern(self):
        frame = pd.DataFrame({
            "hcp_id": [1, 1, 1, 2, 2, 2, 3, 3, 4],
            "period": ["2025-03", "2025-01", "2025-02",  # out of row order
                       "2025-01", "2025-02", "2025-04",  # a missing month
                       "2025-01", "2025-02", "2025-03"],  # different HCP
            ATTENDANCE: [0, 0, 1, 0, 1, 0, 0, 1, 0],
        })
        summary, examples = check_speaker_history(frame)
        self.assertEqual(summary["complete_consecutive_triples"], 1)
        self.assertEqual(summary["patterns_010"], 1)
        self.assertEqual(summary["affected_hcps"], 1)
        self.assertEqual(examples["hcp_id"].tolist(), [1])
        # Evidence remains traceable to the original CSV, before sorting.
        self.assertEqual(examples.iloc[0][["csv_line_lag2", "csv_line_lag1", "csv_line"]].tolist(),
                         [3, 4, 2])

    def test_separate_extracts_do_not_create_a_cross_boundary_history(self):
        training = pd.DataFrame({"hcp_id": [1, 1], "period": ["2025-11", "2025-12"],
                                 ATTENDANCE: [0, 1]})
        scoring = pd.DataFrame({"hcp_id": [1, 1, 1], "period": ["2026-01", "2026-02", "2026-03"],
                                ATTENDANCE: [0, 0, 0]})
        train_summary, _ = check_speaker_history(training)
        score_summary, _ = check_speaker_history(scoring)
        self.assertEqual(train_summary["complete_consecutive_triples"], 0)
        self.assertEqual(score_summary["complete_consecutive_triples"], 1)
        self.assertEqual(train_summary["patterns_010"] + score_summary["patterns_010"], 0)


if __name__ == "__main__":
    unittest.main()
