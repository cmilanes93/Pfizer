import unittest
from unittest.mock import patch

import pandas as pd

from scripts.activity_sensitivity import (
    ROOT, CSV_PATH, JSON_PATH, assert_delivery_anchor, evaluate_window,
    load_settings, write_outputs,
)


class ActivitySensitivityTests(unittest.TestCase):
    def test_delivered_later_precision_matches_committed_value(self):
        settings, variants = load_settings()
        frame = pd.read_csv(ROOT / "data/raw/hcp_engagement_train.csv")
        summary, _, _ = evaluate_window(
            frame, "later", settings, {"delivered": variants["delivered"]})
        anchor = assert_delivery_anchor(summary)
        self.assertLessEqual(abs(anchor["reproduced_precision"] - 0.2697572712936318), 1e-9)

    def test_output_writer_rejects_paths_outside_data_checks(self):
        outside = [ROOT / "outputs/activity_sensitivity.csv",
                   "reports/data_checks/../activity_sensitivity.csv",
                   ROOT / "reports/data_checks_backup/activity_sensitivity.csv"]
        with patch("pathlib.Path.write_text") as write:
            for path in outside:
                for csv_path, json_path in [(path, JSON_PATH), (CSV_PATH, path)]:
                    with self.subTest(csv=csv_path, json=json_path):
                        with self.assertRaises(ValueError):
                            write_outputs(pd.DataFrame(), {}, csv_path, json_path)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
