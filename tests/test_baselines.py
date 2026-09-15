import unittest

import numpy as np
import pandas as pd

from src.baselines import history_rate, profile_rate


class BaselineTimingTests(unittest.TestCase):
    def test_future_outcomes_cannot_change_frozen_history(self):
        source = pd.DataFrame({
            "hcp_id": [1, 1, 1, 2],
            "period": ["2025-06", "2025-08", "2025-10", "2025-08"],
            "responded_strict": [0, 1, 0, 0],
        })
        rows = pd.DataFrame({
            "hcp_id": [1, 2, 3, 1],
            "period": ["2025-10", "2025-10", "2025-11", "2025-12"],
        })
        expected = np.array([0.5, 0, 1 / 3, 0.5])
        np.testing.assert_allclose(history_rate(source, rows, "2025-08"), expected)
        source.loc[source.period > "2025-08", "responded_strict"] = 1
        np.testing.assert_allclose(history_rate(source, rows, "2025-08"), expected)

    def test_profile_lookup_preserves_order_and_falls_back_for_unseen_cells(self):
        fitting = pd.DataFrame({
            "decile": [1, 1, 2], "formulary_status": ["A", "A", "B"],
            "specialty": ["X", "X", "Y"], "responded_strict": [0, 1, 1],
        })
        rows = pd.DataFrame({
            "decile": [2, 3, 1], "formulary_status": ["B", "C", "A"],
            "specialty": ["Y", "Z", "X"], "responded_strict": [0, 0, 0],
        })
        expected = [1, 2 / 3, 0.5]
        np.testing.assert_allclose(profile_rate(fitting, rows), expected)
        rows["responded_strict"] = 1
        np.testing.assert_allclose(profile_rate(fitting, rows), expected)


if __name__ == "__main__":
    unittest.main()
