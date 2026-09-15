"""Check the tie calculation against enumeration, including a full cutoff block."""
from itertools import combinations
import unittest

import numpy as np

from src.capacity_audit import cutoff_ties


class CutoffTiesTest(unittest.TestCase):
    def test_distribution_matches_all_possible_selections(self):
        # One guaranteed positive; choose two out of a tied group of four.
        labels = [1, 1, 1, 0, 0, 0]
        scores = [.9, .5, .5, .5, .5, .1]
        result = cutoff_ties(labels, scores, 3)
        possible = [(1 + sum(labels[i] for i in pair)) / 3
                    for pair in combinations(range(1, 5), 2)]
        self.assertAlmostEqual(result["expected_precision"], np.mean(possible))
        self.assertEqual(result["minimum_possible_precision"], min(possible))
        self.assertEqual(result["maximum_possible_precision"], max(possible))
        self.assertEqual(result["places_from_tie"], 2)

    def test_fully_selected_tie_has_no_randomness(self):
        result = cutoff_ties([1, 0, 1, 0], [.9, .5, .5, .1], 3)
        for field in ("expected_precision", "random_tie_lower_95", "random_tie_upper_95",
                      "minimum_possible_precision", "maximum_possible_precision"):
            self.assertEqual(result[field], 2 / 3)


if __name__ == "__main__":
    unittest.main()
