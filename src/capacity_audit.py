"""Reproduce capacity curves and quantify cutoff ties in the empirical table.

This diagnostic keeps the established models and later window fixed. It does
not choose a model or a tie rule using outcomes. Random-tie ranges condition on
the exact observed labels and scores; they are not performance confidence
intervals or forecasts for a future population.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.baselines import PROFILE, TARGET, profile_rate
from src.modeling import decile_scores, fit_logistic, predict, split_window, tie_keys


def cutoff_ties(labels, scores, k):
    """Exact random-without-replacement distribution at one score cutoff."""
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    if len(y) != len(score) or not 1 <= k <= len(y):
        raise ValueError("Labels, scores and selected count must agree")
    if not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Finite scores and binary labels are required")
    threshold = np.sort(score)[-k]
    above, tied = score > threshold, score == threshold
    fixed_positive = int(y[above].sum())
    remaining = int(k - above.sum())
    block, positives = int(tied.sum()), int(y[tied].sum())
    law = hypergeom(block, positives, remaining)
    low = max(0, remaining - (block - positives))
    high = min(remaining, positives)
    return {
        "selected": k, "cutoff_score": float(threshold),
        "rows_above_cutoff": int(above.sum()),
        "positives_above_cutoff": fixed_positive,
        "tied_rows_at_cutoff": block, "positive_rows_in_tie": positives,
        "places_from_tie": remaining,
        "expected_precision": (fixed_positive + remaining * positives / block) / k,
        "random_tie_lower_95": float((fixed_positive + law.ppf(.025)) / k),
        "random_tie_upper_95": float((fixed_positive + law.ppf(.975)) / k),
        "minimum_possible_precision": (fixed_positive + low) / k,
        "maximum_possible_precision": (fixed_positive + high) / k,
    }


def run(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    out = root / "reports/comparison"
    out.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(root / "data/raw/hcp_engagement_train.csv")
    window = json.loads((root / "config/challenger.json").read_text())["later"]
    fields = json.loads((root / "config/revision.json").read_text())["eligible_candidate_sets"]
    fit, evaluation = split_window(training, window)
    scores = {"decile": decile_scores(evaluation),
              "three_field_rate": profile_rate(fit, evaluation)}
    for name, columns in [("core_logistic", fields["core"]),
                          ("delivered_logistic", fields["extended_without_speaker"])]:
        scores[name] = predict(fit_logistic(fit, TARGET, columns, C=.1, seed=42), evaluation)
    curves, tie_rows = [], []
    for month in sorted(evaluation.period.unique()):
        mask = evaluation.period.eq(month).to_numpy()
        rows = evaluation.loc[mask]
        labels = rows[TARGET].to_numpy()
        ties = tie_keys(rows)
        for name, all_scores in scores.items():
            values = all_scores[mask]
            order = np.lexsort((ties, -values))
            cumulative = np.cumsum(labels[order])
            for percentage in range(1, 101):
                capacity = percentage / 100
                k = math.ceil(capacity * len(rows))
                curves.append({"period": month, "model": name, "capacity": capacity,
                               "selected": k, "precision": float(cumulative[k - 1] / k)})
                if name == "three_field_rate" and percentage in (5, 10, 20):
                    tie_rows.append({"period": month, "capacity": capacity,
                                     "unique_scores": int(np.unique(values).size),
                                     "deterministic_precision": float(cumulative[k - 1] / k),
                                     **cutoff_ties(labels, values, k)})
    monthly = pd.DataFrame(curves)
    summary = monthly.groupby(["model", "capacity"], as_index=False).agg(precision=("precision", "mean"))
    ties = pd.DataFrame(tie_rows)
    # The extrema are exact bounds on the mean, with any allowable tie ordering
    # in each month. They are not a probabilistic interval across months.
    tie_summary = ties.groupby("capacity", as_index=False)[[
        "deterministic_precision", "expected_precision",
        "minimum_possible_precision", "maximum_possible_precision",
    ]].mean()
    reference = pd.read_csv(out / "summary.csv")
    reference = reference.loc[reference.stage.eq("later") & reference.model.isin(scores)]
    joined = reference.merge(summary, on=["model", "capacity"], suffixes=("_reference", "_curve"))
    max_error = float(np.max(np.abs(joined.precision_reference - joined.precision_curve)))
    if len(joined) != 12 or max_error > 1e-9:
        raise AssertionError("Capacity audit does not reproduce the primary comparison")
    monthly.to_csv(out / "capacity_monthly.csv", index=False)
    summary.to_csv(out / "capacity_summary.csv", index=False)
    ties.to_csv(out / "table_cutoff_ties.csv", index=False)
    payload = {
        "scope": __doc__.strip(), "fit_through": window["fit_through"],
        "evaluation_months": sorted(evaluation.period.unique()),
        "fitted_profile_cells": int(fit.groupby(PROFILE).ngroups),
        "unique_fitted_response_rates": int(fit.groupby(PROFILE)[TARGET].mean().nunique()),
        "primary_metric_max_absolute_difference": max_error,
        "mean_monthly_tie_sensitivity": tie_summary.to_dict("records"),
        "interpretation": "Random ties are uniform selections without replacement within the exact cutoff score. Monthly 95% ranges describe tie randomness only. Mean monthly extrema are attainable bounds, not confidence intervals. No outcome-aware tie rule is used for delivery.",
    }
    (out / "capacity_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(ties.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
