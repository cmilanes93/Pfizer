"""Compare simple HCP policies on fixed temporal windows.

The empirical profile table and HCP history use fitting-period response rates,
with fitting prevalence for unseen groups. Rolling histories are diagnostic:
the supplied scoring batch contains no newly observed outcomes. These are
exploratory comparisons after delivery selection, without hyperparameter tuning.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import monthly_metrics, paired_hcp_bootstrap, summarise_metrics
from src.modeling import decile_scores, fit_logistic, predict, split_window, tie_keys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ["decile", "formulary_status", "specialty"]
TARGET = "responded_strict"
STAGES = ("development", "later", "six_month")
CAPACITIES = [0.05, 0.10, 0.20]
COMPARISON_CAPACITY = 0.10
LOGISTIC_C = 0.1
SEED = 42
BOOTSTRAP_REPLICATES = 2000
EMAIL_DIGITAL_FIELDS = {
    "email_sent_last_90d", "emails_opened_last_90d", "digital_engagement_score",
}


def history_rate(source, rows, cutoff):
    """Map HCP means using only labels through the specified row-month cutoff."""
    past = source.loc[source.period.le(cutoff)]
    rates = past.groupby("hcp_id")[TARGET].mean()
    return rows.hcp_id.map(rates).fillna(past[TARGET].mean()).to_numpy()


def profile_rate(fitting, rows):
    """Look up unsmoothed response rates for decile, formulary and specialty."""
    rates = fitting.groupby(PROFILE)[TARGET].mean()
    index = pd.MultiIndex.from_frame(rows[PROFILE])
    scores = rates.reindex(index).to_numpy()
    return np.nan_to_num(scores, nan=float(fitting[TARGET].mean()))


def policy_scores(training, fitting, evaluation, window, fields):
    scores = {
        "decile": decile_scores(evaluation),
        "three_field_rate": profile_rate(fitting, evaluation),
        "hcp_history_frozen": history_rate(training, evaluation, window["fit_through"]),
    }
    logistic_inputs = {
        "core_logistic": fields["core"],
        "delivered_logistic": fields["extended_without_speaker"],
        "without_email_digital": [
            field for field in fields["extended_without_speaker"]
            if field not in EMAIL_DIGITAL_FIELDS
        ],
    }
    for name, features in logistic_inputs.items():
        model = fit_logistic(fitting, TARGET, features, C=LOGISTIC_C, seed=SEED)
        scores[name] = predict(model, evaluation)

    rolling = np.zeros(len(evaluation))
    for month in sorted(evaluation.period.unique()):
        mask = evaluation.period.eq(month).to_numpy()
        cutoff = str(pd.Period(month, freq="M") - 2)
        rolling[mask] = history_rate(training, evaluation.loc[mask], cutoff)
    scores["hcp_history_rolling_diagnostic"] = rolling
    return scores


def selected_hcps_by_month(evaluation, scores):
    """Return the selected HCP sets at the fixed illustrative 10% capacity."""
    rows = evaluation[["hcp_id", "period", TARGET]].copy()
    rows["score"] = scores
    selected_sets = []
    for _, month_rows in rows.groupby("period"):
        order = np.lexsort((tie_keys(month_rows, SEED), -month_rows.score.to_numpy()))
        count = math.ceil(COMPARISON_CAPACITY * len(month_rows))
        selected_rows = month_rows.iloc[order[:count]]
        selected_sets.append(set(selected_rows.hcp_id))
    return selected_sets


def later_comparisons(evaluation, scores, selected_sets):
    comparisons = {}
    for baseline in ("three_field_rate", "core_logistic", "hcp_history_frozen"):
        intervals, _ = paired_hcp_bootstrap(
            evaluation, TARGET, scores["delivered_logistic"], scores[baseline],
            COMPARISON_CAPACITY, SEED, BOOTSTRAP_REPLICATES,
        )
        comparisons[baseline] = intervals["difference"]

    comparisons["adjacent_month_retained_fraction"] = {
        name: float(np.mean([
            len(previous & current) / len(current)
            for previous, current in zip(months[:-1], months[1:])
        ]))
        for name, months in selected_sets.items()
    }
    comparisons["same_month_overlap_without_email_digital"] = float(np.mean([
        len(delivered & reduced) / len(delivered)
        for delivered, reduced in zip(
            selected_sets["delivered_logistic"], selected_sets["without_email_digital"]
        )
    ]))
    return comparisons


def response_history_diagnostics(training):
    """Describe response history using exact calendar lags, separately from ranking."""
    indexed = training.assign(
        month=pd.PeriodIndex(training.period, freq="M")
    ).set_index(["hcp_id", "month"])
    lagged_keys = pd.MultiIndex.from_arrays([
        training.hcp_id, pd.PeriodIndex(training.period, freq="M") - 2,
    ])
    lagged_labels = indexed[TARGET].reindex(lagged_keys).to_numpy()
    lagged_rates = {
        str(value): {
            "rows": int((lagged_labels == value).sum()),
            "response_rate": float(training.loc[lagged_labels == value, TARGET].mean()),
        }
        for value in (0, 1)
    }
    rates, labels = [], []
    for month in sorted(training.period.unique())[2:]:
        rows = training.loc[training.period.eq(month)]
        cutoff = str(pd.Period(month, freq="M") - 2)
        rates.extend(history_rate(training, rows, cutoff))
        labels.extend(rows[TARGET])
    positive_counts = training.groupby("hcp_id")[TARGET].sum().sort_values(ascending=False)
    highest_count = math.ceil(0.2 * len(positive_counts))
    return {
        "lag2_response_rates_all_history_descriptive": lagged_rates,
        "pooled_history_rate_auc_temporal_diagnostic": float(roc_auc_score(labels, rates)),
        "positive_share_top_20pct_hcps_descriptive": float(
            positive_counts.head(highest_count).sum() / positive_counts.sum()
        ),
    }


def write_metric_tables(monthly, output, prefix):
    summary = pd.concat([
        summarise_metrics(stage_rows).assign(stage=stage)
        for stage, stage_rows in monthly.groupby("stage")
    ], ignore_index=True)
    monthly.to_csv(output / f"{prefix}_monthly.csv", index=False)
    summary.to_csv(output / f"{prefix}_summary.csv", index=False)
    return summary


def incremental_history(training, settings, core, output):
    """Compare core models with identical fitting rows, with and without past response."""
    training = training.copy()
    history_column = "hcp_prior_response_rate"
    training[history_column] = np.nan
    for month in sorted(training.period.unique())[2:]:
        mask = training.period.eq(month)
        cutoff = str(pd.Period(month, freq="M") - 2)
        training.loc[mask, history_column] = history_rate(training, training.loc[mask], cutoff)

    tables, intervals = [], {}
    for stage in STAGES:
        fitting, evaluation = split_window(training, settings[stage])
        fitting = fitting.dropna(subset=[history_column])
        evaluation[history_column] = history_rate(
            training, evaluation, settings[stage]["fit_through"]
        )
        scores = {}
        for name, features in (
            ("core_comparable_fit", core),
            ("core_plus_history", core + [history_column]),
        ):
            model = fit_logistic(fitting, TARGET, features, C=LOGISTIC_C, seed=SEED)
            scores[name] = predict(model, evaluation)
            metrics = monthly_metrics(
                evaluation, TARGET, scores[name], name, CAPACITIES, seed=SEED
            )
            metrics["stage"] = stage
            tables.append(metrics)
        if stage == "later":
            intervals, _ = paired_hcp_bootstrap(
                evaluation, TARGET, scores["core_plus_history"], scores["core_comparable_fit"],
                COMPARISON_CAPACITY, SEED, BOOTSTRAP_REPLICATES,
            )

    monthly = pd.concat(tables, ignore_index=True)
    write_metric_tables(monthly, output, "history_increment")
    evidence = {
        "scope": (
            "Exploratory core plus past response rate. Both models exclude Jan-Feb 2024 "
            "fitting rows without history. Fit features use labels through t-2; evaluation "
            "histories freeze at fit cutoff. No retuning or delivery replacement."
        ),
        "later_conditional_intervals": intervals,
    }
    (output / "history_increment.json").write_text(json.dumps(evidence, indent=2) + "\n")


def run(repo, output=None):
    root = Path(repo).resolve()
    output = Path(output).resolve() if output else root / "reports/comparison"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "data/raw/hcp_engagement_train.csv"
    training = pd.read_csv(source)
    settings = json.loads((root / "config/challenger.json").read_text())
    fields = json.loads((root / "config/revision.json").read_text())["eligible_candidate_sets"]

    tables, comparisons = [], {}
    for stage in STAGES:
        window = settings[stage]
        fitting, evaluation = split_window(training, window)
        scores = policy_scores(training, fitting, evaluation, window, fields)
        selected_sets = {}
        for name, values in scores.items():
            metrics = monthly_metrics(
                evaluation, TARGET, values, name, CAPACITIES,
                seed=SEED, probability=name != "decile",
            )
            metrics["stage"] = stage
            tables.append(metrics)
            selected_sets[name] = selected_hcps_by_month(evaluation, values)
        if stage == "later":
            comparisons.update(later_comparisons(evaluation, scores, selected_sets))

    monthly = pd.concat(tables, ignore_index=True)
    summary = write_metric_tables(monthly, output, "baseline")
    comparisons.update(response_history_diagnostics(training))
    comparisons["scope"] = (
        "Exploratory comparison of simple policies. Frozen history can use only outcomes "
        "through the fitting cutoff. Rolling history uses newly matured historical outcomes "
        "and does not mirror the supplied six-month scoring batch. Bootstrap intervals "
        "condition on fitted scores and observed months."
    )
    comparisons["source_training_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "checks.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    columns = ["stage", "model", "precision", "average_precision", "roc_auc",
               "selected", "true_positives_selected"]
    print(summary.loc[summary.capacity.eq(COMPARISON_CAPACITY), columns].to_string(index=False))
    print(json.dumps(comparisons, indent=2))
    incremental_history(training, settings, fields["core"], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.repo, args.output)


if __name__ == "__main__":
    main()
