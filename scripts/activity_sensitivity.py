"""Measure the predictive cost of CRM activity exclusions after model selection.

The estimator is built locally to test the five explicit sensitivity variants
while preserving fit_logistic's delivery feature guard. Its preprocessing and
logistic settings match fit_logistic; no delivery specification is changed.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import monthly_metrics, paired_hcp_bootstrap, summarise_metrics
from src.modeling import split_window

REPORT_DIR = ROOT / "reports/data_checks"
CSV_PATH = REPORT_DIR / "activity_sensitivity.csv"
JSON_PATH = REPORT_DIR / "activity_sensitivity.json"
TARGET = "responded_strict"
CAPACITIES = [0.05, 0.10, 0.20]
COMMITTED_LATER_PRECISION = 0.2697572712936318


def load_settings():
    settings = json.loads((ROOT / "config/challenger.json").read_text(encoding="utf-8"))
    revision = json.loads((ROOT / "config/revision.json").read_text(encoding="utf-8"))
    delivered = revision["eligible_candidate_sets"]["extended_without_speaker"]
    if len(delivered) != 11 or len(set(delivered)) != 11:
        raise ValueError("Expected eleven distinct delivered inputs")
    variants = {
        "delivered": delivered,
        "add_calls": delivered + ["calls_last_90d"],
        "add_samples": delivered + ["samples_dropped_last_90d"],
        "add_calls_samples_recency": delivered + [
            "calls_last_90d", "samples_dropped_last_90d", "days_since_last_call"],
        "remove_email_digital": [field for field in delivered if field not in {
            "email_sent_last_90d", "emails_opened_last_90d", "digital_engagement_score"}],
    }
    return settings, variants


def fit_sensitivity(fitting, features, settings):
    """Fit one configured sensitivity without changing the delivery estimator."""
    _, variants = load_settings()
    if list(features) not in variants.values():
        raise ValueError("Inputs must match an explicit activity sensitivity variant")
    if fitting[features].isna().any().any():
        raise ValueError("Sensitivity inputs contain missing values")
    categorical = fitting[features].select_dtypes(
        include=["object", "string", "category"]).columns.tolist()
    numeric = [field for field in features if field not in categorical]
    model = Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ])),
        ("classifier", LogisticRegression(
            C=settings["logistic_C"], solver="lbfgs", max_iter=1000,
            tol=1e-7, random_state=settings["seed"])),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fitting[features], fitting[TARGET])
    return model


def evaluate_window(frame, window_name, settings, variants):
    fitting, evaluation = split_window(
        frame, settings[window_name], horizon=settings["label_horizon_months"])
    monthly, predictions = [], {}
    for name, features in variants.items():
        model = fit_sensitivity(fitting, features, settings)
        scores = model.predict_proba(evaluation[features])[:, 1]
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError("Invalid sensitivity predictions")
        predictions[name] = scores
        monthly.append(monthly_metrics(
            evaluation, TARGET, scores, name, CAPACITIES, seed=settings["seed"]))
    summary = summarise_metrics(pd.concat(monthly, ignore_index=True)).rename(
        columns={"model": "variant"})
    summary["window"] = window_name
    columns = ["variant", "window", "capacity", "precision",
               "true_positives_selected", "selected", "average_precision"]
    return summary[columns], evaluation, predictions


def assert_delivery_anchor(summary):
    committed = pd.read_csv(ROOT / "reports/challenger/later_summary.csv")
    reference = committed.loc[
        committed.model.eq("extended_logistic") & committed.capacity.eq(0.1), "precision"]
    actual = summary.loc[
        summary.variant.eq("delivered") & summary.window.eq("later")
        & summary.capacity.eq(0.1), "precision"]
    assert len(reference) == len(actual) == 1, "Missing or ambiguous delivery anchor"
    assert abs(float(reference.iloc[0]) - COMMITTED_LATER_PRECISION) <= 1e-9
    difference = abs(float(actual.iloc[0]) - float(reference.iloc[0]))
    assert difference <= 1e-9, "Delivered sensitivity does not reproduce later precision"
    return {"source": "reports/challenger/later_summary.csv",
            "model": "extended_logistic", "capacity": 0.1,
            "committed_precision": float(reference.iloc[0]),
            "reproduced_precision": float(actual.iloc[0]),
            "absolute_difference": difference, "tolerance": 1e-9}


def checked_output_path(path):
    path = Path(path)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(REPORT_DIR.resolve()) or resolved == REPORT_DIR.resolve():
        raise ValueError("Sensitivity outputs must stay inside reports/data_checks/")
    return resolved


def write_outputs(summary, record, csv_path=CSV_PATH, json_path=JSON_PATH):
    """Validate both destinations before writing either report."""
    csv_path = checked_output_path(csv_path)
    json_path = checked_output_path(json_path)
    csv_text = summary.to_csv(index=False, lineterminator="\n")
    json_text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text, encoding="utf-8", newline="\n")
    json_path.write_text(json_text, encoding="utf-8", newline="\n")


def main():
    settings, variants = load_settings()
    frame = pd.read_csv(ROOT / "data/raw/hcp_engagement_train.csv")
    summaries, intervals = [], {}
    for window in ["development", "later"]:
        summary, evaluation, predictions = evaluate_window(frame, window, settings, variants)
        summaries.append(summary)
        if window == "later":
            for variant in ["add_calls", "add_samples"]:
                estimates, _ = paired_hcp_bootstrap(
                    evaluation, TARGET, predictions[variant], predictions["delivered"],
                    capacity=0.1, seed=42, replicates=2000)
                intervals[variant] = estimates["difference"]
    summary = pd.concat(summaries, ignore_index=True)
    anchor = assert_delivery_anchor(summary)
    record = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "variants": variants,
        "settings": {
            "target": TARGET, "capacities": CAPACITIES,
            "development": settings["development"], "later": settings["later"],
            "label_horizon_months": settings["label_horizon_months"],
            "seed": settings["seed"], "logistic_C": settings["logistic_C"],
            "solver": "lbfgs", "max_iter": 1000, "tol": 1e-7,
            "numeric_preprocessing": "StandardScaler fitted on the fitting window",
            "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
        },
        "bootstrap": {
            "window": "later", "baseline": "delivered", "capacity": 0.1,
            "replicates": 2000, "seed": 42, "confidence_level": 0.95,
            "method": "Paired whole-HCP resampling with fitted models fixed",
            "difference_units": "precision fraction; multiply by 100 for percentage points",
            "precision_difference_intervals": intervals,
        },
        "delivery_anchor": anchor,
        "aggregation": "Precision and average precision are monthly means; selected and true positives are summed across months.",
        "note": "This check ran after model selection. It changes no delivered score or recorded selection and creates no new holdout. Associations and conditional bootstrap intervals do not establish historical input availability or a causal engagement effect.",
    }
    write_outputs(summary, record)
    print(summary.loc[summary.window.eq("later")].to_string(index=False))
    print(json.dumps(record["bootstrap"], indent=2))


if __name__ == "__main__":
    main()
