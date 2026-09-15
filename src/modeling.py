"""Small, explicit feature sets and temporal fitting for the challenge."""

import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = {"responded_strict", "responded_loose"}
FLAGS = {"followup_call_logged_flag", "territory_target_refresh_flag"}
EXCLUDED = {"hcp_id", "row_id", "period", "campaign_wave_id", "territory_id",
            "calls_last_90d", "samples_dropped_last_90d", "days_since_last_call",
            "patient_volume_est", "nbrx_last_month", "nbrx_3m_avg", "trx_3m_avg"}


def tie_keys(frame, seed=42):
    """Stable, label-independent tie order, unchanged by row shuffling."""
    return np.array([int.from_bytes(hashlib.blake2b(
        f"{seed}|{hcp}|{period}".encode(), digest_size=8).digest(), "big")
        for hcp, period in zip(frame.hcp_id, frame.period)], dtype=np.uint64)


def split_window(frame, window, extra_lag=0, horizon=1):
    start = pd.Period(window["start"], freq="M")
    cutoff = pd.Period(window["fit_through"], freq="M") - extra_lag
    latest_allowed = start - horizon - 1 - extra_lag
    if cutoff > latest_allowed:
        raise ValueError("Training includes labels unavailable at evaluation start")
    fitting = frame.loc[frame.period <= str(cutoff)].copy()
    evaluation = frame.loc[frame.period.between(window["start"], window["end"])].copy()
    if fitting.empty or evaluation.empty:
        raise ValueError("Empty fitting or evaluation partition")
    return fitting, evaluation


def fit_logistic(frame, target, features, C=1.0, seed=42, diagnostic=False):
    features = list(features)
    invalid = set(features) & (TARGETS | EXCLUDED | (set() if diagnostic else FLAGS))
    if invalid:
        raise ValueError(f"Forbidden predictors: {sorted(invalid)}")
    if target not in TARGETS:
        raise ValueError("Unsupported response definition")
    if frame[features].isna().any().any():
        raise ValueError("Selected predictors contain missing values")
    categorical = frame[features].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [feature for feature in features if feature not in categorical]
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    model = Pipeline([
        ("preprocessing", transformer),
        ("classifier", LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                           tol=1e-7, random_state=seed)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame[features], frame[target])
    return model


def predict(model, frame):
    values = model.predict_proba(frame)[:, 1]
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Invalid response scores")
    return values


def decile_scores(frame):
    """Ordinal ranking only; this is not a response probability."""
    return frame.decile.to_numpy(dtype=float) / 10


def validate_submission(submission, template):
    if list(submission.columns) != ["row_id", "score"]:
        raise ValueError("Submission columns differ from the template")
    if not submission.row_id.equals(template.row_id) or not submission.row_id.is_unique:
        raise ValueError("Submission rows differ from the template")
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError("Scores must be finite and between zero and one")
