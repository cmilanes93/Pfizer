"""Monthly capacity metrics and paired HCP bootstrap uncertainty."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.modeling import tie_keys


def ranking_order(scores, ties):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking contains non-finite scores")
    return np.lexsort((ties, -scores))


def monthly_metrics(frame, target, scores, model_name, capacities, seed=42, probability=True):
    work = frame[["hcp_id", "period", target]].copy()
    work["prediction"] = np.asarray(scores)
    work["tie"] = tie_keys(work, seed)
    rows = []
    for period, group in work.groupby("period", sort=True):
        y, score = group[target].to_numpy(), group.prediction.to_numpy()
        order = ranking_order(score, group.tie.to_numpy())
        positives, prevalence = int(y.sum()), y.mean()
        for capacity in capacities:
            k = math.ceil(capacity * len(group))
            found = int(y[order[:k]].sum())
            precision = found / k
            rows.append({"model": model_name, "period": period, "target": target,
                         "capacity": capacity, "rows": len(group), "positives": positives,
                         "selected": k, "true_positives_selected": found,
                         "prevalence": prevalence, "precision": precision,
                         "recall": found / positives if positives else np.nan,
                         "lift": precision / prevalence if prevalence else np.nan,
                         "average_precision": average_precision_score(y, score) if positives else np.nan,
                         "roc_auc": roc_auc_score(y, score) if 0 < positives < len(y) else np.nan,
                         "mean_prediction": score.mean() if probability else np.nan,
                         "brier": brier_score_loss(y, score) if probability else np.nan,
                         "log_loss": log_loss(y, score, labels=[0, 1]) if probability else np.nan})
    return pd.DataFrame(rows)


def summarise_metrics(monthly):
    return monthly.groupby(["model", "target", "capacity"], as_index=False).agg(
        months=("period", "nunique"), rows=("rows", "sum"), positives=("positives", "sum"),
        selected=("selected", "sum"), true_positives_selected=("true_positives_selected", "sum"),
        precision=("precision", "mean"), precision_min=("precision", "min"),
        precision_max=("precision", "max"), recall=("recall", "mean"), lift=("lift", "mean"),
        average_precision=("average_precision", "mean"), roc_auc=("roc_auc", "mean"),
        prevalence=("prevalence", "mean"), mean_prediction=("mean_prediction", "mean"),
        brier=("brier", "mean"), log_loss=("log_loss", "mean"))


def weighted_precision(labels, weights, order, capacity):
    """Exact top-k precision for an integer-weight bootstrap sample."""
    weight = np.asarray(weights, dtype=np.int64)[order]
    total = int(weight.sum())
    if total == 0:
        return np.nan
    k = math.ceil(capacity * total)
    previous = np.cumsum(weight) - weight
    selected = np.minimum(weight, np.maximum(0, k - previous))
    return float(np.dot(selected, np.asarray(labels)[order]) / k)


def paired_hcp_bootstrap(frame, target, selected_scores, baseline_scores,
                         capacity=.1, seed=42, replicates=2000):
    work = frame[["hcp_id", "period", target]].copy()
    work["selected_score"] = np.asarray(selected_scores)
    work["baseline_score"] = np.asarray(baseline_scores)
    work["tie"] = tie_keys(work, seed)
    hcps = np.sort(work.hcp_id.unique())
    groups = []
    for _, group in work.groupby("period", sort=True):
        groups.append((group[target].to_numpy(), np.searchsorted(hcps, group.hcp_id),
                       ranking_order(group.selected_score, group.tie.to_numpy()),
                       ranking_order(group.baseline_score, group.tie.to_numpy())))
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(replicates):
        weights = rng.multinomial(len(hcps), np.full(len(hcps), 1 / len(hcps)))
        model_p, baseline_p = [], []
        for labels, indexes, model_order, baseline_order in groups:
            model_p.append(weighted_precision(labels, weights[indexes], model_order, capacity))
            baseline_p.append(weighted_precision(labels, weights[indexes], baseline_order, capacity))
        selected, baseline = np.mean(model_p), np.mean(baseline_p)
        results.append({"selected_precision": selected, "baseline_precision": baseline,
                        "difference": selected - baseline})
    draws = pd.DataFrame(results)
    estimates = {}
    for column in draws:
        lower, upper = draws[column].quantile([.025, .975])
        estimates[column] = {"lower_95": float(lower), "upper_95": float(upper)}
    return estimates, draws


def reliability_table(labels, scores):
    frame = pd.DataFrame({"target": labels, "score": scores})
    frame["bin"] = pd.qcut(frame.score, q=10, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        rows=("target", "size"), positives=("target", "sum"),
        predicted=("score", "mean"), observed=("target", "mean"),
        minimum_score=("score", "min"), maximum_score=("score", "max")).reset_index()
