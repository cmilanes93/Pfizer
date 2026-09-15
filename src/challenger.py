"""Fixed capacity and algorithm comparison; all historical checks are exploratory."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.eda import check_inputs, load_data
from src.metrics import (monthly_metrics, paired_hcp_bootstrap, ranking_order,
                         reliability_table, summarise_metrics)
from src.modeling import (EXCLUDED, FLAGS, TARGETS, decile_scores, fit_logistic,
                          predict, split_window, tie_keys, validate_submission)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/challenger"
OUTPUT = ROOT / "outputs"
VECTOR = ["average_precision", "precision_05", "precision_10", "precision_20"]
PROVENANCE_FILES = {
    "config": "config/challenger.json", "protocol": "docs/methodology/challenger_protocol.md",
    "feature_sets": "config/revision.json", "training": "data/raw/hcp_engagement_train.csv",
    "implementation": "src/challenger.py", "modeling": "src/modeling.py",
    "metrics": "src/metrics.py", "input_checks": "src/eda.py",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # Explicit LF keeps frozen file hashes stable through Git on Windows.
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return read_json(ROOT / "config/challenger.json")


def fingerprints(root=ROOT):
    return {name: digest(root / relative) for name, relative in PROVENANCE_FILES.items()}


def specifications(settings, root=ROOT):
    features = read_json(root / "config/revision.json")["eligible_candidate_sets"]
    specs = {
        "core_logistic": {"name": "core_logistic", "algorithm": "logistic", "features": features["core"], "parameters": {"C": settings["logistic_C"]}},
        "extended_logistic": {"name": "extended_logistic", "algorithm": "logistic", "features": features["extended_without_speaker"], "parameters": {"C": settings["logistic_C"]}},
        "extended_boosting": {"name": "extended_boosting", "algorithm": "boosting", "features": features["extended_without_speaker"], "parameters": dict(settings["boosting"], categorical_features=None)},
    }
    for spec in specs.values():
        validate_features(spec["features"])
    return specs


def validate_features(features):
    forbidden = set(features) & (TARGETS | FLAGS | EXCLUDED | {"speaker_program_attended_12m"})
    if forbidden:
        raise ValueError(f"Forbidden challenger predictors: {sorted(forbidden)}")


def fit_boosting(fitting, target, spec):
    validate_features(spec["features"])
    if target not in TARGETS:
        raise ValueError("Unsupported response definition")
    features = spec["features"]
    if fitting[features].isna().any().any():
        raise ValueError("Selected predictors contain missing values")
    categorical = fitting[features].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [feature for feature in features if feature not in categorical]
    model = Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ])),
        ("classifier", HistGradientBoostingClassifier(**spec["parameters"])),
    ])
    return model.fit(fitting[features], fitting[target])


def score_spec(fitting, evaluation, target, spec, settings):
    if spec["algorithm"] == "decile":
        return decile_scores(evaluation), None
    validate_features(spec["features"])
    if spec["algorithm"] == "logistic":
        model = fit_logistic(fitting, target, spec["features"], spec["parameters"]["C"], settings["seed"])
    elif spec["algorithm"] == "boosting":
        model = fit_boosting(fitting, target, spec)
    else:
        raise ValueError("Unknown challenger algorithm")
    return predict(model, evaluation), model


def baseline_spec():
    return {"name": "decile", "algorithm": "decile", "features": ["decile"], "parameters": {}}


def dominates(left, right, epsilon):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return bool(np.all(left >= right - epsilon) and np.any(left > right + epsilon))


def choose_spec(vectors, specs, epsilon=1e-12):
    """Admissibility, Pareto filtering, then the protocol's simplicity ordering."""
    table = vectors.set_index("model")
    if not np.isfinite(table[VECTOR].to_numpy(dtype=float)).all():
        raise ValueError("Selection vector contains missing or non-finite metrics")
    admissible = [name for name in specs if dominates(table.loc[name, VECTOR], table.loc["decile", VECTOR], epsilon)]
    frontier = [name for name in admissible if not any(
        other != name and dominates(table.loc[other, VECTOR], table.loc[name, VECTOR], epsilon)
        for other in admissible)]
    if not frontier:
        selected = baseline_spec()
    else:
        name = min(frontier, key=lambda candidate: (
            len(specs[candidate]["features"]), specs[candidate]["algorithm"] != "logistic",
            -float(table.loc[candidate, "average_precision"]), candidate))
        selected = specs[name]
    return {"selected": selected, "admissible": sorted(admissible), "nondominated": sorted(frontier)}


def metric_vectors(summary):
    rows = []
    for model, group in summary.groupby("model", sort=True):
        row = {"model": model, "average_precision": float(group.average_precision.iloc[0])}
        for capacity, label in [(.05, "05"), (.1, "10"), (.2, "20")]:
            row[f"precision_{label}"] = float(group.loc[group.capacity.eq(capacity), "precision"].item())
        rows.append(row)
    return pd.DataFrame(rows)


def old_rule_sensitivity(vectors, specs, settings):
    table = vectors.set_index("model")
    best = max(float(table.loc[name, "precision_10"]) for name in specs)
    rows = []
    for tolerance in settings["old_tolerance_scenarios"]:
        candidates = [name for name in specs if table.loc[name, "precision_10"] >= best - tolerance - settings["comparison_epsilon"]
                      and table.loc[name, "precision_10"] > table.loc["decile", "precision_10"] + settings["comparison_epsilon"]]
        chosen = min(candidates, key=lambda name: (len(specs[name]["features"]), specs[name]["algorithm"] != "logistic", name)) if candidates else "decile"
        rows.append({"simplicity_tolerance": tolerance, "selected": chosen,
                     "precision_10": float(table.loc[chosen, "precision_10"]), "used_for_new_selection": False})
    return rows


def capacity_curve(evaluation, target, scores, name, settings):
    frame = evaluation[["hcp_id", "period", target]].copy()
    frame["score"], frame["tie"] = scores, tie_keys(frame, settings["seed"])
    rows = []
    for period, group in frame.groupby("period", sort=True):
        y = group[target].to_numpy()
        order = ranking_order(group.score, group.tie.to_numpy())
        positives, cumulative = int(y.sum()), np.cumsum(y[order])
        for capacity in settings["curve_capacities"]:
            k = math.ceil(capacity * len(group))
            found = int(cumulative[k - 1])
            rows.append({"model": name, "period": period, "target": target, "capacity": capacity,
                         "rows": len(group), "selected": k, "true_positives_selected": found,
                         "precision": found / k, "recall": found / positives if positives else np.nan,
                         "prevalence": positives / len(group),
                         "lift": (found / k) / (positives / len(group)) if positives else np.nan})
    return pd.DataFrame(rows)


def curve_summary(curves):
    return curves.groupby(["model", "target", "capacity"], as_index=False).agg(
        months=("period", "nunique"), rows=("rows", "sum"), selected=("selected", "sum"),
        true_positives_selected=("true_positives_selected", "sum"), precision=("precision", "mean"),
        precision_min=("precision", "min"), precision_max=("precision", "max"),
        recall=("recall", "mean"), prevalence=("prevalence", "mean"), lift=("lift", "mean"))


def curve_comparisons(summary, epsilon):
    table = summary.pivot(index="capacity", columns="model", values="precision")
    comparisons = []
    for left, right in combinations(table.columns, 2):
        delta = table[left] - table[right]
        crossings, previous = [], None
        for capacity, value in delta.items():
            sign = 1 if value > epsilon else -1 if value < -epsilon else 0
            if sign and previous and previous[1] != sign:
                crossings.append({"from_capacity": float(previous[0]), "to_capacity": float(capacity)})
            if sign:
                previous = (capacity, sign)
        comparisons.append({"left": left, "right": right, "crossing_intervals": crossings,
                            "left_higher_grid_points": int((delta > epsilon).sum()),
                            "right_higher_grid_points": int((delta < -epsilon).sum()),
                            "tied_grid_points": int((delta.abs() <= epsilon).sum())})
    return {"scope": "Mean monthly precision on the 1%-100% display grid; not dominance at every possible list size", "pairs": comparisons}


def compare_window(train, settings, window, prefix):
    fitting, evaluation = split_window(train, window, horizon=settings["label_horizon_months"])
    target = settings["target"]
    metrics, curves = [], []
    predictions = evaluation[["hcp_id", "period", target]].copy()
    for name, spec in {"decile": baseline_spec(), **specifications(settings)}.items():
        scores, model = score_spec(fitting, evaluation, target, spec, settings)
        predictions[name] = scores
        metrics.append(monthly_metrics(evaluation, target, scores, name, settings["capacity_scenarios"], settings["seed"], model is not None))
        curves.append(capacity_curve(evaluation, target, scores, name, settings))
    monthly, curves = pd.concat(metrics, ignore_index=True), pd.concat(curves, ignore_index=True)
    summary, curve_table = summarise_metrics(monthly), curve_summary(curves)
    for suffix, table in [("monthly", monthly), ("summary", summary), ("predictions", predictions),
                          ("curve_monthly", curves), ("curve_summary", curve_table), ("metric_vectors", metric_vectors(summary))]:
        table.to_csv(REPORT / f"{prefix}_{suffix}.csv", index=False)
    write_json(REPORT / f"{prefix}_curve_comparisons.json", curve_comparisons(curve_table, settings["comparison_epsilon"]))
    return summary, predictions, evaluation


def develop(train, settings):
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    freeze = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprints": fingerprints(),
              "source_commit": git.stdout.strip() if git.returncode == 0 else None,
              "analysis_status": settings["analysis_status"]}
    write_json(REPORT / "implementation_freeze.json", freeze)
    summary, _, _ = compare_window(train, settings, settings["development"], "development")
    vectors, specs = metric_vectors(summary), specifications(settings)
    decision = choose_spec(vectors, specs, settings["comparison_epsilon"])
    selection = {"decision": decision, "selected": decision["selected"],
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                 "analysis_status": settings["analysis_status"], "fingerprints": fingerprints(),
                 "implementation_freeze_sha256": digest(REPORT / "implementation_freeze.json"),
                 "development_vectors_sha256": digest(REPORT / "development_metric_vectors.csv"),
                 "metric_vectors": json.loads(vectors.to_json(orient="records", double_precision=15)),
                 "later_periods_previously_examined": True, "new_later_metrics_used_for_selection": False,
                 "selection_rule": settings["selection_rule"]}
    write_json(REPORT / "selection.json", selection)
    write_json(REPORT / "old_rule_sensitivity.json", old_rule_sensitivity(vectors, specs, settings))
    print("EXPLORATORY DEVELOPMENT", flush=True)
    print(vectors.round(6).to_string(index=False), flush=True)
    print("FIXED SELECTION:", decision, flush=True)


def read_selection(settings, report=REPORT, root=ROOT):
    selection = read_json(report / "selection.json")
    if selection["fingerprints"] != fingerprints(root):
        raise ValueError("Source or configuration changed after challenger selection")
    if selection["implementation_freeze_sha256"] != digest(report / "implementation_freeze.json"):
        raise ValueError("Pre-fit implementation freeze changed")
    if read_json(report / "implementation_freeze.json")["fingerprints"] != selection["fingerprints"]:
        raise ValueError("Implementation fingerprints changed between freeze and selection")
    if selection["development_vectors_sha256"] != digest(report / "development_metric_vectors.csv"):
        raise ValueError("Development selection vectors changed")
    decision = choose_spec(pd.read_csv(report / "development_metric_vectors.csv"), specifications(settings, root), settings["comparison_epsilon"])
    if decision != selection["decision"] or decision["selected"] != selection["selected"]:
        raise ValueError("Saved model does not follow the recorded selection rule")
    return selection


def evaluate(train, settings, selection):
    spec, target = selection["selected"], settings["target"]
    later, predictions, evaluation = compare_window(train, settings, settings["later"], "later")
    six, _, _ = compare_window(train, settings, settings["six_month"], "six_month")
    selected_scores, baseline = predictions[spec["name"]].to_numpy(), predictions.decile.to_numpy()
    bootstrap = {}
    for capacity in settings["capacity_scenarios"]:
        intervals, draws = paired_hcp_bootstrap(evaluation, target, selected_scores, baseline,
                                               capacity, settings["seed"], settings["bootstrap_replicates"])
        key = f"{capacity:g}"
        draws.to_csv(REPORT / f"bootstrap_draws_{int(round(capacity * 100)):02d}.csv", index=False)
        primary = later.loc[later.capacity.eq(capacity)].set_index("model")
        bootstrap[key] = {"replicates": settings["bootstrap_replicates"], "confidence": .95,
                          "mean_precision_difference": float(primary.loc[spec["name"], "precision"] - primary.loc["decile", "precision"]),
                          "intervals": intervals}
    write_json(REPORT / "bootstrap.json", {
        "scope": "Paired whole-HCP resampling conditional on fixed fitted models and observed later months; intervals are not simultaneous across capacities or forecasts for 2026",
        "capacities": bootstrap})

    fitting, _ = split_window(train, settings["later"], horizon=settings["label_horizon_months"])
    lag_fit, lag_eval = split_window(train, settings["later"], extra_lag=1, horizon=settings["label_horizon_months"])
    hcps = np.sort(train.hcp_id.unique())
    held_ids = np.random.default_rng(settings["seed"]).choice(hcps, int(np.ceil(len(hcps) * .2)), replace=False)
    held_fit, held_eval = fitting.loc[~fitting.hcp_id.isin(held_ids)], evaluation.loc[evaluation.hcp_id.isin(held_ids)]
    if set(held_fit.hcp_id) & set(held_eval.hcp_id):
        raise ValueError("Withheld HCPs entered fitting")
    sensitivity = []
    for name, fit, ev, label in [("extra_label_delay", lag_fit, lag_eval, target),
                                ("loose", fitting, evaluation, "responded_loose"),
                                ("unfamiliar_hcp", held_fit, held_eval, target)]:
        scores, model = score_spec(fit, ev, label, spec, settings)
        sensitivity.extend([
            monthly_metrics(ev, label, scores, name + "_selected", settings["capacity_scenarios"], settings["seed"], model is not None),
            monthly_metrics(ev, label, decile_scores(ev), name + "_decile", settings["capacity_scenarios"], settings["seed"], False)])
        audit = ev[["hcp_id", "period", label]].copy()
        audit["selected_score"], audit["decile_score"] = scores, decile_scores(ev)
        audit.to_csv(REPORT / f"{name}_predictions.csv", index=False)
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    sensitivity.to_csv(REPORT / "sensitivity_monthly.csv", index=False)
    summarise_metrics(sensitivity).to_csv(REPORT / "sensitivity_summary.csv", index=False)
    pd.DataFrame({"hcp_id": np.sort(held_ids)}).to_csv(REPORT / "stress_held_hcps.csv", index=False)
    if spec["algorithm"] != "decile":
        reliability_table(evaluation[target].to_numpy(), selected_scores).to_csv(REPORT / "reliability.csv", index=False)
    monthly = pd.read_csv(REPORT / "later_monthly.csv")
    improved = {}
    for capacity in settings["capacity_scenarios"]:
        pivot = monthly.loc[monthly.capacity.eq(capacity)].pivot(index="period", columns="model", values="precision")
        improved[f"{capacity:g}"] = int((pivot[spec["name"]] > pivot.decile + settings["comparison_epsilon"]).sum())
    results = {"analysis_status": settings["analysis_status"], "selected": spec,
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
               "selection_sha256": digest(REPORT / "selection.json"),
               "selection_preceded_new_evaluation": selection["selected_at_utc"],
               "months_evaluated": int(evaluation.period.nunique()), "months_improved_by_capacity": improved,
               "production_performance_established": False, "causal_effect_established": False,
               "six_month_check_independent": False, "monthly_inputs_may_change_while_model_frozen": True,
               "outcome_interpretation": "Supplied strict/loose labels; next-period construction remains unverified and subject to the separate temporal audit"}
    write_json(REPORT / "evaluation.json", results)
    print("EXPLORATORY LATER VECTORS", flush=True)
    print(metric_vectors(later).round(6).to_string(index=False), flush=True)
    print("SIX-MONTH VECTORS", flush=True)
    print(metric_vectors(six).round(6).to_string(index=False), flush=True)


def score_final(train, score, template, settings, selection):
    cutoff = pd.Period(settings["scoring_fit_through"], freq="M")
    if cutoff + settings["label_horizon_months"] >= pd.Period(score.period.min(), freq="M"):
        raise ValueError("Final fitting includes unavailable labels")
    fitting = train.loc[train.period <= str(cutoff)]
    values, model = score_spec(fitting, score, settings["target"], selection["selected"], settings)
    submission = pd.DataFrame({"row_id": score.row_id, "score": values})
    validate_submission(submission, template)
    path = OUTPUT / "submission_final.csv"
    submission.to_csv(path, index=False, float_format="%.10f")
    validate_submission(pd.read_csv(path), template)
    metadata = {"analysis_status": settings["analysis_status"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "target": settings["target"], "specification": selection["selected"],
                "training_row_cutoff": str(cutoff), "training_rows": len(fitting),
                "scoring_rows": len(submission), "scoring_periods": sorted(score.period.unique()),
                "model_frozen_across_scoring_periods": True, "monthly_inputs_required_at_each_ranking": True,
                "score_meaning": "Uncalibrated model response score used for ranking" if model is not None else "Ordinal decile ranking; not a probability",
                "submission_sha256": digest(path), "selection_sha256": digest(REPORT / "selection.json"),
                "fingerprints": fingerprints(), "scoring_sha256": digest(ROOT / "data/raw/hcp_engagement_score.csv"),
                "production_feature_availability_verified": False, "target_temporal_construction_verified": False,
                "selection_made_during_retrospective_exercise": True, "previous_submissions_preserved": True}
    write_json(OUTPUT / "submission_final_metadata.json", metadata)
    if model is not None and selection["selected"]["algorithm"] == "logistic":
        classifier = model.named_steps["classifier"]
        coefficients = pd.DataFrame({"term": model.named_steps["preprocessing"].get_feature_names_out(), "coefficient": classifier.coef_[0]})
        coefficients.loc[len(coefficients)] = ["intercept", classifier.intercept_[0]]
        coefficients.to_csv(REPORT / "final_coefficients.csv", index=False)
    print(f"Validated {len(submission):,} final scoring rows; prior deliveries preserved.", flush=True)


def records(path):
    return json.loads(pd.read_csv(path).to_json(orient="records", double_precision=15))


def presentation_evidence(settings, selection):
    evidence = {"analysis_status": settings["analysis_status"], "selected": selection["selected"],
                "models": specifications(settings), "selection": selection,
                "capacity_scenarios": settings["capacity_scenarios"],
                "curve_resolution": "One-percentage-point display increments; not an operational capacity policy",
                "old_rule_sensitivity": read_json(REPORT / "old_rule_sensitivity.json"),
                "sensitivities": records(REPORT / "sensitivity_summary.csv"),
                "bootstrap": read_json(REPORT / "bootstrap.json"),
                "evaluation": read_json(REPORT / "evaluation.json"),
                "scoring": read_json(OUTPUT / "submission_final_metadata.json")}
    winners = {}
    for stage in ["development", "later", "six_month"]:
        evidence[stage] = records(REPORT / f"{stage}_summary.csv")
        evidence[f"{stage}_monthly"] = records(REPORT / f"{stage}_monthly.csv")
        evidence[f"{stage}_curves"] = records(REPORT / f"{stage}_curve_summary.csv")
        evidence[f"{stage}_curve_comparisons"] = read_json(REPORT / f"{stage}_curve_comparisons.json")
        vectors = pd.read_csv(REPORT / f"{stage}_metric_vectors.csv").set_index("model")
        winners[stage] = {metric: {"value": float(vectors[metric].max()),
                                 "models": vectors.index[vectors[metric] >= vectors[metric].max() - settings["comparison_epsilon"]].tolist()}
                          for metric in VECTOR}
    evidence["metric_winners"] = winners
    write_json(REPORT / "presentation_data.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["develop", "evaluate", "score", "all"], default="all")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    settings = config()
    train, score, template = load_data(ROOT)
    check_inputs(train, score, template)
    if args.stage in ["develop", "all"]:
        develop(train, settings)
    selection = read_selection(settings)
    if args.stage in ["evaluate", "all"]:
        evaluate(train, settings, selection)
    if args.stage in ["score", "all"]:
        score_final(train, score, template, settings, selection)
    if args.stage == "all":
        presentation_evidence(settings, selection)


if __name__ == "__main__":
    main()
