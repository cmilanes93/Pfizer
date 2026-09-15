"""Four exploratory decision checks; no fitting, relabelling or score changes."""

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.metrics import ranking_order
from src.modeling import tie_keys


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["responded_strict", "responded_loose"]
MODELS = ["decile", "core_logistic", "extended_logistic"]
CAPACITIES = [.05, .1, .2]
PROFILE = ["decile", "specialty", "formulary_status"]


def temporal_target_check(train):
    """Conditionally compare NBRx(t+1) with NBRx(t), using t+2/t+1 rows.

    The supplied label baseline is unknown. This is not label reconstruction.
    Only consecutive records within this one training extract are admissible.
    """
    if train.duplicated(["hcp_id", "period"]).any():
        raise ValueError("Temporal check requires unique HCP-month records")
    work = train[["hcp_id", "period", "nbrx_last_month", *TARGETS]].copy()
    work["csv_line"] = range(2, len(work) + 2)
    work = work.sort_values(["hcp_id", "period"])
    work["month"] = pd.PeriodIndex(work.period, freq="M").asi8
    groups = work.groupby("hcp_id", sort=False)
    for lead in (1, 2):
        for column in ("month", "period", "nbrx_last_month", "csv_line"):
            work[f"{column}_lead{lead}"] = groups[column].shift(-lead)
    work["within_extract_horizon"] = work.month <= work.month.max() - 2
    work["checked"] = ((work.month_lead1 - work.month == 1)
                       & (work.month_lead2 - work.month == 2))
    checked = work.loc[work.checked].copy()
    checked["conditional_nbrx_change"] = (checked.nbrx_last_month_lead2
                                           - checked.nbrx_last_month_lead1)
    checked["direction"] = np.select(
        [checked.conditional_nbrx_change.gt(0), checked.conditional_nbrx_change.lt(0)],
        ["up", "down"], default="equal")
    parts = []
    for target in TARGETS:
        for period, subset in [("ALL", checked), *list(checked.groupby("period"))]:
            table = subset.groupby([target, "direction"]).size().rename("rows").reset_index()
            table = table.rename(columns={target: "label_value"})
            table["target"] = target
            table["period"] = period
            table["share_within_label"] = table.rows / table.groupby("label_value").rows.transform("sum")
            parts.append(table)
    coverage = work.groupby("period", as_index=False).agg(
        input_rows=("hcp_id", "size"), within_extract_horizon=("within_extract_horizon", "sum"),
        checked_rows=("checked", "sum"), strict_positives=("responded_strict", "sum"),
        loose_positives=("responded_loose", "sum"))
    covered_labels = checked.groupby("period", as_index=False)[TARGETS].sum().rename(
        columns={"responded_strict": "checked_strict_positives", "responded_loose": "checked_loose_positives"})
    coverage = coverage.merge(covered_labels, on="period", how="left", validate="one_to_one")
    coverage[["checked_strict_positives", "checked_loose_positives"]] = coverage[
        ["checked_strict_positives", "checked_loose_positives"]].fillna(0).astype(int)
    coverage["coverage_fraction"] = coverage.checked_rows / coverage.input_rows
    summary = {
        "input_rows": len(train), "within_extract_horizon": int(work.within_extract_horizon.sum()),
        "checked_rows": len(checked), "checked_hcps": int(checked.hcp_id.nunique()),
        "excluded_end_of_extract_rows": int((~work.within_extract_horizon).sum()),
        "missing_consecutive_records_within_horizon": int((work.within_extract_horizon & ~work.checked).sum()),
        "checked_label_period_start": checked.period.min() if len(checked) else None,
        "checked_label_period_end": checked.period.max() if len(checked) else None,
        "condition": "If following period means t+1, baseline means t, and row t+1/t+2 nbrx_last_month measure t/t+1",
        "does_not_reconstruct_labels_or_prove_label_error": True,
    }
    keep = ["hcp_id", "period", *TARGETS, "period_lead1", "period_lead2",
            "nbrx_last_month_lead1", "nbrx_last_month_lead2", "conditional_nbrx_change",
            "direction", "csv_line", "csv_line_lead1", "csv_line_lead2"]
    return summary, pd.concat(parts, ignore_index=True), coverage, checked[keep]


def select_monthly(predictions, models=MODELS, capacities=CAPACITIES, seed=42):
    """Reuse the existing capacity rounding, ranking order and HCP-month ties."""
    selected = []
    for period, group in predictions.groupby("period", sort=True):
        ties = tie_keys(group, seed)
        for model in models:
            order = ranking_order(group[model], ties)
            for capacity in capacities:
                k = math.ceil(capacity * len(group))
                rows = group.iloc[order[:k]][["hcp_id", "period", "responded_strict"]].copy()
                rows["model"] = model
                rows["capacity"] = capacity
                rows["rank"] = range(1, k + 1)
                rows["score"] = group.iloc[order[:k]][model].to_numpy()
                rows["eligible_monthly_rows"] = len(group)
                selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def selection_diagnostics(selected):
    summaries, frequencies, turnover, overlaps = [], [], [], []
    for (model, capacity), group in selected.groupby(["model", "capacity"]):
        frequency = group.groupby("hcp_id").size()
        counts = frequency.value_counts().sort_index()
        for count, hcps in counts.items():
            frequencies.append({"model": model, "capacity": capacity,
                                "months_selected": int(count), "hcps": int(hcps)})
        summaries.append({"model": model, "capacity": capacity, "months": group.period.nunique(),
                          "selection_slots": len(group), "unique_selected_hcps": frequency.size,
                          "hcps_selected_more_than_once": int(frequency.gt(1).sum()),
                          "repeat_slots_after_first_selection": len(group) - frequency.size,
                          "observed_positive_selections": int(group.responded_strict.sum())})
        previous_period, previous_set = None, set()
        for period, month in group.groupby("period", sort=True):
            current = set(month.hcp_id)
            if previous_period is not None:
                if pd.Period(period, freq="M").ordinal - pd.Period(previous_period, freq="M").ordinal != 1:
                    raise ValueError("Turnover requires consecutive evaluation months")
                shared = len(current & previous_set)
                turnover.append({"model": model, "capacity": capacity, "previous_period": previous_period,
                                 "period": period, "previous_selected": len(previous_set),
                                 "current_selected": len(current), "retained": shared,
                                 "new_to_monthly_list": len(current - previous_set),
                                 "removed_from_monthly_list": len(previous_set - current),
                                 "share_current_also_selected_previous_month": shared / len(current),
                                 "jaccard": shared / len(current | previous_set)})
            previous_period, previous_set = period, current
    for (period, capacity), group in selected.groupby(["period", "capacity"]):
        sets = {model: set(part.hcp_id) for model, part in group.groupby("model")}
        labels = group.drop_duplicates("hcp_id").set_index("hcp_id").responded_strict
        for left, right in combinations(sorted(sets), 2):
            a, b = sets[left], sets[right]
            only_a, only_b, shared = a - b, b - a, a & b
            overlaps.append({"period": period, "capacity": capacity, "left_model": left,
                             "right_model": right, "left_selected": len(a), "right_selected": len(b),
                             "shared_selected": len(shared), "left_only_selected": len(only_a),
                             "right_only_selected": len(only_b),
                             "shared_positives": int(labels.loc[sorted(shared)].sum()),
                             "left_only_positives": int(labels.loc[sorted(only_a)].sum()),
                             "right_only_positives": int(labels.loc[sorted(only_b)].sum()),
                             "jaccard": len(shared) / len(a | b)})
    monthly = selected.groupby(["model", "capacity", "period"], as_index=False).agg(
        selected=("hcp_id", "size"), observed_positive_selections=("responded_strict", "sum"),
        eligible_rows=("eligible_monthly_rows", "first"))
    return pd.DataFrame(summaries), pd.DataFrame(frequencies), pd.DataFrame(turnover), pd.DataFrame(overlaps), monthly


def decile_tie_expectation(predictions, capacities=CAPACITIES, seed=42):
    """Exact expected precision under uniform random choice at boundary ties."""
    rows = []
    for period, group in predictions.groupby("period", sort=True):
        order = ranking_order(group.decile, tie_keys(group, seed))
        for capacity in capacities:
            k = math.ceil(capacity * len(group))
            threshold = group.iloc[order[k - 1]].decile
            above, tied = group[group.decile > threshold], group[group.decile == threshold]
            boundary_slots = k - len(above)
            expected = above.responded_strict.sum() + boundary_slots * tied.responded_strict.mean()
            rows.append({"period": period, "capacity": capacity, "selected": k,
                         "boundary_score": threshold, "strictly_above_rows": len(above),
                         "tied_rows": len(tied), "tied_positives": int(tied.responded_strict.sum()),
                         "slots_from_boundary_ties": boundary_slots,
                         "expected_positives_random_ties": expected,
                         "expected_precision_random_ties": expected / k,
                         "actual_precision_fixed_seed": group.iloc[order[:k]].responded_strict.mean()})
    return pd.DataFrame(rows)


def profile_support(train, score):
    """Exact joint-profile support; zero means absent, with no rare-cell cutoff."""
    score = score.copy()
    score["cohort"] = np.where(score.hcp_id.isin(train.hcp_id.unique()), "known_hcp", "new_hcp")
    profiles, summaries, monthly = [], [], []
    references = {"all_supplied_training": train,
                  "final_fit_through_2025-11": train[train.period <= "2025-11"]}
    for reference, history in references.items():
        support = history.groupby(PROFILE, as_index=False).agg(
            historical_rows=("hcp_id", "size"), historical_hcps=("hcp_id", "nunique"),
            historical_strict_positives=("responded_strict", "sum"))
        joined = score.merge(support, on=PROFILE, how="left", validate="many_to_one")
        for count in ["historical_rows", "historical_hcps", "historical_strict_positives"]:
            joined[count] = joined[count].fillna(0).astype(int)
        joined["profile_seen"] = joined.historical_rows.gt(0)
        detail = joined.groupby(["cohort", *PROFILE], as_index=False).agg(
            scoring_rows=("hcp_id", "size"), scoring_hcps=("hcp_id", "nunique"),
            historical_rows=("historical_rows", "first"), historical_hcps=("historical_hcps", "first"),
            historical_strict_positives=("historical_strict_positives", "first"),
            profile_seen=("profile_seen", "first"))
        detail["reference"] = reference
        profiles.append(detail)
        for cohort, group in joined.groupby("cohort"):
            summaries.append({"reference": reference, "cohort": cohort, "scoring_rows": len(group),
                              "scoring_hcps": group.hcp_id.nunique(),
                              "unseen_profile_rows": int((~group.profile_seen).sum()),
                              "unseen_profile_row_fraction": (~group.profile_seen).mean(),
                              "unseen_profile_hcps": group.loc[~group.profile_seen, "hcp_id"].nunique(),
                              "distinct_scoring_profiles": len(group[PROFILE].drop_duplicates()),
                              "minimum_profile_historical_rows": int(group.historical_rows.min()),
                              "median_profile_historical_rows_across_scoring_rows": group.historical_rows.median(),
                              "minimum_profile_historical_hcps": int(group.historical_hcps.min())})
        months = joined.groupby(["cohort", "period", "profile_seen"], as_index=False).agg(
            scoring_rows=("hcp_id", "size"), scoring_hcps=("hcp_id", "nunique"))
        months["reference"] = reference
        monthly.append(months)
    return pd.concat(profiles, ignore_index=True), pd.DataFrame(summaries), pd.concat(monthly, ignore_index=True)


def main(root=ROOT):
    root = Path(root)
    output = root / "reports/decision_audit"
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "data/raw/hcp_engagement_train.csv")
    score = pd.read_csv(root / "data/raw/hcp_engagement_score.csv")
    summary, cross, coverage, rows = temporal_target_check(train)
    cross.to_csv(output / "target_direction.csv", index=False)
    coverage.to_csv(output / "target_coverage.csv", index=False)
    rows.to_csv(output / "target_checked_rows.csv", index=False)
    examples = rows.groupby([*TARGETS, "direction"], sort=True).head(3)
    examples.to_csv(output / "target_examples.csv", index=False)
    (output / "target_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TARGET SUMMARY", json.dumps(summary), flush=True)
    print(cross[cross.period == "ALL"].to_string(index=False), flush=True)
    tables = {name: [] for name in ["selection_summary", "selection_frequency", "selection_turnover",
                                   "policy_overlap", "selection_monthly", "decile_tie_expectation"]}
    input_paths = ["data/raw/hcp_engagement_train.csv", "data/raw/hcp_engagement_score.csv"]
    reconciliation = {}
    for horizon in ["later", "six_month"]:
        source = f"reports/challenger/{horizon}_predictions.csv"
        input_paths.append(source)
        predictions = pd.read_csv(root / source)
        selected = select_monthly(predictions)
        selected.to_csv(output / f"{horizon}_selected_rows.csv", index=False)
        diagnostics = selection_diagnostics(selected)
        metric_source = f"reports/challenger/{horizon}_monthly.csv"
        input_paths.append(metric_source)
        existing = pd.read_csv(root / metric_source)
        compared = diagnostics[4].merge(existing, on=["model", "capacity", "period"],
                                        validate="one_to_one", suffixes=("_audit", "_existing"))
        if (len(compared) != len(diagnostics[4])
                or not compared.selected_audit.eq(compared.selected_existing).all()
                or not compared.observed_positive_selections.eq(compared.true_positives_selected).all()):
            raise ValueError("Selection audit disagrees with model evaluation metrics")
        reconciliation[horizon] = {"policy_month_capacity_rows": len(compared),
                                   "selection_counts_and_positive_counts_match": True}
        for name, table in zip(list(tables)[:5], diagnostics):
            table["horizon"] = horizon
            tables[name].append(table)
        ties = decile_tie_expectation(predictions)
        ties["horizon"] = horizon
        tables["decile_tie_expectation"].append(ties)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    for name, table in zip(["profile_support", "profile_summary", "profile_monthly"], profile_support(train, score)):
        table.to_csv(output / f"{name}.csv", index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "Exploratory target and ranking diagnostics using fixed evaluation predictions",
        "sources_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in input_paths},
        "capacities_are_scenarios_not_business_requirements": CAPACITIES,
        "seed": 42, "target_assumptions": summary["condition"],
        "reconciliation_to_evaluation_metrics": reconciliation,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Saved four diagnostic families to", output, flush=True)


if __name__ == "__main__":
    main()
