"""Descriptive checks for the supplied HCP extracts. No models are fitted."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["responded_strict", "responded_loose"]
FLAGS = ["followup_call_logged_flag", "territory_target_refresh_flag"]
COMMON = [
    "hcp_id", "period", "specialty", "region", "territory_id",
    "institution_type", "decile", "years_in_practice", "calls_last_90d",
    "days_since_last_call", "email_sent_last_90d", "emails_opened_last_90d",
    "samples_dropped_last_90d", "speaker_program_attended_12m",
    "digital_engagement_score", "campaign_wave_id", "patient_volume_est",
    "nbrx_last_month", "nbrx_3m_avg", "trx_3m_avg", "competitor_share_pct",
    "payer_mix_commercial_pct", "formulary_status", *FLAGS,
]
CATEGORIES = ["specialty", "region", "territory_id", "institution_type",
              "campaign_wave_id", "formulary_status"]
PHASES = ["2024-01 to 2025-06", "2025-07 to 2025-12", "2026-01 to 2026-06"]


def load_data(root=ROOT):
    raw = Path(root) / "data" / "raw"
    return tuple(pd.read_csv(raw / name) for name in [
        "hcp_engagement_train.csv", "hcp_engagement_score.csv",
        "submission_template.csv",
    ])


def check_inputs(train, score, template):
    """Fail on structural violations; report semantic concerns separately."""
    assert set(train.columns) == set(COMMON + TARGETS), "Training schema changed"
    assert set(score.columns) == set(COMMON + ["row_id"]), "Scoring schema changed"
    assert list(template.columns) == ["row_id", "score"]
    assert (len(train), len(score), len(template)) == (48585, 12723, 12723)
    assert score.row_id.is_unique and template.row_id.is_unique
    assert score.row_id.equals(template.row_id), "Submission row alignment differs"
    assert train[TARGETS].isin([0, 1]).all().all()
    assert (train.responded_strict <= train.responded_loose).all()
    for frame in [train, score]:
        assert not frame.duplicated(["hcp_id", "period"]).any()
        assert frame.period.str.fullmatch(r"\d{4}-\d{2}").all()
        pd.to_datetime(frame.period, format="%Y-%m", errors="raise")
        numbers = frame.select_dtypes("number")
        assert np.isfinite(numbers.drop(columns=FLAGS).to_numpy()).all()
        assert frame.decile.between(1, 10).all()
        for column in ["digital_engagement_score", "competitor_share_pct",
                       "payer_mix_commercial_pct"]:
            assert frame[column].between(0, 100).all(), column
        for column in FLAGS + ["speaker_program_attended_12m"]:
            assert frame[column].dropna().isin([0, 1]).all(), column
        assert not frame.drop(columns="followup_call_logged_flag").isna().any().any()
        for column in ["years_in_practice", "calls_last_90d", "days_since_last_call",
                       "email_sent_last_90d", "emails_opened_last_90d",
                       "samples_dropped_last_90d", "patient_volume_est",
                       "nbrx_last_month", "nbrx_3m_avg", "trx_3m_avg"]:
            assert frame[column].ge(0).all(), column
    assert sorted(train.period.unique()) == pd.period_range(
        "2024-01", "2025-12", freq="M").astype(str).tolist()
    assert sorted(score.period.unique()) == pd.period_range(
        "2026-01", "2026-06", freq="M").astype(str).tolist()


def build_tables(train, score, template):
    check_inputs(train, score, template)
    tables = {}
    combined = pd.concat([train.assign(dataset="train"),
                          score.assign(dataset="score")], ignore_index=True)
    combined["phase"] = np.select(
        [combined.period < "2025-07", combined.period < "2026-01"],
        PHASES[:2], default=PHASES[2])
    recent = train.loc[train.period >= "2025-07"]
    tables["structure"] = pd.DataFrame([
        {"dataset": name, "rows": len(frame), "hcps": frame.hcp_id.nunique(),
         "first_period": frame.period.min(), "last_period": frame.period.max(),
         "duplicate_hcp_months": int(frame.duplicated(["hcp_id", "period"]).sum()),
         "missing_cells": int(frame.isna().sum().sum()),
         "min_rows_per_hcp": int(frame.groupby("hcp_id").size().min()),
         "median_rows_per_hcp": float(frame.groupby("hcp_id").size().median()),
         "max_rows_per_hcp": int(frame.groupby("hcp_id").size().max())}
        for name, frame in [("train", train), ("score", score)]])
    tables["targets"] = pd.DataFrame([
        {"target": target, "rows": len(train), "positives": int(train[target].sum()),
         "prevalence": train[target].mean()} for target in TARGETS])
    tables["target_overlap"] = train.groupby(TARGETS).size().rename("rows").reset_index()
    tables["monthly"] = combined.groupby(["dataset", "period"]).agg(
        rows=("hcp_id", "size"), hcps=("hcp_id", "nunique"),
        strict_rate=("responded_strict", "mean"), loose_rate=("responded_loose", "mean"),
        strict_positives=("responded_strict", lambda s: s.sum(min_count=1)),
        followup_missing_rate=(FLAGS[0], lambda s: s.isna().mean()),
        calls_mean=("calls_last_90d", "mean"), samples_mean=("samples_dropped_last_90d", "mean"),
        recency_mean=("days_since_last_call", "mean"), nbrx_mean=("nbrx_3m_avg", "mean"),
        patient_volume_mean=("patient_volume_est", "mean"),
        preferred_share=("formulary_status", lambda s: s.eq("Preferred").mean()),
    ).reset_index().sort_values("period")
    tables["phases"] = combined.groupby("phase").agg(
        rows=("hcp_id", "size"), strict_rate=(TARGETS[0], "mean"),
        loose_rate=(TARGETS[1], "mean"), calls_mean=("calls_last_90d", "mean"),
        samples_mean=("samples_dropped_last_90d", "mean"),
        preferred_share=("formulary_status", lambda s: s.eq("Preferred").mean()),
        followup_missing_rate=(FLAGS[0], lambda s: s.isna().mean()),
    ).reset_index()
    score_cohort = np.where(score.hcp_id.isin(train.hcp_id), "seen in train", "new in score")
    tables["score_cohorts"] = score.assign(cohort=score_cohort).groupby("cohort").agg(
        rows=("hcp_id", "size"), hcps=("hcp_id", "nunique")).reset_index()
    tables["score_cohorts"]["row_share"] = tables["score_cohorts"].rows / len(score)

    numeric_rows = []
    for column in train.select_dtypes("number").columns.difference(["hcp_id", *TARGETS]):
        before, after = recent[column].dropna(), score[column].dropna()
        numeric_rows.append({
            "feature": column, "train_mean": train[column].mean(),
            "recent_train_mean": before.mean(), "score_mean": after.mean(),
            "recent_train_unique": before.nunique(), "score_unique": after.nunique(),
            "recent_train_missing": recent[column].isna().mean(),
            "score_missing": score[column].isna().mean(),
            "ks_distance_observed": ks_2samp(before, after).statistic,
            "strict_pearson_correlation_train": train[column].corr(train.responded_strict),
        })
    tables["numeric_shift"] = pd.DataFrame(numeric_rows).sort_values(
        "ks_distance_observed", ascending=False)
    category_rows, category_summary = [], []
    for column in CATEGORIES:
        shares = pd.concat([
            train[column].value_counts(normalize=True).rename("train_share"),
            recent[column].value_counts(normalize=True).rename("recent_train_share"),
            score[column].value_counts(normalize=True).rename("score_share"),
        ], axis=1).fillna(0)
        for category, row in shares.iterrows():
            category_rows.append({"feature": column, "category": category, **row.to_dict()})
        category_summary.append({
            "feature": column, "train_categories": train[column].nunique(),
            "score_categories": score[column].nunique(),
            "unseen_score_rows": int((~score[column].isin(train[column])).sum()),
            "recent_score_total_variation":
                abs(shares.recent_train_share - shares.score_share).sum() / 2,
        })
    tables["category_shares"] = pd.DataFrame(category_rows)
    tables["category_shift"] = pd.DataFrame(category_summary)
    tables["decile_mapping"] = combined.groupby(["dataset", "decile"]).agg(
        rows=("hcp_id", "size"), patient_values=("patient_volume_est", "nunique"),
        patient_value=("patient_volume_est", "first"), nbrx_values=("nbrx_3m_avg", "nunique"),
        nbrx_value=("nbrx_3m_avg", "first")).reset_index()

    pairs = train.loc[train.period == "2025-12"].merge(
        score.loc[score.period == "2026-01"], on="hcp_id", suffixes=("_train", "_score"))
    paired_rows = []
    for column in ["patient_volume_est", "nbrx_3m_avg", "nbrx_last_month", "trx_3m_avg",
                   "days_since_last_call", "calls_last_90d", "samples_dropped_last_90d"]:
        ratios = pairs[column + "_score"] / pairs[column + "_train"].replace(0, np.nan)
        paired_rows.append({"feature": column, "hcp_pairs": len(pairs),
                            "nonzero_denominators": int(ratios.notna().sum()),
                            "ratio_p10": ratios.quantile(.1), "ratio_median": ratios.median(),
                            "ratio_p90": ratios.quantile(.9)})
    tables["paired_shift"] = pd.DataFrame(paired_rows)
    tables["flag_associations"] = train.groupby(FLAGS).agg(
        rows=("hcp_id", "size"), strict_positives=(TARGETS[0], "sum"),
        strict_rate=(TARGETS[0], "mean"), loose_rate=(TARGETS[1], "mean")).reset_index()
    tables["flag_associations_by_phase"] = combined.loc[combined.dataset == "train"].groupby(
        ["phase", *FLAGS]).agg(rows=("hcp_id", "size"), strict_rate=(TARGETS[0], "mean"),
                               loose_rate=(TARGETS[1], "mean")).reset_index()
    segments = []
    for column in ["decile", "specialty", "region", "institution_type", "formulary_status"]:
        for value, group in train.groupby(column):
            segments.append({"feature": column, "value": value, "rows": len(group),
                             "strict_positives": int(group.responded_strict.sum()),
                             "strict_rate": group.responded_strict.mean(),
                             "loose_rate": group.responded_loose.mean()})
    tables["segment_targets"] = pd.DataFrame(segments)
    stability = []
    for name, frame in [("train", train), ("score", score)]:
        for column in [*CATEGORIES, "decile", "years_in_practice"]:
            counts = frame.groupby("hcp_id")[column].nunique()
            stability.append({"dataset": name, "feature": column, "hcps": len(counts),
                              "hcps_with_changes": int(counts.gt(1).sum()),
                              "max_values_per_hcp": int(counts.max())})
    tables["hcp_stability"] = pd.DataFrame(stability)
    tables["semantic_checks"] = combined.groupby("dataset").apply(
        lambda g: pd.Series({"rows": len(g), "zero_visits_recent_call": int(
            ((g.calls_last_90d == 0) & (g.days_since_last_call <= 90)).sum())}),
        include_groups=False).reset_index()
    windows = []
    for label, cutoff, start, end in [("development", "2025-05", "2025-07", "2025-09"),
                                      ("later_check", "2025-08", "2025-10", "2025-12")]:
        fitting, evaluation = train.loc[train.period <= cutoff], train.loc[train.period.between(start, end)]
        windows.append({"window": label, "fit_through": cutoff, "evaluate_from": start,
                        "evaluate_through": end, "fitting_rows": len(fitting),
                        "evaluation_rows": len(evaluation),
                        "strict_positives": int(evaluation.responded_strict.sum()),
                        "loose_positives": int(evaluation.responded_loose.sum()),
                        "unseen_hcp_rows": int((~evaluation.hcp_id.isin(fitting.hcp_id)).sum())})
    tables["proposed_windows"] = pd.DataFrame(windows)
    return tables


def make_figures(tables):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110})
    blue, orange = "#185F91", "#B04C11"
    monthly = tables["monthly"].copy()
    dates = pd.to_datetime(monthly.period)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), layout="constrained")
    historical = monthly.dataset.eq("train")
    ax.plot(dates[historical], monthly.loc[historical, "strict_rate"], "o-",
            color=blue, label="Strict: substantial increase")
    ax.plot(dates[historical], monthly.loc[historical, "loose_rate"], "s--",
            color=orange, label="Loose: any increase")
    ax.set(title="Strict response is uncommon; both labels vary by month",
           ylabel="Positive HCP-months / all HCP-months", xlabel="Row period", ylim=(0, .36))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="center left")
    fig.supxlabel("Training extract: 48,585 HCP-months, Jan 2024–Dec 2025", fontsize=9)
    figures = {"target_rates": fig}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    plot_fields = [("calls_mean", "Logged visits", "Mean visits in trailing 90 days"),
                   ("preferred_share", "Preferred formulary coverage", "Share of HCP-months"),
                   ("nbrx_mean", "New prescriptions", "Mean reported 3-month NBRx average"),
                   ("recency_mean", "Time since last visit", "Mean reported days")]
    for ax, (column, title, label) in zip(axes.flat, plot_fields):
        for dataset, color, marker in [("train", blue, "o"), ("score", orange, "s")]:
            selected = monthly.dataset.eq(dataset)
            ax.plot(dates[selected], monthly.loc[selected, column], marker + "-",
                    color=color, markersize=3, label=dataset)
        ax.axvline(pd.Timestamp("2025-07-01"), color="#777777", ls=":", lw=1)
        ax.set(title=title, ylabel=label, xlabel="Row period", ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
        if column == "preferred_share":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="x", rotation=25)
    axes[0, 0].legend()
    fig.suptitle("Changes begin in July 2025; further scale shifts appear in 2026", fontsize=14)
    fig.supxlabel("All rows in each month. Dotted line: Jul 2025. Train: 48,585; score: 12,723 HCP-months.", fontsize=9)
    figures["monthly_changes"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    deciles = tables["segment_targets"].loc[lambda d: d.feature.eq("decile")]
    axes[0].bar(deciles.value.astype(str), deciles.strict_rate, color=blue)
    axes[0].set(title="Response increases with prescribing decile", xlabel="Reported decile",
                ylabel="Strict response rate", ylim=(0, .20))
    flags = tables["flag_associations"]
    labels = [f"Follow-up {int(r[FLAGS[0]])}, refresh {int(r[FLAGS[1]])}\n(n={int(r['rows']):,})"
              for _, r in flags.iterrows()]
    axes[1].barh(labels, flags.strict_rate, color=orange)
    axes[1].set(title="Operational flags carry a strong association", xlabel="Strict response rate", xlim=(0, .72))
    axes[1].invert_yaxis()
    for i, rate in enumerate(flags.strict_rate):
        axes[1].text(rate + .012, i, f"{rate:.1%}", va="center", fontsize=10)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    fig.supxlabel("Training extract, all 48,585 HCP-months. Descriptive associations; availability and causality are unverified.", fontsize=9)
    figures["response_associations"] = fig
    return figures


def run(root=ROOT):
    tables = build_tables(*load_data(root))
    destination = Path(root) / "reports" / "eda"
    (destination / "tables").mkdir(parents=True, exist_ok=True)
    (destination / "figures").mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / "tables" / f"{name}.csv", index=False)
    figures = make_figures(tables)
    for name, figure in figures.items():
        figure.savefig(destination / "figures" / f"{name}.png", dpi=140)
        plt.close(figure)
    summary = {"status": "descriptive analysis; no fitted model or predictions",
               "comparison_reference": "Jul-Dec 2025 unless otherwise labelled",
               "targets": tables["targets"].to_dict(orient="records"),
               "scoring_cohorts": tables["score_cohorts"].to_dict(orient="records"),
               "validated": ["schema", "HCP-month uniqueness", "period coverage",
                             "numeric bounds", "target nesting", "submission row alignment"]}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(tables)} evidence tables and {len(figures)} figures to {destination}")
    return tables


if __name__ == "__main__":
    run()
