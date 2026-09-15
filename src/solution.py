"""Reproduce the delivered scores and consolidated policy comparison."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src import baselines, capacity_audit, challenger
from src.eda import check_inputs, load_data
from src.metrics import monthly_metrics, summarise_metrics
from src.modeling import split_window

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {
    "decile", "three_field_rate", "core_logistic",
    "delivered_logistic", "hcp_history_frozen",
}


def consolidate(root=ROOT):
    """Keep equal-population policy comparisons separate from sensitivities."""
    report = root / "reports/comparison"
    baseline = pd.read_csv(report / "baseline_summary.csv")
    primary = [baseline.loc[baseline.model.isin(PRIMARY)]]
    monthly = pd.read_csv(report / "baseline_monthly.csv")
    monthly = [monthly.loc[monthly.model.isin(PRIMARY)]]
    for stage in ("development", "later", "six_month"):
        for suffix, tables in (("summary", primary), ("monthly", monthly)):
            table = pd.read_csv(root / f"reports/challenger/{stage}_{suffix}.csv")
            tables.append(table.loc[table.model.eq("extended_boosting")].assign(stage=stage))
    pd.concat(primary, ignore_index=True).sort_values(
        ["stage", "model", "capacity"]
    ).to_csv(report / "summary.csv", index=False)
    pd.concat(monthly, ignore_index=True).sort_values(
        ["stage", "model", "period", "capacity"]
    ).to_csv(report / "monthly.csv", index=False)


def verify_development_selection():
    """Reproduce development metrics in memory against the recorded selection."""
    settings = challenger.config()
    selection = challenger.read_selection(settings)
    train, score, template = load_data(ROOT)
    check_inputs(train, score, template)
    fitting, evaluation = split_window(
        train, settings["development"], horizon=settings["label_horizon_months"]
    )
    specifications = challenger.specifications(settings)
    metrics = []
    for name, specification in {
        "decile": challenger.baseline_spec(), **specifications,
    }.items():
        scores, model = challenger.score_spec(
            fitting, evaluation, settings["target"], specification, settings
        )
        metrics.append(monthly_metrics(
            evaluation, settings["target"], scores, name,
            settings["capacity_scenarios"], settings["seed"], model is not None,
        ))
    vectors = challenger.metric_vectors(summarise_metrics(pd.concat(metrics, ignore_index=True)))
    recorded = pd.read_csv(ROOT / "reports/challenger/development_metric_vectors.csv")
    actual = vectors.set_index("model").sort_index()[challenger.VECTOR]
    expected = recorded.set_index("model").sort_index()[challenger.VECTOR]
    if not actual.index.equals(expected.index):
        raise ValueError("Development candidates differ from the recorded selection")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    decision = challenger.choose_spec(vectors, specifications, settings["comparison_epsilon"])
    if decision != selection["decision"]:
        raise ValueError("Reproduced development metrics imply a different model selection")
    print("Verified development metrics and recorded model selection without rewriting them.")


def main():
    subprocess.run([sys.executable, "scripts/verify_inputs.py"], cwd=ROOT, check=True)
    verify_development_selection()
    # Each command completes before its outputs are read by the next stage.
    for command in (
        [sys.executable, "-m", "src.eda"],
        [sys.executable, "-m", "src.feature_audit"],
        [sys.executable, "-m", "src.challenger", "--stage", "evaluate"],
        [sys.executable, "-m", "src.challenger", "--stage", "score"],
        [sys.executable, "-m", "src.decision_audit"],
        [sys.executable, "scripts/activity_sensitivity.py"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)
    metadata_path = ROOT / "outputs/submission_final_metadata.json"
    metadata = challenger.read_json(metadata_path)
    metadata.pop("previous_submissions_preserved", None)
    challenger.write_json(metadata_path, metadata)
    subprocess.run(
        [sys.executable, "scripts/verify_delivery_data.py"], cwd=ROOT, check=True
    )
    baselines.run(ROOT)
    consolidate()
    capacity_audit.run(ROOT)
    print("Consolidated evidence: reports/comparison/summary.csv")
    print("Predictions: outputs/submission_final.csv")


if __name__ == "__main__":
    main()
