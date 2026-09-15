"""Post-evaluation check of the dictionary's rolling attendance definition."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ATTENDANCE = "speaker_program_attended_12m"


def check_speaker_history(frame):
    """Count 0-1-0 only within an HCP and three consecutive calendar months.

    Run separately for each extract: joining train/scoring would add an
    unverified assumption of consistent feature construction across extracts.
    Source CSV line numbers assume a header and preserve the input row order.
    Counts refer to overlapping triples, not independent attendance events.
    """
    history = frame[["hcp_id", "period", ATTENDANCE]].copy()
    history["csv_line"] = range(2, len(history) + 2)
    history = history.sort_values(["hcp_id", "period"])
    history["month"] = pd.PeriodIndex(history["period"], freq="M").asi8
    groups = history.groupby("hcp_id", sort=False)
    for lag in (1, 2):
        for column in ("month", "period", "csv_line", ATTENDANCE):
            history[f"{column}_lag{lag}"] = groups[column].shift(lag)
    consecutive = (
        (history["month"] - history["month_lag1"] == 1)
        & (history["month_lag1"] - history["month_lag2"] == 1)
    )
    pattern = (
        consecutive
        & (history[f"{ATTENDANCE}_lag2"] == 0)
        & (history[f"{ATTENDANCE}_lag1"] == 1)
        & (history[ATTENDANCE] == 0)
    )
    columns = [
        "hcp_id", "period_lag2", "period_lag1", "period",
        f"{ATTENDANCE}_lag2", f"{ATTENDANCE}_lag1", ATTENDANCE,
        "csv_line_lag2", "csv_line_lag1", "csv_line",
    ]
    examples = history.loc[pattern, columns].reset_index(drop=True)
    summary = {
        "rows": len(history),
        "hcps": int(history["hcp_id"].nunique()),
        "complete_consecutive_triples": int(consecutive.sum()),
        "patterns_010": int(pattern.sum()),
        "affected_hcps": int(examples["hcp_id"].nunique()),
    }
    return summary, examples


def run(root=ROOT):
    root = Path(root)
    output = root / "reports" / "data_checks"
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "stage": "Attendance-definition check after the initial model evaluation",
        "definition": "Attendance in the previous 12 months, per supplied dictionary",
        "interpretation": (
            "Consecutive 0-1-0 histories conflict with the literal rolling-window "
            "definition absent corrections or changed semantics; not proof of leakage"
        ),
    }
    for dataset in ("train", "score"):
        frame = pd.read_csv(root / "data" / "raw" / f"hcp_engagement_{dataset}.csv")
        summary, examples = check_speaker_history(frame)
        examples.to_csv(output / f"speaker_010_{dataset}.csv", index=False)
        results[dataset] = summary
    (output / "feature_audit.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
