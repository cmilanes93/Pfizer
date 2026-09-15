"""Reproduce delivery diagnostics without fitting models or rewriting scores."""

from importlib.metadata import metadata
import io
import json
from pathlib import Path
import platform
import hashlib
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from packaging.specifiers import SpecifierSet
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/data_checks"
FLAGS = ["followup_call_logged_flag", "territory_target_refresh_flag"]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_table(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, lineterminator="\n")


def workbook_rows(workbook, sheet_path):
    """Read original dictionary cells directly; no workbook edits or formulas."""
    with ZipFile(workbook) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(element.itertext()) for element in
                      ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rows = []
        for row in ET.fromstring(archive.read(sheet_path)).findall(".//s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                value = cell.find("s:v", NS)
                if cell.get("t") == "inlineStr":
                    text = "".join(cell.find("s:is", NS).itertext())
                elif cell.get("t") == "s":
                    text = shared[int(value.text)]
                else:
                    text = value.text if value is not None else ""
                values[cell.get("r")] = text
            rows.append(values)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = ["data/raw/hcp_engagement_train.csv", "data/raw/hcp_engagement_score.csv",
                    "data/raw/submission_template.csv", "docs/source/Data_Dictionary.xlsx",
                    "outputs/submission_final.csv", "outputs/submission_final_metadata.json",
                    "requirements-lock.txt", "src/challenger.py"]
    before = {relative: sha(ROOT / relative) for relative in source_paths}
    train = pd.read_csv(ROOT / source_paths[0])
    score = pd.read_csv(ROOT / source_paths[1])
    reference = train.loc[train.period.between("2025-07", "2025-12")].copy()
    delivery_metadata = json.loads((ROOT / "outputs/submission_final_metadata.json").read_text())
    features = delivery_metadata["specification"]["features"]
    assert len(features) == 11 and len(set(features)) == 11
    assert reference.period.nunique() == score.period.nunique() == 6
    assert not reference[features].isna().any().any()
    assert not score[features].isna().any().any()
    assert not reference.duplicated(["hcp_id", "period"]).any()
    assert not score.duplicated(["hcp_id", "period"]).any()

    numeric, categorical, category_detail = [], [], []
    for feature in features:
        left, right = reference[feature], score[feature]
        if pd.api.types.is_numeric_dtype(left):
            pooled_sd = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
            mean_difference = float(right.mean() - left.mean())
            if pooled_sd == 0:
                smd = 0.0 if mean_difference == 0 else None
            else:
                smd = float(mean_difference / pooled_sd)
            numeric.append({"feature": feature, "reference_rows": len(left), "scoring_rows": len(right),
                            "reference_mean": left.mean(), "scoring_mean": right.mean(),
                            "mean_difference_scoring_minus_reference": mean_difference,
                            "reference_sd": left.std(ddof=1), "scoring_sd": right.std(ddof=1),
                            "standardized_mean_difference": smd,
                            "ks_statistic": float(ks_2samp(left, right, method="asymp").statistic),
                            "reference_missing": int(left.isna().sum()), "scoring_missing": int(right.isna().sum())})
        else:
            left_n, right_n = left.value_counts(), right.value_counts()
            categories = left_n.index.union(right_n.index)
            p = left_n.reindex(categories, fill_value=0) / len(left)
            q = right_n.reindex(categories, fill_value=0) / len(right)
            categorical.append({"feature": feature, "reference_rows": len(left), "scoring_rows": len(right),
                                "reference_categories": len(left_n), "scoring_categories": len(right_n),
                                "new_scoring_categories": len(right_n.index.difference(left_n.index)),
                                "total_variation_distance": float(.5 * (p - q).abs().sum()),
                                "max_absolute_category_share_difference": float((p - q).abs().max())})
            for category in categories:
                category_detail.append({"feature": feature, "category": category,
                                        "reference_count": int(left_n.get(category, 0)), "scoring_count": int(right_n.get(category, 0)),
                                        "reference_fraction": float(p[category]), "scoring_fraction": float(q[category]),
                                        "difference_scoring_minus_reference": float(q[category] - p[category])})
    write_table("retained_numeric_stability.csv", numeric)
    write_table("retained_categorical_stability.csv", categorical)
    write_table("retained_category_counts.csv", category_detail)

    flag_counts, flag_monthly, associations = [], [], []
    for field in FLAGS:
        for dataset, frame in [("all_training", train), ("recent_training", reference), ("scoring", score)]:
            for group_name, group in [("ALL", frame), *list(frame.groupby("period", sort=True))]:
                counts = {"missing": int(group[field].isna().sum()), "zero": int(group[field].eq(0).sum()),
                          "one": int(group[field].eq(1).sum())}
                assert sum(counts.values()) == len(group), "Unexpected operational-flag value"
                row = {"feature": field, "dataset": dataset, "period": group_name, "rows": len(group),
                       **counts, **{key + "_fraction": value / len(group) for key, value in counts.items()}}
                (flag_counts if group_name == "ALL" else flag_monthly).append(row)
            if dataset != "scoring":
                for value in [0, 1]:
                    selected = frame.loc[frame[field].eq(value)]
                    for target in ["responded_strict", "responded_loose"]:
                        associations.append({"feature": field, "dataset": dataset, "flag_value": value,
                                             "target": target, "rows": len(selected),
                                             "positive_labels": int(selected[target].sum()),
                                             "positive_label_fraction": float(selected[target].mean())})
    write_table("operational_flag_counts.csv", flag_counts)
    write_table("operational_flag_monthly.csv", flag_monthly)
    write_table("operational_flag_associations.csv", associations)

    joint_associations = []
    for dataset, frame in [("all_training", train), ("recent_training", reference)]:
        assert not frame[FLAGS].isna().any().any()
        for values, group in frame.groupby(FLAGS, sort=True):
            for target in ["responded_strict", "responded_loose"]:
                joint_associations.append({
                    "dataset": dataset, **dict(zip(FLAGS, map(int, values))),
                    "target": target, "rows": len(group),
                    "positive_labels": int(group[target].sum()),
                    "positive_label_fraction": float(group[target].mean()),
                })
    write_table("operational_flag_joint_associations.csv", joint_associations)

    dictionary_path = ROOT / "docs/source/Data_Dictionary.xlsx"
    dictionary = workbook_rows(dictionary_path, "xl/worksheets/sheet2.xml")
    descriptions = []
    for row in dictionary:
        for address, value in row.items():
            if address.startswith("A") and value in FLAGS:
                number = address[1:]
                descriptions.append({"feature": value, "sheet": "Data dictionary",
                                     "field_cell": address, "type_cell": "B" + number,
                                     "description_cell": "E" + number,
                                     "type_verbatim": row["B" + number],
                                     "description_verbatim": row["E" + number]})
    assert len(descriptions) == 2
    write_json(OUT / "operational_flag_definitions.json", {
        "source": "docs/source/Data_Dictionary.xlsx", "source_sha256": sha(dictionary_path),
        "definitions": descriptions,
        "interpretation": "Downstream/cycle-update timing is documented, but exact availability timestamps are not. Blank follow-up flags remain missing; they are not treated as zero. Association with supplied outcomes does not establish prediction-time availability or leakage by itself."})

    delivery_path = ROOT / "outputs/submission_final.csv"
    data = delivery_path.read_bytes()
    lf_bytes = data.replace(b"\r\n", b"\n")
    # This hypothetical line-ending conversion stays in memory; no CSV is changed.
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(data)), pd.read_csv(io.BytesIO(lf_bytes)), check_exact=True)
    line_endings = {"path": "outputs/submission_final.csv", "rows_including_header": len(data.splitlines()),
                    "crlf_count": data.count(b"\r\n"), "lone_lf_count": data.count(b"\n") - data.count(b"\r\n"),
                    "actual_sha256": hashlib.sha256(data).hexdigest(),
                    "hypothetical_lf_sha256": hashlib.sha256(lf_bytes).hexdigest(),
                    "lf_conversion_changes_bytes": lf_bytes != data, "lf_conversion_preserves_parsed_values_exactly": True,
                    "conversion_performed_on_delivery": False,
                    "portable_claim": "CSV row IDs and scores are portable across platforms. Regeneration uses pandas default line endings, so identical values can have different file hashes on different operating systems.",
                    "git_attributes": "*.csv -text preserves committed bytes; it does not make newly generated CSV line endings platform-independent.",
                    "comparison_basis": "Compare parsed row IDs and scores independently of line-ending serialization."}
    write_json(OUT / "csv_portability.json", line_endings)

    versions = []
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        package, pinned = line.split("==", 1)
        details = metadata(package)
        required = details.get("Requires-Python", "")
        assert details["Version"] == pinned
        spec = SpecifierSet(required)
        versions.append({"package": package, "pinned_version": pinned, "installed_version": details["Version"],
                         "requires_python": required, "allows_python_3_10": "3.10" in spec,
                         "allows_python_3_11": "3.11" in spec, "allows_python_3_12": "3.12" in spec})
    write_table("runtime_requirements.csv", versions)
    runtime = {"tested_python": platform.python_version(), "platform": platform.system(),
               "all_locked_packages_installed_at_pinned_versions": True,
               "all_locked_packages_allow_python_3_12": all(row["allows_python_3_12"] for row in versions),
               "packages_excluding_python_3_11": [row["package"] for row in versions if not row["allows_python_3_11"]],
               "runtime_requirement": "The pinned environment requires Python 3.12."}
    write_json(OUT / "runtime_compatibility.json", runtime)

    assert before == {relative: sha(ROOT / relative) for relative in source_paths}
    summary = {
        "scope": "Input-distribution stability, operational flags, runtime compatibility and CSV portability",
        "reference_window": "2025-07 to 2025-12", "scoring_window": "2026-01 to 2026-06",
        "reference_rows": len(reference), "reference_hcps": int(reference.hcp_id.nunique()),
        "scoring_rows": len(score), "scoring_hcps": int(score.hcp_id.nunique()),
        "retained_features": features, "numeric_features_checked": len(numeric), "categorical_features_checked": len(categorical),
        "maximum_absolute_standardized_mean_difference": max(abs(row["standardized_mean_difference"]) for row in numeric),
        "maximum_numeric_ks_statistic": max(row["ks_statistic"] for row in numeric),
        "maximum_categorical_total_variation": max(row["total_variation_distance"] for row in categorical),
        "no_unseen_retained_categories": all(row["new_scoring_categories"] == 0 for row in categorical),
        "retained_input_missing_cells": 0,
        "stability_claim": "The eleven retained inputs show small marginal distribution changes between these two extracts; their distributions are not identical.",
        "metric_definitions": {
            "standardized_mean_difference": "(Scoring mean - reference mean) / sqrt((reference sample variance + scoring sample variance)/2); signed and expressed in pooled standard deviations",
            "ks_statistic": "Largest absolute empirical-CDF difference for one numeric feature; descriptive D only, without an independence-based p-value",
            "total_variation": "Half the sum of absolute category-probability differences over the union of categories; ranges from 0 to 1",
        },
        "interpretation_limits": [
            "Rows are HCP-months and HCPs recur; these are descriptive row-weighted comparisons, not independent-sample significance tests.",
            "No pass/fail threshold or claim of equivalent distributions is imposed.",
            "Marginal similarity does not establish joint-distribution stability, stable individual HCP values, target calibration, data correctness or historical feature availability.",
            "Cohort composition differs between periods; small marginal differences do not resolve the target-definition or as-of risks.",
        ],
        "flag_counts_source": "reports/data_checks/operational_flag_counts.csv",
        "flag_definitions_source": "reports/data_checks/operational_flag_definitions.json",
        "joint_flag_associations_source": "reports/data_checks/operational_flag_joint_associations.csv",
        "runtime_source": "reports/data_checks/runtime_compatibility.json",
        "csv_portability_source": "reports/data_checks/csv_portability.json",
        "source_hashes_before_and_after_unchanged": before,
        "diagnostic_script_sha256": sha(Path(__file__)),
    }
    write_json(OUT / "delivery_checks.json", summary)
    print(json.dumps({key: summary[key] for key in ["reference_rows", "scoring_rows", "maximum_absolute_standardized_mean_difference",
                                                    "maximum_numeric_ks_statistic", "maximum_categorical_total_variation"]}, indent=2))
    print("Verified the delivery evidence; sources, recipes and score files are unchanged.")


if __name__ == "__main__":
    main()
