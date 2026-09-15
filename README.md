# HCP engagement prioritisation

Requires Python 3.12 for the pinned environment; tested version: 3.12.14.

Commercial needs to decide which healthcare professionals to prioritise next
cycle and whether further investment is justified.

**Recommendation: iterate on the outcome and decision-time data before adding
model complexity.** At illustrative 10% monthly capacity in October–December
2025, the delivered eleven-field logistic model reaches 26.98% precision,
against 26.48% for a three-field response table and 27.30% for four-field
logistic. The eleven-field model performs better at 5% capacity. These
exploratory results support testing the simplest credible policy; they do not
establish the commercial effect of contacting selected HCPs.

- [Initial data review](notebooks/01_exploration.ipynb): executed notebook with saved outputs.
- [Final predictions](outputs/submission_final.csv): all 12,723 template rows.
- [Six-slide presentation](reports/presentation/Pfizer_challenge_final.pdf) and [editable PPTX](reports/presentation/Pfizer_challenge_final.pptx).
- [Findings and investment decision](reports/comparison/findings.md).
- [Analysis guide](docs/analysis_guide.md): methods, assumptions and evidence.

## Run

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. On Windows, the Python
launcher can create the environment with `py -3.12 -m venv .venv`.

```bash
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python -m src.solution
```

The entry point runs the analytical workflow. The PDF and PPTX are supplied
presentation artifacts. See the [guide](docs/analysis_guide.md#execution-and-evidence)
for module responsibilities and the evidence behind each result.

## Files

| Location | Contents |
| --- | --- |
| `data/raw/`, `docs/source/` | Supplied data, brief and dictionary |
| `config/`, `src/`, `scripts/`, `tests/` | Specifications, analysis and checks |
| `reports/comparison/`, `reports/data_checks/` | Main comparisons and supporting diagnostics |
| `outputs/` | Predictions and score metadata |
| `reports/presentation/` | Final PDF and editable presentation |
