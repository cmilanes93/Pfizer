# Project conventions

Read the supplied brief and dictionary before changing the analysis.
The README is the entry point; `python -m src.solution` reproduces the solution.

## Data and implementation

Keep `data/raw/`, `docs/source/` and the input manifest unchanged.
Write analytical evidence under `reports/` and final predictions under `outputs/`.
Keep credentials, local environments and caches out of Git.

The final model specification, selection record and implementation fingerprints
are fixed. Changes to protected files require an explicit model-version change,
not an editorial update. Preserve the distinction between development selection
and subsequent exploratory comparisons.

Use the configured seed, temporal windows and monthly ranking metrics.
After implementation changes, run the relevant tests and compare reproduced
metrics and final scores with the supplied reference files.

## Documentation

Write professional English for a human reader. Explain the question, evidence,
decision and commercial consequence directly. Give each result a source.
Keep execution instructions and material limitations concise. Avoid duplicate
reports, editing instructions in deliverables and unsupported production claims.
