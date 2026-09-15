# Analysis guide

The [findings](../reports/comparison/findings.md) answer the investment question.
The [feature matrix](methodology/feature_decisions.md) explains which information
enters the delivered model and which definitions remain unresolved.
The [executed exploration notebook](../notebooks/01_exploration.ipynb) documents
the initial data review with saved outputs.

## Decision and deliverables

The [brief](source/Candidate_Brief.docx) asks whether the supplied data justify
further investment in next-cycle HCP prioritisation. Strict response is a
provisional proxy for a substantial prescription increase: it is a more relevant
starting point for limited contacts than any increase, but its threshold,
reference baseline and commercial value require confirmation. Loose response
is a sensitivity; its higher prevalence does not establish a better objective.

The required delivery is Python code, a `row_id,score` CSV for all 12,723 scoring
rows, and a PDF of at most six slides. Scores may be ranking values in [0,1].
The slides cover data observations, solution and exclusions, validation and
expected performance, investment recommendation, uncertainties, and two weeks
of further work with another engineer. The brief specifies four hours and
Python 3.10 or later; this dependency set uses 3.12. Run instructions, pinned
dependencies, a fixed seed and meaningful tests support the reproducibility bonus.

## Timing and evaluation

Monthly ranking is an interpretation of the HCP-month schema. The brief does
not supply an operating capacity, incumbent ranking or recontact policy.

| Stage | Fit through | Evaluate or score |
| --- | --- | --- |
| Development | May 2025 | July–September 2025 |
| Later comparison | August 2025 | October–December 2025 |
| Six-month comparison | May 2025 | July–December 2025 |
| Final scoring | November 2025 | January–June 2026 |

The skipped row month assumes next-calendar-month outcomes available at that
month's end. This is an analytical assumption, not a verified reporting SLA.
Models remain fixed within each horizon; each month's inputs are assumed
available when that month is ranked. The six-month comparison overlaps
development. The 2026 extract supplies no outcomes.

This is a retrospective analysis. The delivered recipe was selected using
development metrics. The broader simple-policy and history comparisons were
added after selection; they inform investment rather than replace that choice.
All historical periods have been examined, so none is an untouched holdout.

Before evaluation or scoring, the code validates the file fingerprints,
[implementation freeze](../reports/challenger/implementation_freeze.json) and
development metrics, then checks that the saved model follows the selection
rule. The main pipeline also refits the development candidates and reproduces
their metrics and recorded choice. These checks establish reproducibility and
internal consistency; they do not independently establish when the specification
was fixed or prove that the later period was unseen.

## Policies and model choice

| Policy | Construction |
| --- | --- |
| Decile | Descending prescribing-volume rank; a proposed reference, not an established commercial incumbent |
| Three-field response table | Unsmoothed fitting response rate for decile × formulary × specialty; unseen groups use fitting prevalence |
| Four-field logistic | Decile, specialty, formulary and competitor share; C=0.1 |
| Delivered logistic | Four core fields plus seven profile, market and activity fields; C=0.1 |
| Frozen HCP response history | Prior strict-response rate through the fitting cutoff; unseen HCPs use fitting prevalence |
| Eleven-field boosting | One fixed HistGradientBoostingClassifier configuration using the same eleven inputs as delivered logistic |

Logistic uses training-only numeric standardisation and one-hot encoding.
The boosting comparison tests a bounded nonlinear alternative. C=0.1 and the
boosting settings are fixed engineering choices, not claims of optimal tuning.
The [configuration](../config/challenger.json) specifies the parameters.
Candidate feature sets are defined in [config/revision.json](../config/revision.json),
which is included in the recorded fingerprints.

The recorded selection compares mean monthly average precision and precision
at 5%, 10% and 20% capacity. A learned model must be no worse than decile on
every metric and better on at least one. Among admissible models, remove those
dominated on these metrics, then prefer fewer inputs, logistic over boosting,
higher average precision and finally model name. If none is admissible, use
decile. Four-field logistic is dominated in development; the eleven-field
models trade off metrics, so this rule selects logistic. This is a provisional
choice under unknown utility, not a commercial optimum. See the
[selection](../reports/challenger/selection.json) and
[protocol](methodology/challenger_protocol.md).

The history-increment sensitivity adds an HCP's prior-response rate to the
four-field model. Fitting row t uses outcomes through t−2. Both alternatives
exclude January–February 2024 rows without preceding history; evaluation
histories freeze at the fitting cutoff. Their core reference therefore differs
from the primary comparison. A separate rolling-history diagnostic allows
newly matured outcomes through t−2; it cannot update the unlabelled 2026 batch.

## Metrics and limits

Each month selects `ceil(capacity × eligible rows)` using the same seed-42 tie
rule. Precision is the fraction of selected positions with the supplied label;
average precision summarises ordering across recall levels. Metrics give each
month equal weight; counts sum positions and positives. The 5%, 10% and 20%
capacities are scenarios, not supplied limits or utility-optimised thresholds.
At 10% in the later window, 608 positions correspond to 401 distinct HCPs for
the delivered model. Recontact constraints would define a different policy.

Paired bootstrap comparisons resample whole HCP histories 2,000 times while
keeping fitted scores fixed. Their 95% intervals condition on these models and
months, excluding fitting, selection and future calendar uncertainty. They do
not model dependence between different HCPs in the same territory, and are not
simultaneous across capacities. An interval crossing zero establishes neither
superiority nor equivalence.

Capacity curves retain the 1–100% grid. The cutoff-tie audit holds scores,
labels and positions above the cutoff fixed and varies selections within the
tied group. Its attainable bounds and random-tie ranges describe tie handling,
not confidence intervals for future model performance.

The target audit compares NBRx in t+1 with t using `nbrx_last_month` in rows t+2
and t+1, within consecutive training records. This tests one explicit alignment;
it neither reconstructs the official label nor establishes its formula.
Activity sensitivities measure predictive information without certifying when
that information was available. These data do not identify engagement's causal
effect or return on investment.

## Execution and evidence

Use the environment and commands in the [README](../README.md). The workflow
entry point is `python -m src.solution`.

It checks the recorded development selection in memory, then regenerates the
data audits, model evaluation, policy comparisons and scores. The selection
and implementation-freeze records are not overwritten. The
[reproduction check](../reports/verification.json) records the tested environment
and score equivalence; the PDF and PPTX are supplied presentation artifacts.

| Module | Responsibility |
| --- | --- |
| `src.solution` | Coordinate input checks, audits, fixed-model evaluation and supporting comparisons |
| `src.challenger` | Recorded selection rule, logistic/boosting evaluation and final scoring |
| `src.baselines`, `src.capacity_audit` | Simple policies, incremental history, capacity curves and cutoff ties |
| `src.eda`, `src.feature_audit`, `src.decision_audit` | Exploration, attendance semantics, target alignment, repeated selections and profile support |
| `scripts/activity_sensitivity.py`, `scripts/verify_delivery_data.py` | Activity variants, operational flags, retained-input distributions and delivery checks |
| `src.modeling`, `src.metrics` | Fitting, timing guards, aggregation and paired HCP bootstrap |

| Question | Evidence |
| --- | --- |
| How do policies compare? | [Summary](../reports/comparison/summary.csv), [monthly counts](../reports/comparison/monthly.csv), [conditional intervals](../reports/comparison/checks.json) |
| What does prior response add? | [Matched-population comparison](../reports/comparison/history_increment_summary.csv), [intervals](../reports/comparison/history_increment.json) |
| How much do capacity and ties matter? | [Capacity curves](../reports/comparison/capacity_summary.csv), [tie audit](../reports/comparison/capacity_audit.json) |
| What supports the target concern? | [Direction checks](../reports/decision_audit/target_direction.csv), [coverage](../reports/decision_audit/target_coverage.csv), [source-row examples](../reports/decision_audit/target_examples.csv) |
| How often are people selected again? | [Selection counts](../reports/decision_audit/selection_summary.csv), [turnover](../reports/decision_audit/selection_turnover.csv) |
| Are scoring profiles represented? | [Joint-profile support](../reports/decision_audit/profile_support.csv), [cohort summaries](../reports/decision_audit/profile_summary.csv) |
| What supports activity exclusions and slide 2? | [Activity sensitivity](../reports/data_checks/activity_sensitivity.csv), [intervals](../reports/data_checks/activity_sensitivity.json), [joint flag counts](../reports/data_checks/operational_flag_joint_associations.csv) |
| What was scored and checked? | [Submission metadata](../outputs/submission_final_metadata.json), [delivery checks](../reports/data_checks/delivery_checks.json) |

For slide 2, use `all_training` and `responded_strict` in the joint flag file.
In follow-up/refresh order, groups (0,0), (0,1), (1,0), (1,1) give 0.8%, 3.2%,
25.3% and 60.9% after rounding. These are joint groups, not marginal flag rates.

The final file contains one uncalibrated ranking score per template row. Portable
reproduction compares IDs and numeric scores with an explicit tolerance; CSV
bytes can differ through serialization. Tests check implementation properties,
not the official target formula, historical input availability or commercial impact.

AI tooling assisted with implementation and document drafting. The analytical decisions, the evidence linked here and the conclusions are mine, and every result is reproducible from the code in this repository.
