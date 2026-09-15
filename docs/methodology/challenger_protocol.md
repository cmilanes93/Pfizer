# Capacity and algorithm comparison protocol

Written on 13 September 2026 before fitting the new boosting candidate.
This is an exploratory continuation after previous historical results were
examined. The prior protocols, selections, predictions and reports are preserved.

## Purpose and decision

The brief asks whether further investment in next-cycle HCP prioritisation is
warranted. Capacity, costs and response value are not supplied. Return scores
for every scoring row; do not impose a 10% contact policy.

The new question is whether a limited nonlinear competitor adds robust ranking
value, and whether the choice depends on the operating capacity or a practical
one-percentage-point tolerance. Retain strict response as a provisional label
proxy, subject to the separate temporal-definition audit. Never replace the
labels with a guessed reconstruction. A predictive comparison does not identify
the effect caused by engagement.

## Fixed candidates and implementation choices

Compare exactly these candidates and the decile reference:

- core_logistic: the existing four-field core, logistic C=0.1.
- extended_logistic: the existing eleven fields without speaker, logistic C=0.1.
- extended_boosting: the same eleven fields, HistGradientBoostingClassifier.

The boosting configuration is learning_rate=0.05, max_iter=100,
max_leaf_nodes=7, max_depth=3, min_samples_leaf=50, l2_regularization=1.0,
early_stopping=False, random_state=42. Numeric fields pass through and categories
use a fitting-only one-hot encoder with unknown categories ignored. Trees do not
need numerical standardisation. Do not use automatic categorical inference that
could turn integer decile into a nominal feature.

These are conservative engineering choices, not tuned or business-derived
constants: bounded depth/leaves limit interactions and model size; leaf support
and L2 discourage fitting very small pockets; shrinkage with a fixed number of
iterations limits the search; disabling automatic early stopping avoids a random
internal validation split. Seed 42 provides reproduction, not scientific evidence.
Do not try a second boosting configuration based on its measured result.

Reuse the old logistic preprocessing. No flags, identifiers, attendance or
previously excluded measurements enter the new candidate.

## Metrics and a rule without an unknown business utility

Use mean monthly average precision (AP) and mean monthly precision at 5%, 10%
and 20% as a vector of four development metrics. AP weights precision at recall
increments and summarises ranking quality without selecting one list size.
It is not a financial objective. The three capacities are existing sparse,
central and broader illustrations, retained to avoid choosing favourable
scenarios after the new results; none is claimed to be the true capacity.

Before evaluation, select using this fixed conservative multi-criteria rule:

1. A learned candidate is admissible only if it is no worse than decile on
   all four metrics and strictly better on at least one.
2. Within admissible candidates, remove a candidate if another is no worse on
   all four metrics and strictly better on at least one.
3. From the remaining candidates choose fewer input fields, then logistic before
   boosting, then higher mean monthly AP, then model name for a deterministic tie.
4. If no learned candidate is admissible, select decile.

A tolerance of 1e-12 is only floating-point comparison tolerance. There is no
one-percentage-point practical margin in this rule. Simplicity resolves measured
trade-offs because real contact utility is unknown and the brief favours
defensible simple solutions. This is a provisional delivery rule; it does not
prove the chosen model is best for every capacity or commercial cost.

Show the entire capacity curve at one-percentage-point display increments
(1% through 100%), plus exact 5%, 10%, 20% counts and month-specific results.
This grid is plot resolution, not operational policy. Identify curve crossings,
and do not claim full-curve dominance from dominance on the four selection metrics.

As a sensitivity only, report what the old P@10% rule would choose using
simplicity tolerances 0, 0.5, 1 and 2 percentage points. These are scenarios around
the old rule, not validated economic thresholds; do not use them to change the
new selection. Include capacity-specific winners and AP comparisons so the cost
of a different preference is visible.

## Timing and provenance

Development fits through May 2025 and ranks July–September 2025.
Freeze candidate configuration and implementation hashes before running it.
Save selection, the full metric vector and source hashes before new later metrics.

Then refit the fixed candidate specifications through August and evaluate
October–December. Do not reselect using this period. Refit through May and rank
July–December with one frozen model as a duration sensitivity. Development months
overlap this six-month check, so it is not independent validation.

Monthly ranking, next-calendar-month labels and availability at that month's end
remain documented assumptions. An additional-month label-delay check uses the
selected model. If selection changes, also regenerate loose-label and
withheld-HCP sensitivities and ranking-score calibration diagnostics so old
model evidence is not attributed to the new one.

For uncertainty, use the existing 2,000 paired whole-HCP bootstrap procedure at
the 5%, 10% and 20% scenarios. Keep fitted models fixed and label the intervals
conditional, not simultaneous across capacities or a forecast for 2026.

## Outputs and final interpretation

All new evidence is under reports/challenger/. Final predictions are
outputs/submission_final.csv and corresponding metadata. Fit the selected
recipe through November 2025 and freeze it across January–June 2026; each month's
supplied features are assumed available at that month's ranking.

The investment recommendation uses magnitude, monthly stability, capacity
sensitivity and data-definition risks. A favourable ranking comparison supports
a bounded data and prospective-validation phase. Costs, target construction,
historical feature vintages and incremental engagement value remain unresolved.
If the new audit changes the interpretation of the target, report that limitation
as central rather than silently substituting a different outcome.
