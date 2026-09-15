# Findings and investment decision

**Iterate with model complexity held fixed.** The data contain useful ranking
signal, but a three-field response table captures much of it. Fund agreement on
the outcome and information available at the decision date, then test the
simplest credible policy. Additional inputs must demonstrate value at the
actual contact capacity and cost before further model investment.

## Data and fitness for the decision

Training contains 48,585 HCP-months across 2,100 physicians in January
2024–December 2025. The scoring extract contains 12,723 rows in January–June
2026, including 2,318 rows from 400 physicians unseen in training. Strict
response occurs in 7.05% of training rows. Its supplied definition of a
substantial prescription increase makes it a provisional prioritisation target;
its numerical threshold and reference baseline remain unknown.

Initial observations affecting model scope were strong associations with
downstream operational flags, 87.77% missing follow-up in scoring, and changed
prescription-scale mappings. There are two distinct temporal changes: activity,
formulary coverage and response prevalence change in mid-2025; prescribing
scale changes at the 2025 training/2026 scoring boundary. Stable category
names or distributions do not establish that inputs were available at ranking
time. [Monthly data](../eda/tables/monthly.csv), [phases](../eda/tables/phases.csv)
and [scale comparisons](../eda/tables/numeric_shift.csv).

Visual summaries: [response rates over time](../eda/figures/target_rates.png),
[monthly activity and prescribing measures](../eda/figures/monthly_changes.png),
and [decile and operational-flag associations](../eda/figures/response_associations.png).

Further semantic checks identified attendance patterns inconsistent with a
literal trailing-twelve-month definition and a target-alignment concern. Among
2,887 checked strict-positive records, 1,504 show growth under an explicit
immediate-month NBRx comparison. Alternative baselines, source measures or
timing could explain this; the check does not prove the labels are wrong.
Reconciling the official outcome is the first investment condition.
[Target counts](../decision_audit/target_direction.csv),
[coverage](../decision_audit/target_coverage.csv) and
[feature decisions](../../docs/methodology/feature_decisions.md).

## What the alternatives achieve

The brief supplies neither an incumbent ranking nor contact capacity. Decile
is a proposed volume reference. The three-field table estimates response rates
for decile × formulary × specialty using fitting outcomes, with fitting
prevalence for unseen groups. It is a supervised estimator with fewer input
sources, not a rule independent of outcome data.

The table below reports mean monthly metrics on the supplied strict label.
Average precision summarises ranking quality across recall levels; the capacity
scenarios compare equal numbers selected each month.

| October–December 2025 | Precision at 5% | Precision at 10% | Precision at 20% | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Decile | 18.01% | 17.92% | 15.49% | 0.1432 |
| Three-field response table | 30.16% | 26.48% | 20.26% | 0.2343 |
| Four-field logistic | 31.16% | 27.30% | 20.10% | 0.2396 |
| Delivered eleven-field logistic | 33.76% | 26.98% | 19.85% | 0.2433 |
| Frozen HCP response history | 27.54% | 21.54% | 17.38% | 0.1840 |
| Eleven-field boosting | 33.12% | 26.15% | 20.18% | 0.2496 |

At 10%, delivered logistic selects 164 positive labels, the table 161, core 166
and decile 109, from 608 positions each. Its advantage over the table is 0.50
percentage points, with a conditional paired-HCP 95% interval of −2.96 to +2.30.
This establishes neither superiority nor equivalence. At 5%, delivered
logistic performs best among these alternatives; at 20%, the table is slightly
higher. The operating capacity can change the preference.

The table reaches 23.15% versus delivered logistic's 26.43% in development,
and 24.73% versus 26.45% in the six-month comparison. The latter overlaps
development. These trade-offs support testing simpler policies, without
claiming that they are interchangeable with the delivered model.
[Summary](summary.csv), [monthly counts](monthly.csv) and [intervals](checks.json).

The delivered eleven-field logistic recipe, C=0.1, follows the recorded
development selection. Four-field logistic is dominated on the development
criteria; the eleven-field logistic and boosting alternatives trade off
metrics, and the simplicity preference selects logistic. The subsequent
simple-policy comparisons inform investment rather than replace that choice.
The later 5% result is supporting evidence, not the original selection reason.
[Selection](../challenger/selection.json) and
[method](../../docs/analysis_guide.md#policies-and-model-choice).

Cutoff ties do not explain the entire observed 5% gap: even the table's most
favourable possible tie selections yield 31.47%, below delivered logistic's
33.76%. This bound is conditional on those scores and labels, not a guarantee
of future superiority. [Capacity curves](capacity_summary.csv) and
[tie audit](capacity_audit.json).

## What additional information earns

Prior response contains predictive information: strict response is 13.27%
after a positive two months earlier, versus 6.50% after a negative.
The more relevant test adds an HCP's past response rate to four-field logistic
on matching fitting populations. It yields 28.46% later precision at 10%,
versus 27.47% for its core reference: six more positive selections, or 0.99
points, with a conditional 95% interval of −0.82 to +2.30. The reference differs
from 27.30% above because both sensitivity models exclude the first two fitting
months without prior history. History merits fresh comparison; it is not a
demonstrated replacement. [History comparison](history_increment_summary.csv)
and [interval](history_increment.json).

Excluding visits and samples does forgo possible signal. Adding calls gives
28.12% later precision at 10% but reduces precision at 5% from 33.76% to 31.79%;
adding samples gives 27.96% at 10%. These results do not resolve late-entry
risks. Removing the three email/digital fields gives 25.99% at 10%, versus
26.98% delivered; their availability also remains provisional. The
[feature matrix](../../docs/methodology/feature_decisions.md#activity-evidence)
sets out the common timing concern and the measured sensitivities.

## Expected performance and uncertainty

The working expectation is **about 27% precision at illustrative 10% capacity
on the supplied strict label**, conditional on comparable definitions,
population, prevalence and decision-time inputs. It is a reference to test on
a new reserved period, not validated 2026 performance. Reconciliation of the
outcome or inputs may change both the task and its apparent signal.

The delivered model fits through November 2025 and remains fixed across
January–June 2026, using each month's inputs when that month is ranked. These
are recurring priorities, not January-only forecasts. Scores are uncalibrated
ranking values; the scoring file supplies no outcomes for validation.

All historical comparisons are exploratory. Bootstrap intervals condition on
fitted scores and observed HCP histories; they omit fitting, selection and
future calendar uncertainty. Neither predictive concentration nor these
intervals identify prescriptions caused by engagement or financial return.

## Two weeks and one additional engineer

| Period | Data scientist | Engineer |
| --- | --- | --- |
| Week 1 | Reconcile the target; agree capacity, recontact rules, costs and minimum worthwhile gain with Commercial | Reconstruct inputs available at each decision date, starting with simple policies |
| Week 2 | Freeze the comparison on validated inputs and design a prospective policy test | Version extracts and scoring with availability checks |

Before expanding model complexity, require a positive paired lower 95% bound
against a prespecified simple reference in a new matured period, together with
the business minimum agreed before outcomes are opened. For scaling a contact
policy, the proposed gate is a lower 95% bound above zero for incremental net
value, with an estimated gain meeting the prespecified commercial minimum at
actual capacity and cost. These are proposed decision criteria, not supplied
business thresholds.

Simplify when extra inputs do not justify their cost. Stop or redesign if the
target cannot be reconciled. Sample size depends on effect size, variance,
outcome timing and randomisation unit, including territory dependence. Two
weeks can establish the data contract and prepare the test; future outcomes
still need time to mature.
