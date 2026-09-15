# Feature eligibility and evidence

The matrix covers all 28 supplied fields. The delivered logistic model uses
four core inputs and seven additional inputs. A monthly record date does not
certify when its values became available: CRM activity can be entered late or
revised, and prescribing data have an unspecified reporting lag. **All retained
inputs remain conditional on decision-time availability checks.**

Definitions come from the [brief](../source/Candidate_Brief.docx) and
[dictionary](../source/Data_Dictionary.xlsx). In these sources a representative
“call” means a visit; NBRx means new prescriptions, while TRx includes refills.
Decimal prescribing and patient-volume values are permitted.

## Field matrix

| Field | Role in the analysis | Reason and timing condition |
| --- | --- | --- |
| `row_id` | Submission key; excluded from predictors | Preserve all 12,723 template IDs and their order. No business meaning is supplied. |
| `hcp_id` | Group histories, uncertainty and cohorts; excluded as a predictor | Repeated monthly records are not independent people. Scoring includes 400 new HCPs and 2,318 associated rows. |
| `period` | Chronological splits and monthly ranking; excluded as a predictor | The calendar month does not identify event, entry or ingestion time. |
| `specialty` | Core input | Clinical context; five categories with no unseen scoring categories. Historical master-data version is unverified. |
| `region` | Additional retained input | Broad geographic context. Five consistent categories; no recoding is needed. The two source systems' vintages are unspecified. |
| `territory_id` | Excluded | Detailed assignment may be revised. There are 347 training levels and 24 scoring rows with an unseen territory. |
| `institution_type` | Additional retained input | Practice setting; four categories with no unseen scoring values. Update timing is unspecified. |
| `decile` | Core input and volume-policy reference | Rank 10 is highest. Annual refresh requires the correct historical vintage; its order avoids a guessed conversion of prescribing scales. |
| `years_in_practice` | Additional retained input | Experience is constant within HCP in training, which may reflect an extract-time snapshot rather than historical values. |
| `calls_last_90d` | Excluded | Visit counts are exposed to late/batched entry and unresolved count/recency definitions. See activity evidence below. |
| `days_since_last_call` | Excluded | Last logged visit depends on entry timing. Count/recency conflicts and the scoring-boundary shift remain unresolved. |
| `email_sent_last_90d` | Additional retained input | Prior contact volume; window endpoint and log availability are unspecified. Similar distributions do not certify availability. |
| `emails_opened_last_90d` | Additional retained input | Prior digital interaction. Unique emails versus repeated opening events is undefined, so no open rate is derived. |
| `samples_dropped_last_90d` | Excluded | Samples are left during visits; historical counts may contain late entries. The visit inconsistency does not prove sample leakage. |
| `speaker_program_attended_12m` | Excluded | Consecutive 0→1→0 histories conflict with a literal trailing-twelve-month indicator unless corrections or another definition apply. |
| `digital_engagement_score` | Additional retained input | Vendor composite in [0,100]; formula, lookback and refresh dates are unknown. |
| `campaign_wave_id` | Excluded | Assignment timing, wave meaning and reuse are unclear. Matching 219 category labels across extracts do not establish comparable campaigns. |
| `patient_volume_est` | Excluded | One exact value per decile within each extract; the mapping changes in scoring. No conversion factor is inferred. |
| `nbrx_last_month` | Excluded | Prior-month prescriptions may arrive after ranking time; mean falls from 9.03 in late 2025 to 3.76 in scoring. |
| `nbrx_3m_avg` | Excluded | Last included month is unspecified and values map exactly to decile. Decile 6 maps to 9.62 in training and 4.02 in scoring. |
| `trx_3m_avg` | Excluded | Window endpoint and publication lag are unknown; mean falls from 30.67 in late 2025 to 12.78 in scoring. |
| `competitor_share_pct` | Core input | Relative prescribing context avoids reliance on absolute scale, but lookback, publication lag and vendor comparability remain unverified. |
| `payer_mix_commercial_pct` | Additional retained input | Coverage composition; refresh cadence and historical vintage are unspecified. |
| `formulary_status` | Core input | Access under the main payer can change. Preferred coverage rises from 22.33% before July 2025 to 40.79% afterward. |
| `followup_call_logged_flag` | Diagnostic only; excluded | Downstream CRM update, complete in training but 87.77% missing in scoring. Blank is not equivalent to zero. |
| `territory_target_refresh_flag` | Diagnostic only; excluded | Cycle targeting update carries the same downstream timing warning. Strong association does not establish advance availability. |
| `responded_strict` | Primary outcome; never a predictor | Supplied substantial-response proxy: 3,424 positives (7.05%). Threshold, baseline and exact timing are unconfirmed. |
| `responded_loose` | Outcome sensitivity; never a predictor | Supplied any-increase label: 13,186 positives (27.14%); all strict positives are nested. Higher prevalence does not establish greater business relevance. |

Sources: [structure](../../reports/eda/tables/structure.csv),
[categories](../../reports/eda/tables/category_shift.csv),
[HCP stability](../../reports/eda/tables/hcp_stability.csv),
[decile mappings](../../reports/eda/tables/decile_mapping.csv),
[numeric shifts](../../reports/eda/tables/numeric_shift.csv) and
[period summaries](../../reports/eda/tables/phases.csv).

## Activity evidence

The brief warns that representatives may enter activity late or in batches at
cycle close. Event time alone therefore cannot establish when a count was
available. Supporting checks find 3,194 training rows and 1,298 scoring rows
with zero visits in 90 days but a last visit within 90 days. Mean visit recency
changes from 4.62 in July–December 2025 to 16.56 in scoring. Different inclusion
rules or logging conventions could explain these patterns; they require
reconciliation rather than an assumed correction.
[Semantic checks](../../reports/eda/tables/semantic_checks.csv).

This post-selection sensitivity holds the logistic settings and historical
evaluation windows fixed. It changes no delivered score and creates no new holdout:

| Variant | Later precision at 5% | Later precision at 10% | 95% interval for the 10% difference versus delivered |
| --- | ---: | ---: | --- |
| Delivered eleven fields | 33.76% | 26.98% | Reference |
| Add calls | 31.79% | 28.12% | −0.33 to +3.62 percentage points |
| Add samples | 33.76% | 27.96% | −0.50 to +2.64 percentage points |
| Add calls, samples and recency | 31.79% | 28.12% | Not estimated |

The joint variant does not isolate recency's contribution. Removing emails
sent, emails opened and digital score gives 25.99% later precision at 10%.
Visits and samples may add predictive information, and an interval including
zero is not evidence of no contribution. These checks do not certify availability
for either the excluded or retained activity fields. The delivered exclusions
remain; retained email/digital fields are provisional pending the same availability
standard. [Results](../../reports/data_checks/activity_sensitivity.csv) and
[conditional intervals](../../reports/data_checks/activity_sensitivity.json).

## Attendance and outcome definitions

The attendance check, completed after the initial model evaluation, finds
0→1→0 patterns in 4,598 of 41,395 consecutive training triples (1,887 HCPs) and
856 of 7,898 scoring triples (783 HCPs). These are overlapping triples, not
independent events. The pattern supports exclusion pending clarification,
without identifying the cause or proving leakage.
[Attendance audit](../../reports/data_checks/feature_audit.json).

For strict response, only 1,504 of 2,887 checked positives show growth under the
explicit immediate-month comparison. The official baseline, source measure or
alignment may differ, so this does not establish erroneous labels. Neither
outcome is yet a validated operational investment measure.
[Target-direction checks](../../reports/decision_audit/target_direction.csv) and
[coverage](../../reports/decision_audit/target_coverage.csv).

These decisions concern eligibility under stated assumptions. They do not prove
that every retained input adds value or that contact causes the observed response.
The [analysis guide](../analysis_guide.md) defines the comparison and uncertainty.
