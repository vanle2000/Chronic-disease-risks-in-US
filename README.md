# Preventable Burden — U.S. Chronic Disease Intervention Prioritization

I built this to answer a resource-allocation question: **given a fixed prevention budget, which state, condition, and demographic segments should be funded first — and how many people would that reach?**

The output is a single number with units: `excess_cases`, the count of people who would not have a condition if their state performed at a rate another state has already achieved. Not an index. Not a score. A headcount you can put in a budget request and argue about.

**This repository is mid-rebuild.** The section below states exactly what runs today and what does not. I would rather you find this README conservative than find it overstated.

---

## Status

| Phase | Scope | State |
|---|---|---|
| **P0** | Remove placeholder files, archive coursework, honest README | **Done** |
| **P1** | Socrata + Census ACS ingestion, watermarks, Parquet landing | Not started |
| **P2** | dbt models, unit enforcement, tests, contracts, snapshot | Not started |
| **P3** | Disparity + ROI marts, Tableau Public dashboards, recommendations | Not started |
| **P4** | AI pipeline steps with validation gates, text-to-SQL eval harness | Not started |
| **P5** | Medicaid-expansion event study | Optional |

**What runs today:** the v1 Python pipeline (`src/data/preprocessing.py`, `src/modeling/train.py`) and five DuckDB SQL files under `sql/`. **I do not recommend using its outputs.** The audit below explains why, in specific terms, with line references. Correcting those defects is the entire purpose of P1–P3.

**Test suite:** 27 tests collect. 23 pass, 4 fail. All four failures share one root cause, documented in Defect 5.

---

## The question this answers

A public health agency has a prevention budget that cannot cover every condition in every state. Ranking by raw disease counts sends money to the largest states. Ranking by rate sends it to small states with volatile estimates. Neither answers the question a budget owner actually asks: *where does a dollar move the most people?*

Three deliverables:

1. **Where the burden is** — excess cases by state, condition, and demographic group.
2. **Who carries it** — disparity gaps, and critically whether each gap is widening or closing.
3. **Where money goes furthest** — expected cases averted per dollar, which reorders the list from (1).

Deliverable 3 is the point. The largest problem is usually not the best investment, and a ranking that shows the difference is a decision input rather than a report.

---

## The metric

```
excess_cases = (state_rate − benchmark_rate) ÷ 100 × adult_population
```

`benchmark_rate` is the national 25th percentile for that indicator, year, and demographic stratum. It is an **empirically achieved rate** — some state is already there — so the counterfactual is not hypothetical.

Everything the metric depends on is a stated, checkable choice:

| Component | Choice | Why it is defensible |
|---|---|---|
| Indicator panel | ~10 CDI questions, one `DataValueType` each | Prevents averaging percentages with per-100k rates |
| Grain | `state × indicator × year × stratum` | Enforced by a `unique` test, not by convention |
| Benchmark | National p25 | Achieved by real states; sensitivity to p10/median published |
| Denominator | Census ACS adult population | Converts a rate into people |

Contrast with what I built first:

```sql
-- v1, sql/04_mart_intervention_priority.sql:55
  0.40 * burden_score + 0.30 * mortality_score
+ 0.20 * reliability_score + 0.10 * COALESCE(yoy_pct_change, 0)
```

Nobody can demonstrate that 0.40 is wrong, which is precisely why nobody should believe it. A weighted sum of percentile ranks produces a number that survives every challenge because it answers to no unit. I replaced it rather than retuning it.

---

## Audit of v1: five defects I found in my own work

I wrote the code below. I am documenting it because the reasoning is the useful part, and because a reviewer will find these anyway — better that they find them already named and scheduled.

### Defect 1 — Averaging incompatible units
**Where:** `sql/01_create_analytics_views.sql`, every mart.

CDI stores `Crude Prevalence (%)`, `Age-adjusted Rate per 100,000`, `Number`, and `Per capita $` in one `DataValueAlt` column, distinguished only by `DataValueType`. Every view groups without filtering on it:

```sql
SELECT LocationAbbr, Topic, AVG(DataValueAlt) AS avg_indicator_value
FROM analytics.cdi_observations GROUP BY LocationAbbr, Topic;
```

This averages `12.4` (percent) with `245.7` (per 100,000). The result has no unit.

**Impact:** contaminates `mart_state_topic_summary`, `mart_topic_year_trends`, `mart_demographic_risk`, the `risk_level` bins, and every term of the priority score.
**Fix:** P2 pins one `DataValueType` per question and enforces it with a singular dbt test that fails the build.

### Defect 2 — A metric that measures the dataset, not the disease
**Where:** `sql/01_create_analytics_views.sql`, `mortality_source_rate`.

```sql
AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END)
```

`is_mortality` is derived from `DataSource` (`preprocessing.py:132`). So this is the share of *rows* drawn from a mortality-flagged source — a property of how CDC composed its surveillance catalogue, not of how many people died. A state reporting more mortality-sourced indicators scores higher.

**Impact:** 30% of the v1 priority score measures the dataset's own structure.
**Fix:** dropped. Mortality enters P2 as a separate indicator with its own denominator, or not at all.

### Defect 3 — A term pointing the wrong way
**Where:** `sql/04_mart_intervention_priority.sql:38`.

```sql
PERCENT_RANK() OVER (ORDER BY record_count) AS reliability_score
```

Well-resourced health departments report more indicators across more strata, accumulating more rows, and therefore rank as *higher priority for intervention funding*. The term inverts the goal it was meant to serve.

Two arithmetic problems ride along. The weights I documented (35/25/20/10/10, including a disparity term) never matched the weights I shipped (40/30/20/10, no disparity term). And raw `yoy_pct_change` is summed into a composite of `[0,1]` percentile ranks — a single 300% swing contributes `3.0` and dominates every other term.

**Fix:** the composite is deleted. Data sufficiency becomes a suppression rule (drop cells below a reporting threshold), not a scoring term — coverage decides whether a row is publishable, not how urgent it is.

### Defect 4 — The risk classifier can recover its own label
**Where:** `preprocessing.py:145` and `train.py:37`.

```python
df["risk_level"] = pd.cut(df["DataValueAlt"],
    bins=[-np.inf, 100_000, 250_000, 500_000, np.inf],
    labels=["Low", "Moderate", "High", "Very High"])
```

The thresholds are 100,000 / 250,000 / 500,000. A prevalence percentage (0–100) and a per-100,000 rate (typically under 2,000) **cannot reach the first threshold by construction**. Only `Number`-type rows — raw counts — can. So `risk_level` is not a measure of risk; it is a detector for "large raw count," which is why 99%+ of rows land in `Low`.

`RISK_FEATURES` then hands the classifier `DataValueType_enc` — the encoded unit type. The forest can reach the label almost deterministically through the same variable that produced it.

Two related notes: `value_normalized` (`preprocessing.py:143`) is an unmodified copy of `DataValueAlt` — the name asserts a transformation that does not happen, and it is fed to the mortality model as a feature. And `StratifiedKFold(shuffle=True)` (`train.py:105`, `train.py:161`) randomly splits a `state × year` panel, placing future observations in the training set for past ones.

**Impact:** the reported macro-F1 and feature importances describe the preprocessing code, not chronic disease. The class imbalance I corrected with `class_weight` is an artifact of the binning, not a property of the population.
**Fix:** the label is removed. P5 forecasts the rate series directly with time-ordered splits.

### Defect 5 — A production guard that makes the code untestable
**Where:** `preprocessing.py:87`.

```python
assert len(df) > 100, "Dataset unexpectedly small after cleaning"
```

A data-volume expectation is hardcoded as a bare `assert` inside a transformation. Two consequences: unit tests using realistic small fixtures (50 rows) raise instead of asserting behaviour — this is the single root cause of all 4 current test failures — and the check vanishes entirely under `python -O`, so it is absent exactly where it was meant to protect.

**Fix:** P2 moves volume and freshness expectations to `dbt test` and `dbt source freshness`, where they run against real data, report which rows failed, and cannot be optimised away.

---

## Architecture

```
CDC CDI (Socrata) ─┐
                   ├─→ raw/ Parquet ─→ dbt staging ─[dbt test]─→ fct_excess_burden ─┬─→ Tableau extracts
Census ACS ────────┘   (watermarked)   (one unit/question)          ↑               └─→ AI brief + triage
                                                                    │
                                                    population denominator
```

Two mechanisms carry the design. The **test gate** fails the build instead of publishing a broken mart. The **population join** is the single step converting a prevalence rate into a count of people; remove it and the headline metric reverts to something uninterpretable.

| Layer | Tool | What it provides |
|---|---|---|
| Ingest | Socrata API, incremental `$where` + watermark | Resumable extraction, replayable backfills |
| Land | Partitioned Parquet, `batch_id` / `ingested_at` / `row_hash` | Idempotent re-runs, audit trail |
| Transform | dbt Core on DuckDB | Layered models, tests, contracts, snapshots, exposures |
| Orchestrate | Makefile + GitHub Actions cron | Scheduled, logged, reproducible from a clean clone |
| Serve | Tableau Public via Google Sheets connector | A public dashboard that refreshes rather than going stale |
| Docs | `dbt docs` on GitHub Pages | Browsable column-level lineage |

**Trade-off I made deliberately:** no Airflow. Actions gives me cron and logs; the Makefile gives me replayability. What I give up is retry semantics, task-level observability, and a backfill UI. At production scale — multiple sources, SLAs with real consequences, on-call — I would move orchestration to Airflow and keep dbt exactly as it is. I chose the lighter tool because this workload is one weekly batch over a slowly-changing public dataset, and adding a scheduler I do not need would be complexity I would then have to justify.

---

## Repository layout

Current contents. Directories arrive when the phase that needs them does.

```
sql/                     v1 DuckDB views — superseded by P2, retained as the audit subject
src/data/preprocessing.py    v1 cleaning pipeline
src/modeling/train.py        v1 KMeans / LogReg / RandomForest baselines
src/eda.py                   exploratory plots
scripts/run_sql_pipeline.py  DuckDB runner
tests/                       27 tests (23 pass, 4 fail — Defect 5)
archive/                     original coursework notebook + deck, unmaintained
```

---

## Running it

```bash
pip install -r requirements.txt
```

Place the CDC extract at `data/raw/U.S._Chronic_Disease_Indicators__CDI_.csv`, then:

```bash
python src/data/preprocessing.py           # → data/processed/cdi_processed.parquet
python scripts/run_sql_pipeline.py         # builds the v1 DuckDB views
pytest tests/ -v
```

`pyarrow` and `duckdb` were missing from `requirements.txt`; a clean install could not execute the documented commands. Both are now pinned.

Automated ingestion (`make ingest`) arrives in P1. Until then the raw file is a manual download — stated plainly rather than implied to be automated.

---

## Engineering standards this project holds itself to

Landing in P2–P3, listed here because they are commitments, not aspirations:

- **Grain enforced by test.** `unique` on `(state, indicator_id, year, stratum)`. If the grain is not what I claim, the build fails.
- **Units enforced by test.** A singular test fails if any question resolves to more than one `DataValueType`. Defect 1 becomes impossible to reintroduce.
- **Contract on the published mart.** A column rename breaks the build rather than silently emptying a dashboard.
- **Freshness as an SLA.** `dbt source freshness` thresholds, wired to fail the weekly job.
- **Snapshots for real change.** CDC revises indicator definitions between releases; `snapshot` captures that as SCD2 because the data genuinely changes that way.
- **Runbooks.** Backfill and failure runbooks, written before they are needed.
- **Cost accounting.** Actions minutes, API spend per run, and storage, reported in the run summary.

---

## Where AI fits

The pipeline generates a weekly brief, triages anomalies, and drafts intervention memos. In every case the model **labels, explains, and writes prose — it never decides and never computes.**

| Step | What the model does | Gate |
|---|---|---|
| Weekly brief | Writes narrative from mart rows | Every numeral in the output must appear in the input frame, or the job fails |
| Anomaly triage | Classifies flagged points as artifact / definition change / real signal | Detection is deterministic statistics; the model only labels what was already found |
| Text-to-SQL | Generates queries against the marts | `EXPLAIN`-validated; execution accuracy on a committed 30-question gold set reported in CI |
| Intervention memo | Drafts recommendations | Every effectiveness claim must cite a row in the evidence seed; uncited claims are stripped |

Figures are templated from SQL. A model that writes sentences around verified numbers is useful; a model that produces the numbers is a liability. The gates are the engineering, not the prompt.

---

## Leadership Principles

Included because it was asked for. Each row points at something in this repository that can be checked, rather than restating the principle.

| Principle | Where it shows up here |
|---|---|
| **Customer Obsession** | The deliverable is a budget owner's question — "where does a dollar move the most people?" — not a model metric. `excess_cases` is denominated in people because that is the unit the decision is made in. |
| **Ownership** | Runbooks, freshness SLA, and cost accounting are in scope. I am specifying what happens when this breaks at 3am, not only what happens when it works. |
| **Invent and Simplify** | Five v1 marts and a four-term composite score collapse into one contracted fact table. I removed a scheduler rather than adding one. |
| **Are Right, A Lot** | I was not right the first time. Five defects, found and documented by me, with the reasoning that produced each error. |
| **Learn and Be Curious** | Callaway–Sant'Anna over two-way fixed effects for staggered adoption; wild cluster bootstrap for ~50 clusters. Both chosen because the naive method fails in this specific design. |
| **Insist on the Highest Standards** | Grain, units, and freshness are build-breaking tests rather than review conventions. The current 4 test failures are reported in this README instead of hidden. |
| **Think Big** | The architecture targets the general problem — benchmark-relative burden with a population denominator — not one dataset. Swapping CDI for another indicator source changes the staging layer only. |
| **Bias for Action** | P0 shipped in one sitting: 14 placeholder files removed, a broken test import fixed, two missing dependencies pinned. The rebuild does not wait on a complete plan. |
| **Frugality** | Entire stack is free: DuckDB, dbt Core, GitHub Actions, Tableau Public. Model generations are cached by input hash so an unchanged re-run costs nothing. |
| **Earn Trust** | The status table says what does not work. The audit names my own errors with line numbers. I would rather be checked than believed. |
| **Dive Deep** | Defect 4 required reading the bin thresholds against the unit distribution to see that a 100,000 cutoff is unreachable for a percentage — a fact invisible from the model metrics. |
| **Have Backbone; Disagree and Commit** | I argued against my own v1 metric and replaced it rather than retuning weights to make the ranking look reasonable. |
| **Deliver Results** | P0–P3 is a complete, defensible project on its own. P4–P5 are upside, sequenced so the deliverable is never hostage to the interesting part. |
| **Success and Scale Bring Broad Responsibility** | Disparity is a first-class output, not a filter. Suppression rules prevent publishing unstable estimates for small demographic cells, where a wrong number does the most harm. |

**Not evidenced here:** *Hire and Develop the Best* and *Strive to be Earth's Best Employer*. Both require a team, and this is solo work. Claiming them would undercut the rest of the table.

---

## Data sources

| Source | Use | Access |
|---|---|---|
| [CDC Chronic Disease Indicators](https://data.cdc.gov/) (`g4ie-h725`) | Disease indicators by state, year, stratum | Public, Socrata API |
| U.S. Census ACS | Adult population denominators | Public API |
| CDC Community Guide | Intervention effectiveness ratings | Curated seed, cited per row |

## Known limitations

- **Panel selection is a judgment call.** Choosing ~10 questions from CDI's hundreds shapes every result. Each inclusion carries a `rationale` column in the seed.
- **The p25 benchmark is a choice.** Sensitivity across p10 / p25 / median is published so the ranking's stability is visible.
- **Cost-per-case estimates vary widely in the literature.** Low/central/high bands carry through to the ROI output; no single point estimate is quoted.
- **ACS and CDI stratifications do not align perfectly.** Age bands and race/ethnicity categories differ. Where a stratum cannot be matched, the pipeline emits null rather than an approximation.
- **CDI is repeated cross-sections of state aggregates**, not an individual-level panel. Any causal work runs on ~50 clusters, where conventional clustered standard errors over-reject.

## License

Analysis code released under MIT. Source data is public domain (CDC, U.S. Census Bureau).
