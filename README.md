# Uncovering Patterns and Predicting Chronic Disease Risks in the U.S.

End-to-end analysis of the CDC's U.S. Chronic Disease Indicators (CDI) dataset (2001–2021), with risk-level classification, hyperparameter tuning, outlier detection, and public-health recommendations.

## Interactive Dashboard

Tableau Public dashboard: **<TABLEAU_PUBLIC_URL_TBD>** (replace this placeholder with your published URL — see [`tableau/README.md`](tableau/README.md) for the build guide).

![Dashboard A thumbnail](asset/tableau_thumbnail.png)

## 1. Problem Statement
I aim to use data-mining algorithms to tackle a challenge in U.S. public health through risk-level classification and predictive modeling of chronic-disease trends across U.S. states from 2001 to 2021. The dataset is the CDC's [U.S. Chronic Disease Indicators (CDI)](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi). The aim is to surface state-level patterns that can inform preventive measures and resource allocation.

Two modeling tasks are explored:
1. **Mortality classification** (binary): predict `DeadStatus` from disease topic + year + duration using Logistic Regression and Naive Bayes.
2. **Risk-level classification** (4-class: Low / Moderate / High / Very High): predict a binned `RiskLevel` derived from `DataValueAlt` using a Random Forest, with cross-validation and grid search.

## 2. Dataset

[U.S. Chronic Disease Indicators (CDI)](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi) — surveillance indicators developed by the CDC, the Council of State and Territorial Epidemiologists (CSTE), and the National Association of Chronic Disease Directors (NACDD).

- **~900K rows** across 20 years (2001–2021), all 50 states, plus DC and territories
- **22 disease topics**: Alcohol, Arthritis, Asthma, Cancer, Cardiovascular Disease, Chronic Kidney Disease, COPD, Diabetes, Disability, Immunization, Mental Health, Nutrition/Physical Activity/Weight, Older Adults, Oral Health, Overarching Conditions, Reproductive Health, Tobacco
- **15 source variables** plus engineered fields (`Latitude`, `Longitude`, `diseaseDuration`, `DeadStatus`, `RiskLevel`)

Key columns:

| Column | Description |
| --- | --- |
| `YearStart`, `YearEnd` | Reporting period for the indicator |
| `LocationAbbr` | State / territory abbreviation |
| `DataSource` | Origin of the measurement (survey, registry, NVSS, etc.) |
| `Topic`, `TopicID` | Disease topic group |
| `Question`, `QuestionID` | Specific indicator question |
| `DataValue`, `DataValueAlt` | Reported measurement (e.g., rate, count) |
| `DataValueType`, `DataValueUnit` | Units / type for the measurement |
| `StratificationCategory1`, `Stratification1` | Demographic stratification (sex, race, age group) |
| `GeoLocation` | `POINT (lon lat)` string, split into `Longitude` / `Latitude` |

## 3. How to Run

```bash
pip install -r requirements.txt
jupyter notebook CSE469FinalProject.ipynb
```

The notebook expects the raw CDI CSV at the path used in cell 4 (`U.S._Chronic_Disease_Indicators__CDI_.csv`). Download it from the [data.gov catalog page](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi) and place it next to the notebook (or update the path).

Reusable functions (data loading, lat/lon extraction, outlier detection, multi-class evaluation) live in `helpers.py` and are imported by the notebook.

### Reproducing the Tableau dashboard

The notebook's final "Tableau Exports" section writes four CSVs into `tableau/`. Once they exist:

1. Open Tableau Desktop Public Edition (Mac/Windows) or `public.tableau.com` Web Authoring (any OS).
2. Follow the step-by-step build guide in [`tableau/README.md`](tableau/README.md) — data-source connections, eight worksheets, two dashboards, and publishing instructions.
3. Save to Tableau Public, then paste the resulting URL into the **Interactive Dashboard** section at the top of this README.

## 4. Data Exploration

EDA covers data-type cleanup, missingness (rows missing `DataValueAlt` or `GeoLocation` are dropped, with the trade-off noted), demographic and topic distributions, and geographic trends. K-Means is used to identify the most-reported topic per state.

## 5. Data Quality & Validation

A dedicated section runs **before** any model training:

- **Audit:** null counts, duplicates, coverage (states/topics/years), and domain checks (no negative `DataValue`; percentages in [0, 100]).
- **Outliers:** per-topic IQR (k=3), per-topic z-score (|z|>3), and Isolation Forest (contamination 1%). Impossible values are dropped; flagged-but-plausible rows are retained and reported separately.
- **Hypothesis tests across 20 years:** Spearman trend on Cancer values, Kruskal-Wallis across topics, Mann-Whitney U comparing Cardiovascular Disease pre/post-2010, and chi-square for Topic x State independence.
- **Model-assumption checks:** for Logistic Regression — pairwise correlations, **VIF**, and events-per-variable; for Gaussian Naive Bayes — within-class feature correlations (independence) and Shapiro-Wilk normality on a 5K sample.

The reusable helpers (`detect_outliers_iqr`, `detect_outliers_zscore`, `detect_outliers_isolation_forest`, `compute_vif`, `evaluate_multiclass`) live in `helpers.py`.

## 6. Modeling

### 6.1 Mortality classifiers (binary)
Logistic Regression and Naive Bayes predict `DeadStatus` from 8 topic / time features. ROC curves compare both.

### 6.2 Risk-level classifier (multi-class Random Forest)
The original feature set included `DataValue` (the same measurement the target `RiskLevel` is binned from) and `DeadStatus` (derived from the data source), which caused **target leakage** and a misleading 99.99% accuracy. The corrected feature set excludes both. The model is now trained with:

- **Stratified 70/30 split** to preserve class proportions
- `class_weight="balanced"` to counter the Low-class majority
- **Majority-class baseline** reported for context
- **Macro PR-AUC and macro ROC-AUC** alongside accuracy (accuracy is unreliable under imbalance)
- **`GridSearchCV` with stratified 5-fold cross-validation** tuning `n_estimators`, `max_depth`, and `min_samples_split`
- **`cross_val_score`** sanity check on the tuned model

Class imbalance is acknowledged explicitly: the "Very High" class has only a handful of instances, so its recall is reported separately and not averaged into headline accuracy.

## 7. Conclusions & Business Recommendations

1. **Cardiovascular disease and cancer drive the majority of high-`DataValue` indicators** across the 20-year window; resource allocation should prioritize states whose cardiovascular indicator trends are rising year-over-year (visible in the geographic-trend plots).
2. **The Random Forest classifier predicts Low / Moderate / High risk levels reliably after leakage removal**, but **cannot be trusted for the "Very High" class** (only a handful of training examples). A state health agency using this model should treat "Very High" predictions as a *flag for human review*, not an automated alert.
3. **Outlier rows concentrate in mortality-source topics** (death-certificate-derived counts) — these are real, not data errors, and should be retained but reported separately when communicating average rates to non-technical stakeholders.
4. **Class imbalance is a structural property of public-health data** (most state-topic-year cells are "Low"); future iterations should re-bin `RiskLevel` using domain-defined thresholds (CDC cut-points) rather than uniform numeric bins, and consider SMOTE or focal loss for the rare classes.

## 8. Project Layout

```
CSE469FinalProject.ipynb   # main analysis notebook
helpers.py                 # reusable functions (data loading, outliers, evaluation)
requirements.txt           # pinned minimum versions
tableau/                   # Tableau Public dashboard extracts + build guide
presentation.pdf           # slide deck
asset/                     # supplementary assets (thumbnails, images)
```
