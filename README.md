# Chronic Disease Risk Intelligence: Predicting Outbreak Patterns Across the U.S.

---

## Case Study

### Introduction
Chronic diseases are the leading cause of death and disability in the United States, accounting for 7 of the top 10 causes of mortality and consuming 90% of the nation's $4.1 trillion annual healthcare spend. Public health agencies operate under severe resource constraints — they cannot intervene everywhere. The critical question is: **where do they intervene first, and for which diseases?**

This project applies data mining and machine learning to 20 years of CDC surveillance data to answer that question — building a system that identifies geographic risk clusters, predicts mortality-linked outcomes, and classifies disease severity across all 50 states.

### Problem
The U.S. Chronic Disease Indicators dataset is dense, multi-source, and high-cardinality. Raw records mix mortality statistics with survey percentages and clinical measurements in the same `DataValue` column. Geographic coverage is uneven — some states have robust surveillance infrastructure while others have systematic data gaps. Most critically, **the highest-risk populations are the least represented in the data**, making naive accuracy metrics dangerously misleading.

The challenge was to build models that are honest about what they know and don't know, and that surface actionable signals rather than inflated performance numbers.

### Solution
A three-stage pipeline: clean → explore → model.

**Stage 1 — Data Cleaning & Feature Engineering (794K records)**
- Parsed WKT `GeoLocation` strings into `Latitude` / `Longitude` for geospatial analysis
- Imputed missing `DataValueUnit` using within-group mode by `Topic × Question` — preserving domain context rather than global imputation
- Engineered `DeadStatus` (binary) from `DataSource` field: mortality-linked sources (NVSS, Death Certificates) = 1
- Engineered `diseaseDuration` from `YearEnd − YearStart` to capture multi-year disease burden
- One-hot encoded all 17 disease topics as indicator features
- Dropped low-signal columns: footnote fields, redundant confidence bounds

**Stage 2 — Exploratory Analysis**
- Geographic heatmap (state × year): data density spikes post-2015 reflect expanded CDC programs, not actual disease increases — a critical distinction that prevents modeling artifacts
- Demographic breakdown by gender and race: persistent disparities in Diabetes and Cardiovascular Disease for Black, non-Hispanic populations
- Topic trend analysis (2010–2021): Mental Health shows the steepest upward trajectory post-2016

**Stage 3 — Modeling**
- **K-Means clustering (k=3):** Segmented states into three surveillance intensity tiers (Silhouette Score: 0.635)
- **Logistic Regression:** Predicted `DeadStatus` (mortality-linked) using disease topic indicators, year, and duration
- **Naive Bayes:** Benchmarked against Logistic Regression on the same task
- **Random Forest classifier:** Classified records into 4 risk levels (Low / Moderate / High / Very High) based on `DataValueAlt` magnitude bins

### Results

| Model | Task | Key Metric |
|-------|------|-----------|
| K-Means (k=3) | State clustering by surveillance intensity | Silhouette: **0.635** |
| Logistic Regression | Mortality prediction (DeadStatus) | Accuracy: **0.77**, AUC: **0.73** |
| Naive Bayes | Mortality prediction (DeadStatus) | Accuracy: 0.42 |
| Random Forest | Risk level classification | Weighted F1: **1.00** (imbalanced), Macro F1: **0.85** |

**Honest assessment of the Random Forest result:** Raw accuracy of 0.9999 is dominated by the 238K-record `Low` class versus only 3 `Very High` records. The model correctly identifies Low risk but has 0.33 recall on Very High — the most critical class. This is a documented limitation, not a success.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| Geospatial | GeoPandas |
| Machine learning | Scikit-learn (KMeans, Logistic Regression, GaussianNB, Random Forest, DecisionTree) |
| Visualization | Matplotlib, Seaborn, Plotly (interactive state-year heatmaps) |
| Environment | Google Colab |

---

## Data Architecture

```
CDC U.S. Chronic Disease Indicators (900K+ rows, 2001–2021)
│
├── Raw: 34 columns including YearStart/End, LocationAbbr, Topic, Question,
│        DataValue, DataValueType, StratificationCategory, GeoLocation,
│        DataSource, and 10+ administrative metadata columns
│
├── Cleaning:
│   ├── Parse GeoLocation → Latitude, Longitude
│   ├── Mode imputation for DataValueUnit (by Topic × Question group)
│   ├── Drop: LocationDesc, footnote columns, LowConfidenceLimit, HighConfidenceLimit
│   └── Remove rows where GeoLocation or DataValueAlt is null → 794K clean records
│
├── Feature Engineering:
│   ├── DeadStatus — binary flag from DataSource (NVSS / Death Certificate = 1)
│   ├── diseaseDuration — YearEnd - YearStart
│   ├── One-hot encoding of 17 disease Topic values
│   └── RiskLevel — pd.cut on DataValueAlt (bins: Low/Moderate/High/Very High)
│
└── Modeling datasets:
    ├── Clustering: [States × disease frequency counts]
    ├── Mortality: [YearStart, topic indicators, diseaseDuration] → DeadStatus
    └── Risk level: [all topic indicators + DeadStatus + LocationAbbr + diseaseDuration] → RiskLevel
```

---

## Key Insights & Analytics

1. **Cancer has the highest surveillance volume (127K+ records)** but Cardiovascular Disease and Tobacco are the strongest co-predictors of mortality-linked outcomes in the Logistic Regression model — consistent with decades of epidemiological research.

2. **Three state tiers emerged from clustering:** High-surveillance states (Northeast + California) generate 3–4x more records than low-surveillance states. Low surveillance does not mean low risk — it means unmeasured risk. These are the states most needing targeted infrastructure investment.

3. **Mental Health indicators grew faster than any other topic post-2016.** This is not noise — it reflects a structural shift in both reporting practices and actual population health burden. Future resource allocation should prioritize mental health infrastructure.

4. **The 5-year duration cohort (YearEnd − YearStart = 4)** represents 15% of records and covers longitudinal surveillance programs — these records carry qualitatively different information than single-year snapshots and should be modeled separately.

5. **Naive Bayes fails badly (42% accuracy) due to feature dependence.** Disease topic indicators are correlated by design — patients with Diabetes often co-occur with Cardiovascular Disease. The independence assumption is structurally violated, confirming that Logistic Regression is the correct baseline.

---

## How to Reuse / Scale

**To run this notebook:**
1. Download the dataset from [CDC Open Data](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi)
2. Upload to Google Drive at `My Drive/DM Project/U.S._Chronic_Disease_Indicators__CDI_.csv`
3. Open `CSE469FinalProject.ipynb` in Google Colab and run all cells

**Scaling to real-time public health monitoring:**
- Replace Google Drive CSV loading with a CDC API pull (SODA API) for automated data refresh
- Replace K-Means with DBSCAN for geographic cluster detection that doesn't require a predetermined k
- Deploy the risk classifier as a FastAPI microservice for state-level real-time dashboards
- Integrate with county-level data (FIPS codes) for sub-state geographic granularity
- Add SMOTE or cost-sensitive learning to address the Very High risk class imbalance — the most critical gap for operational use

**Generalizes to:**
- Any multi-state, multi-indicator public health surveillance dataset
- International equivalents: WHO Global Health Observatory, EU health indicator datasets

---

## Challenges & What Could Be Improved

| Challenge | Improvement Path |
|-----------|-----------------|
| `Very High` risk class has only 3 samples — recall = 0.33 | SMOTE oversampling, cost-sensitive Random Forest, or anomaly detection framing |
| Risk bins based on raw DataValue magnitudes, not clinical thresholds | Consult CDC severity definitions; use quantile-based or clinically-validated cutoffs |
| 0.9999 accuracy is misleading and could misrepresent model quality | Report macro F1 (0.85) as the headline; add confusion matrix prominently in README |
| No temporal forecasting | Add ARIMA or Prophet for disease burden trend forecasting at state level |
| No causal inference | Apply difference-in-differences to evaluate the impact of specific public health policy interventions |
| Single monolithic notebook | Refactor into `src/` modules: `data_cleaning.py`, `eda.py`, `models.py` for reproducibility |
| README copy-paste errors from prior version | Fixed — all content now reflects the actual CDI dataset |
