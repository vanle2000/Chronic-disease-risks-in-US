# Tableau Public Publishing Checklist

This guide is for publishing the Chronic Disease Risk Analytics dashboard to Tableau Public.

## Files to connect in Tableau

Use the CSV files exported from the SQL pipeline.

```text
tableau/dashboard_kpis.csv
tableau/topic_year_trends.csv
tableau/state_topic_summary.csv
tableau/risk_distribution.csv
tableau/mortality_by_topic.csv
tableau/demographic_risk.csv
```

## Dashboard 1

### Executive Public Health Overview

Goal: Give a fast national view of chronic disease surveillance.

Use these sheets:

1. KPI cards from `dashboard_kpis.csv`
2. Disease topic volume bar chart from `topic_year_trends.csv`
3. Topic trend line chart from `topic_year_trends.csv`
4. State map from `state_topic_summary.csv`
5. Topic and year filters

Suggested title:

```text
Chronic Disease Risk Analytics in the U.S.
```

Suggested subtitle:

```text
SQL and Tableau analysis of CDC Chronic Disease Indicators
```

## Dashboard 2

### State and Topic Deep Dive

Goal: Help users compare states and disease topics.

Use these sheets:

1. State topic heatmap from `state_topic_summary.csv`
2. Ranked state topic table from `state_topic_summary.csv`
3. Mortality source rate by topic from `mortality_by_topic.csv`
4. State, topic, year, and stratification filters

## Dashboard 3

### Modeling Readiness

Goal: Show whether the data is ready for predictive modeling.

Use these sheets:

1. Risk level distribution from `risk_distribution.csv`
2. Demographic risk comparison from `demographic_risk.csv`
3. Feature base notes from the SQL layer
4. Model summary and feature importance after the ML notebook runs

## Recommended Tableau calculated fields

### Mortality source percent

```text
[Mortality Source Rate] * 100
```

### Average indicator rounded

```text
ROUND([Avg Indicator Value], 2)
```

### Record count label

```text
STR([Record Count]) + " records"
```

## Publishing steps

1. Open Tableau Public or Tableau Desktop
2. Connect to the CSV files inside the `tableau` folder
3. Build the three dashboards listed above
4. Add clear titles, filters, and source notes
5. Save the workbook to Tableau Public
6. Copy the public dashboard link
7. Add the dashboard link to the main README

## Source note for the dashboard

Use this note in the dashboard footer.

```text
Data source: CDC Chronic Disease Indicators. Analysis created with SQL, Tableau, Python, and machine learning notebooks.
```

## README link format

After publishing, add this section to the main README.

```markdown
## Tableau Public Dashboard

I published the interactive dashboard here:

[View the Tableau Public Dashboard](PASTE_YOUR_TABLEAU_PUBLIC_LINK_HERE)
```
