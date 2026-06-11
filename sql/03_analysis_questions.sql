-- SQL analysis questions for the Chronic Disease Risk Analytics project
-- Run this file after sql/01_create_analytics_views.sql

-- Question 1
-- What is the overall size and coverage of the dataset
SELECT *
FROM analytics.vw_dashboard_kpis;

-- Question 2
-- Which chronic disease topics have the highest surveillance volume
SELECT
    Topic,
    SUM(record_count) AS total_records,
    AVG(avg_indicator_value) AS avg_indicator_value,
    AVG(mortality_source_rate) AS avg_mortality_source_rate
FROM analytics.mart_topic_year_trends
GROUP BY Topic
ORDER BY total_records DESC;

-- Question 3
-- Which topics are growing the most over time based on record volume
WITH yearly AS (
    SELECT
        Topic,
        YearStart,
        SUM(record_count) AS record_count
    FROM analytics.mart_topic_year_trends
    GROUP BY Topic, YearStart
), endpoints AS (
    SELECT
        Topic,
        MIN(YearStart) AS first_year,
        MAX(YearStart) AS latest_year
    FROM yearly
    GROUP BY Topic
), joined AS (
    SELECT
        e.Topic,
        e.first_year,
        y1.record_count AS first_year_records,
        e.latest_year,
        y2.record_count AS latest_year_records
    FROM endpoints e
    LEFT JOIN yearly y1
        ON e.Topic = y1.Topic
       AND e.first_year = y1.YearStart
    LEFT JOIN yearly y2
        ON e.Topic = y2.Topic
       AND e.latest_year = y2.YearStart
)
SELECT
    Topic,
    first_year,
    first_year_records,
    latest_year,
    latest_year_records,
    latest_year_records - first_year_records AS record_change
FROM joined
ORDER BY record_change DESC;

-- Question 4
-- Which state and topic pairs have the highest average indicator values
SELECT
    LocationAbbr,
    Topic,
    record_count,
    avg_indicator_value,
    mortality_source_rate
FROM analytics.mart_state_topic_summary
WHERE record_count >= 100
ORDER BY avg_indicator_value DESC
LIMIT 50;

-- Question 5
-- Which topics are most connected to mortality related records
SELECT
    Topic,
    record_count,
    mortality_source_records,
    mortality_source_rate
FROM analytics.mart_mortality_by_topic
ORDER BY mortality_source_rate DESC;

-- Question 6
-- How imbalanced is the risk level target before modeling
SELECT
    risk_level,
    record_count,
    pct_of_records
FROM analytics.mart_risk_distribution
ORDER BY record_count DESC;

-- Question 7
-- Which demographic groups show elevated average indicator values
SELECT
    Topic,
    StratificationCategory1,
    Stratification1,
    record_count,
    avg_indicator_value,
    mortality_source_rate
FROM analytics.mart_demographic_risk
WHERE record_count >= 100
ORDER BY avg_indicator_value DESC
LIMIT 100;

-- Question 8
-- What is the modeling feature base size
SELECT
    COUNT(*) AS feature_rows,
    COUNT(DISTINCT LocationAbbr) AS states,
    COUNT(DISTINCT Topic) AS topics,
    MIN(YearStart) AS first_year,
    MAX(YearStart) AS latest_year
FROM analytics.ml_feature_base;
