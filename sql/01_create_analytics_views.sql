CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE TABLE analytics.cdi_observations AS
SELECT *
FROM read_parquet('data/processed/cdi_processed.parquet');

CREATE OR REPLACE VIEW analytics.vw_dashboard_kpis AS
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT LocationAbbr) AS state_count,
    COUNT(DISTINCT Topic) AS topic_count,
    MIN(YearStart) AS first_year,
    MAX(YearStart) AS latest_year,
    AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate
FROM analytics.cdi_observations;

CREATE OR REPLACE VIEW analytics.mart_topic_year_trends AS
SELECT
    YearStart,
    Topic,
    COUNT(*) AS record_count,
    AVG(DataValueAlt) AS avg_indicator_value,
    AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate
FROM analytics.cdi_observations
GROUP BY
    YearStart,
    Topic;

CREATE OR REPLACE VIEW analytics.mart_state_topic_summary AS
SELECT
    LocationAbbr,
    Topic,
    COUNT(*) AS record_count,
    AVG(DataValueAlt) AS avg_indicator_value,
    AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate,
    MIN(Latitude) AS latitude,
    MIN(Longitude) AS longitude
FROM analytics.cdi_observations
GROUP BY
    LocationAbbr,
    Topic;

CREATE OR REPLACE VIEW analytics.mart_risk_distribution AS
SELECT
    risk_level,
    COUNT(*) AS record_count,
    100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct_of_records
FROM analytics.cdi_observations
GROUP BY
    risk_level;

CREATE OR REPLACE VIEW analytics.mart_mortality_by_topic AS
SELECT
    Topic,
    COUNT(*) AS record_count,
    SUM(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_records,
    AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate
FROM analytics.cdi_observations
GROUP BY
    Topic;

CREATE OR REPLACE VIEW analytics.mart_demographic_risk AS
SELECT
    Topic,
    StratificationCategory1,
    Stratification1,
    COUNT(*) AS record_count,
    AVG(DataValueAlt) AS avg_indicator_value,
    AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate
FROM analytics.cdi_observations
WHERE StratificationCategory1 IS NOT NULL
  AND Stratification1 IS NOT NULL
GROUP BY
    Topic,
    StratificationCategory1,
    Stratification1;

CREATE OR REPLACE VIEW analytics.ml_feature_base AS
SELECT
    YearStart,
    YearEnd,
    LocationAbbr,
    Topic,
    Question,
    DataValueType,
    StratificationCategory1,
    Stratification1,
    DataValueAlt,
    value_normalized,
    disease_duration,
    is_mortality,
    risk_level,
    Latitude,
    Longitude
FROM analytics.cdi_observations
WHERE DataValueAlt IS NOT NULL;
