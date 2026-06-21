CREATE OR REPLACE VIEW analytics.mart_intervention_priority AS
WITH state_topic_year AS (
    SELECT
        LocationAbbr AS state,
        Topic AS disease_topic,
        YearStart AS year,
        COUNT(*) AS record_count,
        AVG(DataValueAlt) AS avg_indicator_value,
        AVG(CASE WHEN is_mortality = 1 THEN 1 ELSE 0 END) AS mortality_source_rate,
        COUNT(DISTINCT Stratification1) AS demographic_group_count
    FROM analytics.cdi_observations
    WHERE DataValueAlt IS NOT NULL
    GROUP BY 1, 2, 3
),

trend_features AS (
    SELECT
        *,
        LAG(avg_indicator_value) OVER (
            PARTITION BY state, disease_topic
            ORDER BY year
        ) AS previous_year_value
    FROM state_topic_year
),

scored AS (
    SELECT
        *,
        avg_indicator_value - previous_year_value AS yoy_absolute_change,

        CASE
            WHEN previous_year_value IS NULL OR previous_year_value = 0 THEN NULL
            ELSE (avg_indicator_value - previous_year_value) / previous_year_value
        END AS yoy_pct_change,

        PERCENT_RANK() OVER (ORDER BY avg_indicator_value) AS burden_score,
        PERCENT_RANK() OVER (ORDER BY mortality_source_rate) AS mortality_score,
        PERCENT_RANK() OVER (ORDER BY record_count) AS reliability_score
    FROM trend_features
)

SELECT
    state,
    disease_topic,
    year,
    record_count,
    avg_indicator_value,
    mortality_source_rate,
    yoy_absolute_change,
    yoy_pct_change,
    burden_score,
    mortality_score,
    reliability_score,

    0.40 * burden_score
  + 0.30 * mortality_score
  + 0.20 * reliability_score
  + 0.10 * COALESCE(yoy_pct_change, 0) AS intervention_priority_score
FROM scored;
