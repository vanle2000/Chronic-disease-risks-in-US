-- Which chronic disease topics have the most surveillance records?
SELECT
    Topic,
    COUNT(*) AS record_count
FROM cdi
GROUP BY Topic
ORDER BY record_count DESC;

-- How has disease surveillance changed over time?
SELECT
    YearStart,
    Topic,
    COUNT(*) AS record_count
FROM cdi
GROUP BY YearStart, Topic
ORDER BY YearStart, record_count DESC;

--Which states show the highest average indicator values?
SELECT
    LocationAbbr,
    Topic,
    AVG(DataValueAlt) AS avg_indicator_value,
    COUNT(*) AS record_count
FROM cdi
GROUP BY LocationAbbr, Topic
HAVING COUNT(*) >= 100
ORDER BY avg_indicator_value DESC;

-- What is the mortality-source share by topic?
SELECT
    Topic,
    AVG(is_mortality) AS mortality_source_rate,
    COUNT(*) AS record_count
FROM cdi
GROUP BY Topic
ORDER BY mortality_source_rate DESC;

-- What is the risk-level distribution?
SELECT
    risk_level,
    COUNT(*) AS record_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct_of_total
FROM cdi
GROUP BY risk_level
ORDER BY record_count DESC;

