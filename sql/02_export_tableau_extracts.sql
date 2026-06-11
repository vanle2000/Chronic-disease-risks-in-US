-- Tableau extract exports
-- Run after sql/01_create_analytics_views.sql

COPY analytics.vw_dashboard_kpis
TO 'tableau/dashboard_kpis.csv'
WITH (HEADER, DELIMITER ',');

COPY analytics.mart_topic_year_trends
TO 'tableau/topic_year_trends.csv'
WITH (HEADER, DELIMITER ',');

COPY analytics.mart_state_topic_summary
TO 'tableau/state_topic_summary.csv'
WITH (HEADER, DELIMITER ',');

COPY analytics.mart_risk_distribution
TO 'tableau/risk_distribution.csv'
WITH (HEADER, DELIMITER ',');

COPY analytics.mart_mortality_by_topic
TO 'tableau/mortality_by_topic.csv'
WITH (HEADER, DELIMITER ',');

COPY analytics.mart_demographic_risk
TO 'tableau/demographic_risk.csv'
WITH (HEADER, DELIMITER ',');
