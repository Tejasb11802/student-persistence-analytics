-- name: overall_persistence
SELECT COUNT(*)                                        AS total_learners,
       SUM(1 - persisted_next_term)                    AS at_risk_learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners;

-- name: risk_by_college
SELECT college,
       COUNT(*)                                        AS learners,
       SUM(1 - persisted_next_term)                    AS at_risk,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY college
ORDER BY at_risk_rate_pct DESC;

-- name: risk_by_enrollment_and_modality
SELECT enrollment_status, modality,
       COUNT(*)                                        AS learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY enrollment_status, modality
ORDER BY at_risk_rate_pct DESC;

-- name: risk_by_first_gen_and_pell
SELECT CASE first_generation WHEN 1 THEN 'First-Gen' ELSE 'Continuing-Gen' END AS generation_status,
       CASE pell_eligible WHEN 1 THEN 'Pell' ELSE 'Non-Pell' END                 AS pell_status,
       COUNT(*)                                        AS learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY first_generation, pell_eligible
ORDER BY at_risk_rate_pct DESC;

-- name: engagement_profile_by_outcome
SELECT CASE persisted_next_term WHEN 1 THEN 'Persisted' ELSE 'At Risk' END AS outcome,
       ROUND(AVG(lms_logins_wk1_4), 2)            AS avg_logins_wk1_4,
       ROUND(AVG(lms_logins_wk5_8), 2)            AS avg_logins_wk5_8,
       ROUND(AVG(assignment_submission_rate), 3)  AS avg_submission_rate,
       ROUND(AVG(late_submission_rate), 3)        AS avg_late_rate,
       ROUND(AVG(days_since_last_login), 1)       AS avg_days_since_login,
       ROUND(AVG(term_gpa), 2)                    AS avg_term_gpa,
       ROUND(AVG(1.0 * credits_earned / credits_attempted), 3) AS avg_credit_completion
FROM learners
GROUP BY persisted_next_term;

-- name: inactivity_bands
SELECT CASE
         WHEN days_since_last_login <= 3  THEN '0-3 days'
         WHEN days_since_last_login <= 7  THEN '4-7 days'
         WHEN days_since_last_login <= 14 THEN '8-14 days'
         ELSE '15+ days'
       END                                             AS inactivity_band,
       COUNT(*)                                        AS learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY inactivity_band
ORDER BY MIN(days_since_last_login);

-- name: financial_hold_impact
SELECT CASE financial_hold WHEN 1 THEN 'Hold' ELSE 'No Hold' END AS financial_hold,
       COUNT(*)                                        AS learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY financial_hold;

-- name: support_usage_impact
SELECT MIN(advising_visits + tutoring_sessions, 6)     AS support_touchpoints_capped,
       COUNT(*)                                        AS learners,
       ROUND(100.0 * AVG(1 - persisted_next_term), 2)  AS at_risk_rate_pct
FROM learners
GROUP BY support_touchpoints_capped
ORDER BY support_touchpoints_capped;
