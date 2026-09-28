# Tableau Dashboard Build Guide

Build the **Student Persistence & At-Risk Dashboard** in Tableau Public (free) or Tableau Desktop using the CSV extracts in this folder.

## 1. Connect the Data

1. Open Tableau, choose **Connect > Text file**, select `learner_risk_scores.csv`.
2. Add `segment_profiles.csv`, `risk_by_demographic.csv`, `feature_importance.csv` and `inactivity_bands.csv` as separate data sources (Data > New Data Source).

## 2. Calculated Fields (learner_risk_scores)

| Name | Formula |
|---|---|
| At Risk Learners | `SUM([At Risk])` |
| At Risk Rate | `SUM([At Risk]) / COUNT([Student Id])` (format as percentage) |
| Avg Risk Score | `AVG([Risk Score])` |
| Tier Order | `CASE [Risk Tier] WHEN "High" THEN 1 WHEN "Elevated" THEN 2 WHEN "Moderate" THEN 3 ELSE 4 END` |

## 3. Worksheets

1. **KPI Cards**: separate sheets showing `COUNT([Student Id])`, `At Risk Learners`, `At Risk Rate`. Add model accuracy (78.7%) and AUC (0.91) from `model_comparison.csv`.
2. **Risk Tier Distribution**: Columns = Risk Tier (sort by Tier Order), Rows = `CNT(Student Id)`, Color = Risk Tier (dark red to green).
3. **At-Risk Segments**: from `segment_profiles.csv`. Rows = Segment, Columns = Learners, Tooltip = Recommended Intervention, Share Of At Risk Pct.
4. **Top Drivers**: from `feature_importance.csv`. Rows = Feature (sort descending by Importance), Columns = Importance.
5. **Risk by College**: Rows = College, Columns = At Risk Rate, sorted descending.
6. **Engagement vs GPA Scatter**: Columns = Engagement Index, Rows = Term Gpa, Color = Risk Tier, Detail = Student Id.
7. **Inactivity Bands**: from `inactivity_bands.csv`. Columns = Inactivity Band, Rows = At Risk Rate Pct.
8. **Priority Outreach List**: Rows = Student Id, Segment, Recommended Intervention; Filter Risk Tier = High; Sort by Risk Score descending.

## 4. Filters

Add Risk Tier, College, Enrollment Status, Modality and Segment as filters. Apply to all worksheets using the same data source (right click filter > Apply to Worksheets > All Using This Data Source).

## 5. Dashboard Layout (1400 x 900)

```
+-----------------------------------------------------------------+
| Title | KPI: Learners | KPI: At Risk | KPI: Rate | Accuracy | AUC |
+-------------------------------+---------------------------------+
| Risk Tier Distribution        | At-Risk Segments                |
+-------------------------------+---------------------------------+
| Top Drivers                   | Engagement vs GPA Scatter       |
+-------------------------------+---------------------------------+
| Risk by College | Inactivity  | Priority Outreach List          |
+-----------------------------------------------------------------+
```

Use a Filter action: clicking a segment bar filters the scatter and the outreach list.

## 6. Publish

**File > Save to Tableau Public**, then copy the public link into the main README under a "Live Tableau Dashboard" heading.
