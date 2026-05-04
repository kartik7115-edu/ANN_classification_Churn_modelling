# Learning Agent Diagnostics

These charts are generated for the project report and internal analysis. They are intentionally separate from the Streamlit demo UI.

## Generated Charts

- Feature parameter updates: [feature_parameter_updates.svg](feature_parameter_updates.svg)
- Offer parameter updates: [offer_parameter_updates.svg](offer_parameter_updates.svg)
- Before vs after churn risk: [churn_before_after.svg](churn_before_after.svg)
- Accepted vs rejected feedback trend: [feedback_acceptance_trend.svg](feedback_acceptance_trend.svg)

## Data Exports

- Flattened feature weights: [feature_parameter_updates.csv](feature_parameter_updates.csv)
- Flattened offer weights: [offer_parameter_updates.csv](offer_parameter_updates.csv)
- Feedback summary: [feedback_summary.csv](feedback_summary.csv)

## Regenerate

```powershell
.\.venv312\Scripts\python.exe report_visuals.py
```
