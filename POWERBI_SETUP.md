# Power BI Dashboard Setup

This project exports session data to CSV for use in Power BI.

## Export Location

After each session (when you press ESC), data is saved to:

- **Sessions**: `sessions/session_<exercise>_<timestamp>.json`
- **Power BI export**: `exports/rehab_analytics_<date>.csv`

## CSV Format for Power BI

| Column          | Description                    |
|-----------------|--------------------------------|
| session_id      | Unique session identifier      |
| timestamp       | When the session occurred      |
| exercise        | Exercise name                  |
| target_reps     | Target repetitions             |
| completed_reps  | Actual repetitions completed   |
| completion_pct  | Completed / target (%)         |
| average_score   | Average correctness (0-100)   |
| duration_seconds| Session length (seconds)       |

## Import into Power BI

1. Open Power BI Desktop
2. **Get Data** → **Text/CSV** → select `exports/rehab_analytics_YYYYMMDD.csv`
3. Click **Load**
4. Build visuals:
   - **Line chart**: Average score over time (by timestamp)
   - **Bar chart**: Completed reps by exercise
   - **Gauge**: Overall completion percentage
   - **Table**: Session details

## Manual Export

To export all sessions to CSV (e.g., after multiple runs):

```python
from src.session_manager import export_for_power_bi
export_for_power_bi()  # Exports to exports/rehab_analytics_<date>.csv
```

Or from Python REPL in project directory:

```python
import sys; sys.path.insert(0, '.')
from src.session_manager import export_for_power_bi
path = export_for_power_bi()
print(f"Exported to {path}")
```

## Refreshing Data

Re-run the application and complete sessions. Each ESC (end session) updates the CSV. In Power BI, use **Refresh** to reload the data.
