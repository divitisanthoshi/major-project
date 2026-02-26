"""
Session Manager - Save analytics, export for Power BI.

Stores per-session data and exports to CSV/JSON for dashboards.
"""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any


SESSION_DIR = "sessions"
EXPORT_DIR = "exports"


def ensure_dirs():
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


def save_session(
    exercise: str,
    target_reps: int,
    completed_reps: int,
    scores: List[float],
    feedback_log: List[str],
    duration_sec: float,
) -> str:
    """
    Save session to JSON. Returns path to saved file.
    """
    ensure_dirs()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{exercise}_{session_id}.json"
    path = os.path.join(SESSION_DIR, filename)

    avg_score = sum(scores) / len(scores) if scores else 0
    data = {
        "session_id": session_id,
        "exercise": exercise,
        "target_reps": target_reps,
        "completed_reps": completed_reps,
        "average_score": round(avg_score * 100, 1),
        "duration_seconds": round(duration_sec, 1),
        "timestamp": datetime.now().isoformat(),
        "scores_per_frame": [round(s * 100, 1) for s in scores[-100:]],  # Last 100
        "feedback_samples": feedback_log[-20:],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def export_for_power_bi(sessions_paths: List[str] = None) -> str:
    """
    Export sessions to CSV for Power BI import.
    If no paths given, uses all .json in sessions/.
    Called automatically after each session (ESC); no manual step required.
    """
    ensure_dirs()
    if sessions_paths is None:
        if not os.path.isdir(SESSION_DIR):
            return os.path.join(EXPORT_DIR, f"rehab_analytics_{datetime.now().strftime('%Y%m%d')}.csv")
        sessions_paths = [
            os.path.join(SESSION_DIR, f)
            for f in os.listdir(SESSION_DIR)
            if f.endswith(".json")
        ]

    out_path = os.path.join(EXPORT_DIR, f"rehab_analytics_{datetime.now().strftime('%Y%m%d')}.csv")
    rows = []

    for p in sessions_paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            rows.append({
                "session_id": d.get("session_id", ""),
                "timestamp": d.get("timestamp", ""),
                "exercise": d.get("exercise", ""),
                "target_reps": d.get("target_reps", 0),
                "completed_reps": d.get("completed_reps", 0),
                "completion_pct": round(100 * d.get("completed_reps", 0) / max(1, d.get("target_reps", 1)), 1),
                "average_score": d.get("average_score", 0),
                "duration_seconds": d.get("duration_seconds", 0),
            })
        except Exception:
            pass

    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    return out_path


def load_recent_sessions(n: int = 10) -> List[Dict]:
    """Load last n sessions for dashboard."""
    ensure_dirs()
    files = sorted(
        [os.path.join(SESSION_DIR, f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
        key=os.path.getmtime,
        reverse=True,
    )[:n]
    out = []
    for p in files:
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out
