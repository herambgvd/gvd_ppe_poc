"""
# ======================================
# ANALYTICS ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module calculates dashboard KPIs from events, violations, and live worker stats.
- It separates analytics from UI routes so the same metrics can later be exposed to BI tools or APIs.
- Enterprise dashboards need camera-wise, violation-wise, hourly, and compliance metrics, not just raw detections.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .database import DB


class AnalyticsEngine:
    def summary(self) -> Dict:
        events = DB.list_events(limit=1000)
        violations = DB.list_violations(limit=2000)
        active = [e for e in events if e.get("state") in {"NEW", "ACTIVE"}]
        total = len(violations)
        type_counts = Counter(v.get("violation_type") for v in violations)
        camera_counts = Counter(v.get("camera_id") for v in violations)

        # Approximate compliance: active violations lower the score; this can be upgraded with worker-time denominator.
        compliance_pct = max(0.0, 100.0 - min(100.0, len(active) * 8.0 + total * 0.25))

        return {
            "total_violations": total,
            "active_events": len(active),
            "resolved_events": len([e for e in events if e.get("state") == "RESOLVED"]),
            "compliance_pct": round(compliance_pct, 2),
            "violations_by_type": dict(type_counts),
            "violations_by_camera": dict(camera_counts),
            "recent_events": events[:25],
            "timeline": self.timeline(violations),
        }

    def timeline(self, violations: List[Dict]) -> List[Dict]:
        buckets = defaultdict(int)
        for v in violations:
            ts = v.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            bucket = dt.replace(minute=0, second=0, microsecond=0).isoformat()
            buckets[bucket] += 1
        return [{"bucket": k, "count": v} for k, v in sorted(buckets.items())[-24:]]


ANALYTICS = AnalyticsEngine()
