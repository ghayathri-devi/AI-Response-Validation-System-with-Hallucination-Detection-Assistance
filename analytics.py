from collections import defaultdict

PASS_LABELS = {"Excellent", "Good"}
NEEDS_IMPROVEMENT_LABELS = {"Needs Improvement"}
FAIL_LABELS = {"Poor"}


def _safe_avg(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _date_only(created_at) -> str:
    """Returns just the YYYY-MM-DD portion of an ISO timestamp, for grouping by day."""
    if not created_at:
        return "unknown"
    try:
        return str(created_at)[:10]
    except Exception:
        return "unknown"


def compute_analytics(evaluations: list[dict]) -> dict:
    # Drop legacy records saved before the Verdict Agent existed (no
    # verdict_label at all) — these predate the current scoring system
    # and would otherwise show up as an "Unknown" bucket that doesn't
    # represent real current data.
    evaluations = [e for e in evaluations if e.get("verdict_label")]

    if not evaluations:
        return {
            "total_evaluations": 0,
            "pass_count": 0,
            "needs_improvement_count": 0,
            "fail_count": 0,
            "unknown_count": 0,
            "pass_rate": 0.0,
            "needs_improvement_rate": 0.0,
            "fail_rate": 0.0,
            "verdict_distribution": {},
            "avg_scores": {"accuracy": 0.0, "relevance": 0.0, "completeness": 0.0, "overall": 0.0},
            "hallucination_risk_distribution": {},
            "hallucination_frequency": 0.0,
            "trend": [],
        }

    total = len(evaluations)

    verdict_distribution: dict = defaultdict(int)
    hallucination_risk_distribution: dict = defaultdict(int)
    pass_count = 0
    needs_improvement_count = 0
    fail_count = 0
    unknown_count = 0
    flagged_count = 0

    accuracy_scores = []
    relevance_scores = []
    completeness_scores = []
    overall_scores = []

    trend_by_date: dict = defaultdict(list)

    for e in evaluations:
        label = e.get("verdict_label") or "Unknown"
        verdict_distribution[label] += 1

        if label in PASS_LABELS:
            pass_count += 1
        elif label in NEEDS_IMPROVEMENT_LABELS:
            needs_improvement_count += 1
        elif label in FAIL_LABELS:
            fail_count += 1
        else:
            unknown_count += 1

        risk = e.get("hallucination_risk") or "Unknown"
        hallucination_risk_distribution[risk] += 1

        flagged = e.get("hallucination_flagged_statements") or []
        if isinstance(flagged, list) and len(flagged) > 0:
            flagged_count += 1

        if e.get("accuracy") is not None:
            accuracy_scores.append(e["accuracy"])
        if e.get("relevance") is not None:
            relevance_scores.append(e["relevance"])
        if e.get("completeness") is not None:
            completeness_scores.append(e["completeness"])
        if e.get("overall_score") is not None:
            overall_scores.append(e["overall_score"])
            date_key = _date_only(e.get("created_at"))
            trend_by_date[date_key].append(e["overall_score"])

    trend = [
        {"date": date_key, "avg_overall_score": _safe_avg(scores), "count": len(scores)}
        for date_key, scores in sorted(trend_by_date.items())
        if date_key != "unknown"
    ]

    return {
        "total_evaluations": total,
        "pass_count": pass_count,
        "needs_improvement_count": needs_improvement_count,
        "fail_count": fail_count,
        "unknown_count": unknown_count,
        "pass_rate": round(pass_count / total, 2) if total else 0.0,
        "needs_improvement_rate": round(needs_improvement_count / total, 2) if total else 0.0,
        "fail_rate": round(fail_count / total, 2) if total else 0.0,
        "verdict_distribution": dict(verdict_distribution),
        "avg_scores": {
            "accuracy": _safe_avg(accuracy_scores),
            "relevance": _safe_avg(relevance_scores),
            "completeness": _safe_avg(completeness_scores),
            "overall": _safe_avg(overall_scores),
        },
        "hallucination_risk_distribution": dict(hallucination_risk_distribution),
        "hallucination_frequency": round(flagged_count / total, 2) if total else 0.0,
        "trend": trend,
    }


if __name__ == "__main__":
    # Quick standalone test with a few fake evaluation records
    sample = [
        {"verdict_label": "Excellent", "hallucination_risk": "Low", "hallucination_flagged_statements": [],
         "accuracy": 0.9, "relevance": 0.95, "completeness": 0.9, "overall_score": 0.92, "created_at": "2026-08-01T10:00:00"},
        {"verdict_label": "Poor", "hallucination_risk": "High", "hallucination_flagged_statements": ["fake claim"],
         "accuracy": 0.1, "relevance": 0.5, "completeness": 0.3, "overall_score": 0.2, "created_at": "2026-08-01T11:00:00"},
        {"verdict_label": "Needs Improvement", "hallucination_risk": "Medium", "hallucination_flagged_statements": [],
         "accuracy": 0.5, "relevance": 0.6, "completeness": 0.5, "overall_score": 0.5, "created_at": "2026-08-02T09:00:00"},
    ]
    print(compute_analytics(sample))