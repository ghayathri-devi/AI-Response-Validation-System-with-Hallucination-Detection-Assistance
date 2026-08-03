from typing import Optional

# Weights must sum to 1.0
WEIGHTS = {
    "accuracy": 0.35,
    "hallucination": 0.30,
    "relevance": 0.20,
    "completeness": 0.15,
}

_RISK_TO_SCORE = {"Low": 1.0, "Medium": 0.6, "High": 0.2}


def _verdict_label(overall_score: float) -> str:
    if overall_score >= 0.85:
        return "Excellent"
    elif overall_score >= 0.65:
        return "Good"
    elif overall_score >= 0.4:
        return "Needs Improvement"
    else:
        return "Poor"


def _build_summary(
    overall_score: float,
    verdict_label: str,
    hallucination_risk: str,
    flagged_statements: list[str],
    missing_aspects: list[str],
) -> str:
    parts = [f"Overall verdict: {verdict_label} ({overall_score * 100:.0f}/100)."]

    if flagged_statements:
        parts.append(
            f"{len(flagged_statements)} claim(s) could not be verified against the retrieved context."
        )
    else:
        parts.append("No unsupported claims were flagged.")

    if missing_aspects:
        parts.append(
            f"The response left {len(missing_aspects)} aspect(s) of the question unaddressed."
        )
    else:
        parts.append("The response addressed all required aspects of the question.")

    parts.append(f"Hallucination risk was assessed as {hallucination_risk}.")

    return " ".join(parts)


def build_verdict(
    relevance_result: dict,
    accuracy_result: dict,
    completeness_result: dict,
    hallucination_result: dict,
) -> dict:
    """
    Aggregates the four agents' outputs into a single weighted verdict.

    Expects each *_result dict to already contain the fields produced by
    that agent (e.g. relevance_result["relevance_score"]).

    Returns:
        {
            "overall_score": float (0.0-1.0),
            "verdict_label": "Excellent" | "Good" | "Needs Improvement" | "Poor",
            "summary": str,
            "weights_used": dict
        }
    """

    relevance_score = relevance_result["relevance_score"]
    accuracy_score = accuracy_result["accuracy_score"]
    completeness_score = completeness_result["completeness_score"]
    hallucination_score_proxy = _RISK_TO_SCORE.get(
        hallucination_result["hallucination_risk"], 0.5
    )

    overall_score = round(
        relevance_score * WEIGHTS["relevance"]
        + accuracy_score * WEIGHTS["accuracy"]
        + completeness_score * WEIGHTS["completeness"]
        + hallucination_score_proxy * WEIGHTS["hallucination"],
        2,
    )

    verdict_label = _verdict_label(overall_score)

    summary = _build_summary(
        overall_score=overall_score,
        verdict_label=verdict_label,
        hallucination_risk=hallucination_result["hallucination_risk"],
        flagged_statements=hallucination_result.get("flagged_statements", []),
        missing_aspects=completeness_result.get("missing_aspects", []),
    )

    return {
        "overall_score": overall_score,
        "verdict_label": verdict_label,
        "summary": summary,
        "weights_used": WEIGHTS,
    }


if __name__ == "__main__":
    # Quick standalone test with representative agent outputs
    relevance_result = {"relevance_score": 0.9, "reasoning": "Directly answers the question."}
    accuracy_result = {"accuracy_score": 0.4, "evidence": "Some claims unsupported."}
    completeness_result = {"completeness_score": 0.7, "missing_aspects": ["why it matters"]}
    hallucination_result = {
        "hallucination_risk": "Medium",
        "flagged_statements": ["invented statistic about market size"],
    }

    verdict = build_verdict(relevance_result, accuracy_result, completeness_result, hallucination_result)
    print(verdict)