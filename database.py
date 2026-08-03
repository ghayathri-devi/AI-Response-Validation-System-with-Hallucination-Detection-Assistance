from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime, timezone

from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    """
    No-op for Supabase — the 'evaluations' table is created/altered once
    manually via the Supabase SQL Editor, not at app startup.
    Kept as a function so main.py doesn't need to change.
    """
    pass


def save_evaluation(question, answer, reference_answer, used_source_pdf, retrieved_context, evaluation):
    supabase.table("evaluations").insert({
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "used_source_pdf": bool(used_source_pdf),
        "retrieved_context": retrieved_context,  # jsonb column — pass the list directly

        "accuracy": evaluation["accuracy"],
        "accuracy_evidence": evaluation["accuracy_evidence"],
        "accuracy_supporting_excerpt": evaluation["accuracy_supporting_excerpt"],

        "relevance": evaluation["relevance"],
        "relevance_reasoning": evaluation["relevance_reasoning"],

        "completeness": evaluation["completeness"],
        "completeness_reasoning": evaluation["completeness_reasoning"],
        "completeness_missing_aspects": evaluation["completeness_missing_aspects"],  # jsonb column

        "hallucination_risk": evaluation["hallucination_risk"],
        "hallucination_flagged_statements": evaluation["hallucination_flagged_statements"],  # jsonb column
        "hallucination_reasoning": evaluation["hallucination_reasoning"],

        "overall_score": evaluation["overall_score"],
        "verdict_label": evaluation["verdict_label"],
        "verdict_summary": evaluation["verdict_summary"],

        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def get_history(limit: int = 20):
    result = (
        supabase.table("evaluations")
        .select("*")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data