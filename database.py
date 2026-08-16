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


def save_evaluation(question, answer, reference_answer, used_source_pdf, retrieved_context, evaluation, evaluation_mode="single"):
    supabase.table("evaluations").insert({
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "used_source_pdf": bool(used_source_pdf),
        "retrieved_context": retrieved_context,  # jsonb column — pass the list directly
        "evaluation_mode": evaluation_mode,  # "single" or "batch"

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


def get_all_evaluations(limit: int = 500, start_date: str = None, end_date: str = None, mode: str = None):
    """
    Fetches evaluation records for the Analytics dashboard, oldest first
    (needed for the quality-over-time trend chart to read left-to-right
    chronologically). Supports optional filtering:

      start_date / end_date : "YYYY-MM-DD" strings, inclusive range on created_at
      mode                  : "single" or "batch", filters on evaluation_mode

    All filters are optional — passing none returns the full (capped) history.
    """
    query = supabase.table("evaluations").select("*")

    if start_date:
        query = query.gte("created_at", f"{start_date}T00:00:00")
    if end_date:
        query = query.lte("created_at", f"{end_date}T23:59:59")
    if mode:
        query = query.eq("evaluation_mode", mode)

    result = query.order("created_at", desc=False).limit(limit).execute()
    return result.data