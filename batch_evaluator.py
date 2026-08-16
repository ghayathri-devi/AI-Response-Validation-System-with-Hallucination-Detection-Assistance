import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from retrieval_agent import retrieve_context
from relevance_agent import judge_relevance
from accuracy_agent import judge_accuracy
from hallucination_agent import detect_hallucination
from completeness_agent import judge_completeness
from verdict_agent import build_verdict
from database import save_evaluation


MAX_BATCH_ROWS = 20


# =====================================================
# CSV parsing
# =====================================================

def parse_csv(file_bytes: bytes) -> list[dict]:
    # utf-8-sig strips a BOM if the CSV was exported from Excel
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    required_columns = {"question", "answer"}
    fieldnames = set(reader.fieldnames or [])
    if not required_columns.issubset(fieldnames):
        raise ValueError(
            f"CSV must contain 'question' and 'answer' columns. Found: {sorted(fieldnames) or 'none'}"
        )

    rows = []
    for row in reader:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        reference_answer = (row.get("reference_answer") or "").strip() or None

        if question and answer:
            rows.append({
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
            })

    return rows


# =====================================================
# Evaluate a single row through the full agent pipeline
# =====================================================

def evaluate_single_row(row: dict) -> dict:
    question = row["question"]
    answer = row["answer"]
    reference_answer = row["reference_answer"]

    retrieved_context = retrieve_context(question, top_k=2)

    with ThreadPoolExecutor(max_workers=4) as executor:
        relevance_future = executor.submit(judge_relevance, question, answer)
        accuracy_future = executor.submit(judge_accuracy, answer, reference_answer, retrieved_context)
        hallucination_future = executor.submit(detect_hallucination, answer, retrieved_context, reference_answer)
        completeness_future = executor.submit(judge_completeness, question, answer)

        relevance_result = relevance_future.result()
        accuracy_result = accuracy_future.result()
        hallucination_result = hallucination_future.result()
        completeness_result = completeness_future.result()

    verdict_result = build_verdict(
        relevance_result=relevance_result,
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        hallucination_result=hallucination_result,
    )

    # Build the full evaluation dict — same shape save_evaluation() expects
    # for single evaluations, so batch rows show up in history/analytics
    # with the same richness (reasoning, evidence, etc.), not just scores
    evaluation = {
        "relevance": relevance_result["relevance_score"],
        "relevance_reasoning": relevance_result["reasoning"],

        "accuracy": accuracy_result["accuracy_score"],
        "accuracy_evidence": accuracy_result["evidence"],
        "accuracy_supporting_excerpt": accuracy_result["supporting_excerpt"],

        "completeness": completeness_result["completeness_score"],
        "completeness_reasoning": completeness_result["reasoning"],
        "completeness_missing_aspects": completeness_result["missing_aspects"],

        "hallucination_risk": hallucination_result["hallucination_risk"],
        "hallucination_flagged_statements": hallucination_result["flagged_statements"],
        "hallucination_reasoning": hallucination_result["reasoning"],

        "overall_score": verdict_result["overall_score"],
        "verdict_label": verdict_result["verdict_label"],
        "verdict_summary": verdict_result["summary"],
    }

    # Persist this row to Supabase, tagged as a batch submission so the
    # Analytics dashboard can filter by evaluation mode
    save_evaluation(
        question=question,
        answer=answer,
        reference_answer=reference_answer,
        used_source_pdf=False,
        retrieved_context=retrieved_context,
        evaluation=evaluation,
        evaluation_mode="batch",
    )

    # Return the flattened summary used for the per-row results table
    return {
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "relevance": relevance_result["relevance_score"],
        "accuracy": accuracy_result["accuracy_score"],
        "completeness": completeness_result["completeness_score"],
        "hallucination_risk": hallucination_result["hallucination_risk"],
        "hallucination_flagged_statements": hallucination_result["flagged_statements"],
        "overall_score": verdict_result["overall_score"],
        "verdict_label": verdict_result["verdict_label"],
        "verdict_summary": verdict_result["summary"],
    }


# =====================================================
# Aggregate stats across the whole batch
# =====================================================

def compute_aggregate(results: list[dict]) -> dict:
    if not results:
        return {
            "count": 0,
            "avg_relevance": 0,
            "avg_accuracy": 0,
            "avg_completeness": 0,
            "avg_overall_score": 0,
            "verdict_distribution": {},
            "total_flagged_hallucinations": 0,
        }

    count = len(results)

    verdict_distribution: dict[str, int] = {}
    for r in results:
        label = r["verdict_label"]
        verdict_distribution[label] = verdict_distribution.get(label, 0) + 1

    return {
        "count": count,
        "avg_relevance": round(sum(r["relevance"] for r in results) / count, 2),
        "avg_accuracy": round(sum(r["accuracy"] for r in results) / count, 2),
        "avg_completeness": round(sum(r["completeness"] for r in results) / count, 2),
        "avg_overall_score": round(sum(r["overall_score"] for r in results) / count, 2),
        "verdict_distribution": verdict_distribution,
        "total_flagged_hallucinations": sum(len(r["hallucination_flagged_statements"]) for r in results),
    }


# =====================================================
# Public entry point
# =====================================================

def evaluate_batch(file_bytes: bytes) -> dict:
    rows = parse_csv(file_bytes)

    if not rows:
        raise ValueError(
            "No valid rows found in CSV — each row needs non-empty 'question' and 'answer' values."
        )

    if len(rows) > MAX_BATCH_ROWS:
        raise ValueError(
            f"Batch size ({len(rows)} rows) exceeds the maximum of {MAX_BATCH_ROWS} rows per request. "
            f"This limit exists to avoid excessive LLM API usage in a single request — split the CSV "
            f"into smaller files and submit them separately."
        )

    results = [evaluate_single_row(row) for row in rows]
    aggregate = compute_aggregate(results)

    return {
        "results": results,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    # Quick standalone test using an in-memory sample CSV
    sample_csv = (
        "question,answer,reference_answer\n"
        "What is the capital of France?,Paris is the capital of France.,Paris\n"
        "What is 2+2?,2+2 equals 5.,4\n"
    ).encode("utf-8")

    result = evaluate_batch(sample_csv)
    print(result["aggregate"])
    for r in result["results"]:
        print(r["question"], "->", r["verdict_label"], r["overall_score"])