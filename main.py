import io
import threading
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader

from retrieval_agent import retrieve_context
from relevance_agent import judge_relevance
from accuracy_agent import judge_accuracy
from hallucination_agent import detect_hallucination
from completeness_agent import judge_completeness
from verdict_agent import build_verdict
from batch_evaluator import evaluate_batch
from analytics import compute_analytics
from report_generator import generate_report_pdf
from knowledge_builder import build_knowledge_base
from database import init_db, save_evaluation, get_history, get_all_evaluations


app = FastAPI(
    title="AI Response Quality Evaluator",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# Rebuild the RAG knowledge base on startup. On Render's free tier the
# filesystem is wiped on every redeploy/spin-down, so chroma_db never
# persists between deployments — this recreates it fresh every time the
# server starts. build_knowledge_base() is a no-op if data already exists
# (relevant for local development, where you don't want to re-embed
# everything on every restart).
#
# Runs in a background thread rather than blocking here — Render (and
# similar platforms) expect the app to open its port quickly after
# startup, and running this synchronously delayed that past the health
# check's patience, causing "No open ports detected" failures. Running
# it in the background lets uvicorn bind the port immediately; requests
# that need retrieval before the build finishes will just get no
# context back until it completes (a few minutes at most).
threading.Thread(target=build_knowledge_base, daemon=True).start()


@app.get("/api/health")
def health():
    return {
        "message": "AI Response Quality Evaluator API Running"
    }


def extract_pdf_text(file_bytes: bytes, max_chars: int = 3000) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
        if len(text) >= max_chars:
            break
    return text.strip()[:max_chars]


@app.post("/evaluate")
async def evaluate(
    question: str = Form(...),
    answer: str = Form(...),
    reference_answer: Optional[str] = Form(None),
    source_pdf: Optional[UploadFile] = File(None),
):

    retrieved_context = retrieve_context(question, top_k=2)

    used_source_pdf = False
    if source_pdf is not None:
        pdf_bytes = await source_pdf.read()
        pdf_text = extract_pdf_text(pdf_bytes)
        if pdf_text:
            retrieved_context.append(pdf_text)
            used_source_pdf = True

    # --- Relevance Judge Agent ---
    relevance_result = judge_relevance(question, answer)

    # --- Accuracy Judge Agent ---
    accuracy_result = judge_accuracy(
        answer=answer,
        reference_answer=reference_answer,
        contexts=retrieved_context,
    )

    # --- Hallucination Detection Agent ---
    hallucination_result = detect_hallucination(
        answer=answer,
        contexts=retrieved_context,
        reference_answer=reference_answer,
    )

    # --- Completeness Judge Agent ---
    completeness_result = judge_completeness(question, answer)

    # --- Verdict Agent — aggregates the four results above (no LLM call) ---
    verdict_result = build_verdict(
        relevance_result=relevance_result,
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        hallucination_result=hallucination_result,
    )

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

        "used_reference_answer": accuracy_result["used_reference_answer"],
    }

    save_evaluation(
        question=question,
        answer=answer,
        reference_answer=reference_answer,
        used_source_pdf=used_source_pdf,
        retrieved_context=retrieved_context,
        evaluation=evaluation,
        evaluation_mode="single",
    )

    return {
        "status": "success",
        "question": question,
        "retrieved_context": retrieved_context,
        "used_source_pdf": used_source_pdf,
        "evaluation": evaluation,
    }


@app.get("/history")
def history(limit: int = 20):
    return {
        "status": "success",
        "history": get_history(limit)
    }


@app.post("/batch-evaluate")
async def batch_evaluate(csv_file: UploadFile = File(...)):
    """
    Accepts a CSV of question/answer (and optional reference_answer) pairs,
    evaluates each row through the full agent pipeline, persists each row
    to Supabase (tagged evaluation_mode='batch'), and returns both per-row
    results and aggregated statistics across the batch.
    """

    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    file_bytes = await csv_file.read()

    try:
        result = evaluate_batch(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "row_count": result["aggregate"]["count"],
        "results": result["results"],
        "aggregate": result["aggregate"],
    }


@app.get("/analytics")
def analytics(
    limit: int = 500,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: Optional[str] = None,
):
    """
    Powers the Evaluation Scoring Dashboard: pass/needs-improvement/fail
    counts, average per-dimension scores, hallucination frequency, verdict
    distribution, and a quality trend over time — computed across stored
    evaluations, optionally filtered by date range and/or evaluation mode
    (single vs batch).

    Query params (all optional):
      start_date=YYYY-MM-DD
      end_date=YYYY-MM-DD
      mode=single | batch
    """
    evaluations = get_all_evaluations(limit=limit, start_date=start_date, end_date=end_date, mode=mode)
    stats = compute_analytics(evaluations)

    return {
        "status": "success",
        **stats,
    }


@app.post("/export-report")
async def export_report(payload: dict):
    """
    Generates a structured PDF report from an already-completed batch
    evaluation. Expects the same JSON shape the frontend already has after
    a /batch-evaluate call: { "results": [...], "aggregate": {...} }.
    No new LLM calls are made — this only formats data that was already
    computed and displayed in the Batch tab.
    """
    results = payload.get("results")
    aggregate = payload.get("aggregate")

    if not results or not aggregate:
        raise HTTPException(status_code=400, detail="Request must include 'results' and 'aggregate' from a completed batch evaluation")

    pdf_bytes = generate_report_pdf(results, aggregate)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=evaluation_report.pdf"},
    )


# Serves index.html, style.css, script.js directly from FastAPI.
# Must be mounted LAST — routes above are matched first, and this
# static mount only catches whatever isn't already an API route
# (including "/" itself, via html=True).
app.mount("/", StaticFiles(directory=".", html=True), name="static")