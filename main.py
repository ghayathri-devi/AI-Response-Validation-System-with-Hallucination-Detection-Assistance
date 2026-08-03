import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from retrieval_agent import retrieve_context
from relevance_agent import judge_relevance
from accuracy_agent import judge_accuracy
from hallucination_agent import detect_hallucination
from completeness_agent import judge_completeness
from verdict_agent import build_verdict
from batch_evaluator import evaluate_batch
from database import init_db, save_evaluation, get_history


app = FastAPI(
    title="AI Response Quality Evaluator",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/")
def home():
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
    evaluates each row through the full agent pipeline, and returns both
    per-row results and aggregated statistics across the batch.
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