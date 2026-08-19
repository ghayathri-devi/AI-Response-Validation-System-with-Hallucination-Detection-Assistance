from dotenv import load_dotenv
load_dotenv()

import os
import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from sentence_transformers import util
from knowledge_builder import get_embedding_model

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

#if llm fails, uses the shared embedding model (avoids loading a
# duplicate copy of the model — important for memory on constrained hosts)


def _fallback_relevance(question: str, answer: str) -> dict:
    st_model = get_embedding_model()
    emb_a = st_model.encode(question, convert_to_tensor=True)
    emb_b = st_model.encode(answer, convert_to_tensor=True)
    score = float(util.cos_sim(emb_a, emb_b).item())
    score = round(max(0.0, min(1.0, score)), 2)
    return {
        "relevance_score": score,
        "reasoning": "Fallback: scored using semantic similarity because the LLM judge was unavailable.",
    }


# prompt

SYSTEM_PROMPT = """You are a strict evaluation judge. You score how relevant an AI-generated \
answer is to the question that was asked — NOT whether the answer is factually correct, only \
whether it actually addresses what was asked.

Respond with ONLY a JSON object, no other text, in exactly this format:
{"relevance_score": <float between 0 and 1>, "reasoning": "<one concise sentence explaining the score>"}

Scoring guide:
- 1.0 = directly and fully addresses the question
- 0.5 = partially addresses it, or addresses a related but different question
- 0.0 = completely off-topic, does not address the question at all
"""


def _build_user_prompt(question: str, answer: str) -> str:
    return f"Question: {question}\n\nAnswer: {answer}\n\nScore the relevance of the answer to the question."


def _parse_json_response(raw_text: str) -> dict:
    """
    Reasoning models like gpt-oss can prepend explanatory text before the
    actual JSON answer. A naive greedy regex from the first "{" to the
    last "}" can accidentally span across that preamble AND the real JSON,
    producing an unparseable blob even repair_json can't fix. Try the
    LAST brace-delimited candidate first, since the real answer usually
    comes after any reasoning, not before it.
    """
    candidates = re.findall(r"\{.*?\}(?=\s*$|\s*\{)|\{.*\}", raw_text, re.DOTALL)
    if not candidates:
        candidates = [raw_text]

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        if _HAS_JSON_REPAIR:
            try:
                repaired = repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict) and repaired.get("relevance_score") is not None:
                    return repaired
            except Exception:
                pass

    if _HAS_JSON_REPAIR:
        try:
            repaired = repair_json(raw_text, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except Exception:
            pass

    print(f"  Relevance agent: could not parse JSON from response: {raw_text[:300]!r}")
    raise ValueError("Could not parse a valid JSON object from the response")


def judge_relevance(question: str, answer: str) -> dict:
    if not question.strip() or not answer.strip():
        return {
            "relevance_score": 0.0,
            "reasoning": "Question or answer was empty.",
        }

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(question, answer)),
        ]

        response = groq_llm.invoke(messages)
        parsed = _parse_json_response(response.content)

        score = float(parsed["relevance_score"])
        score = round(max(0.0, min(1.0, score)), 2)
        reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."

        return {
            "relevance_score": score,
            "reasoning": reasoning,
        }

    except Exception as e:
        # Any failure (timeout, rate limit, bad JSON, etc.) falls back to local scoring
        result = _fallback_relevance(question, answer)
        result["reasoning"] += f" (LLM judge error: {type(e).__name__})"
        return result

#test
if __name__ == "__main__":
    result = judge_relevance(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
    )
    print(result)