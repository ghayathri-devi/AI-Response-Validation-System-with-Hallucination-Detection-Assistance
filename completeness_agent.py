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


def _fallback_completeness(question: str, answer: str) -> dict:
    st_model = get_embedding_model()
    emb_a = st_model.encode(question, convert_to_tensor=True)
    emb_b = st_model.encode(answer, convert_to_tensor=True)
    score = float(util.cos_sim(emb_a, emb_b).item())
    score = round(max(0.0, min(1.0, score)), 2)
    return {
        "completeness_score": score,
        "reasoning": "Fallback: scored using semantic similarity because the LLM judge was unavailable.",
        "missing_aspects": [],
    }


SYSTEM_PROMPT = """You are a strict evaluation judge. You check whether an AI-generated answer \
COVERS everything the question actually asked for — not whether it's factually correct (a \
separate accuracy check handles that), only whether it's thorough.

Work in two steps:
1. Break the question down into its distinct required aspects. A question can implicitly or \
   explicitly ask for multiple things — e.g. "what is X and why does it matter" has two aspects \
   (a definition, and an explanation of significance); "compare A and B" has two aspects (A's \
   properties, B's properties); a single "what is X" question usually has just one aspect.
2. For each required aspect, check whether the answer addresses it at all — even a brief mention \
   counts as addressed. Only count an aspect as missing if the answer does not touch on it at all.

Respond with ONLY a JSON object, no other text, in exactly this format:
{
  "completeness_score": <float between 0 and 1>,
  "reasoning": "<1-2 sentences listing the required aspects and which were covered or missing>",
  "missing_aspects": ["<short description of a required aspect the answer did not address>", "..."]
}

Scoring guide:
- 1.0 = every required aspect of the question is addressed
- 0.5-0.7 = most aspects addressed, at least one aspect missing
- 0.2-0.4 = only a small part of what was asked is addressed
- 0.0 = the answer does not address any part of what was asked (this overlaps with low relevance,
  but score it here based on coverage, not topic match)

If missing_aspects is empty, completeness_score should be 0.85 or higher.

IMPORTANT: Your entire response must be valid JSON. Never use double-quote characters (") inside \
string values — use single quotes (') instead if you need to quote text. Do not include line \
breaks inside string values.
"""


def _build_user_prompt(question: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer to check: {answer}\n\n"
        f"Identify the required aspects of the question and check whether the answer covers each one."
    )


def _parse_json_response(raw_text: str) -> dict:
    """
    Reasoning models like gpt-oss can prepend explanatory text before the
    actual JSON answer. A naive greedy regex from the first "{" to the
    last "}" can accidentally span across that preamble AND the real JSON,
    producing an unparseable blob even repair_json can't fix. To handle
    this: find ALL brace-delimited candidates and try the LAST one first
    (the actual answer usually comes after any reasoning), falling back
    through earlier candidates and finally the raw repair_json pass.
    """
    candidates = re.findall(r"\{.*?\}(?=\s*$|\s*\{)|\{.*\}", raw_text, re.DOTALL)
    if not candidates:
        candidates = [raw_text]

    # Try candidates last-first — reasoning models usually put the real
    # structured answer after any thinking/preamble text, not before it
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        if _HAS_JSON_REPAIR:
            try:
                repaired = repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict) and repaired.get("completeness_score") is not None:
                    return repaired
            except Exception:
                pass

    # Last resort: let repair_json try the entire raw response as-is,
    # since its own extraction logic is sometimes smarter than ours
    if _HAS_JSON_REPAIR:
        try:
            repaired = repair_json(raw_text, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except Exception:
            pass

    # Nothing worked — log what we actually got, so the next failure is
    # diagnosable instead of a mystery
    print(f"  Completeness agent: could not parse JSON from response: {raw_text[:300]!r}")
    raise ValueError("Could not parse a valid JSON object from the response")


def _call_llm_once(question: str, answer: str) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(question, answer)),
    ]
    response = groq_llm.invoke(messages)
    parsed = _parse_json_response(response.content)

    score = float(parsed["completeness_score"])
    score = round(max(0.0, min(1.0, score)), 2)
    reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."

    missing = parsed.get("missing_aspects", [])
    if not isinstance(missing, list):
        missing = []
    missing = [str(item).strip() for item in missing if str(item).strip()]

    if not missing and score < 0.85:
        score = 0.85
    if missing and score >= 0.85:
        score = 0.7

    return {
        "completeness_score": score,
        "reasoning": reasoning,
        "missing_aspects": missing,
    }


def judge_completeness(question: str, answer: str) -> dict:
    """
    Checks whether `answer` covers every distinct aspect of `question`,
    with reasoning and a list of any missing aspects, using a single LLM
    call (retried once if the first response isn't valid JSON before
    falling back to local scoring).

    Returns:
        {
            "completeness_score": float (0.0-1.0),
            "reasoning": str,
            "missing_aspects": list[str]
        }
    """

    if not question.strip() or not answer.strip():
        return {
            "completeness_score": 0.0,
            "reasoning": "Question or answer was empty.",
            "missing_aspects": [],
        }

    last_error = None
    for attempt in range(2):
        try:
            return _call_llm_once(question, answer)
        except Exception as e:
            last_error = e
            continue

    result = _fallback_completeness(question, answer)
    result["reasoning"] += f" (LLM judge error after retry: {type(last_error).__name__})"
    return result


if __name__ == "__main__":
    result = judge_completeness(
        question="What is photosynthesis and why is it important for life on Earth?",
        answer="Photosynthesis is the process by which plants convert sunlight into energy.",
    )
    print(result)