from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from sentence_transformers import SentenceTransformer, util

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


# =====================================================
# Groq LLM — one call per evaluation (retried once on
# JSON parse failure before falling back)
# =====================================================

groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# =====================================================
# Local fallback (used only if the LLM call fails twice
# or returns something unparsable, so the API never breaks)
# =====================================================

st_model = SentenceTransformer("all-MiniLM-L6-v2")


def _fallback_accuracy(answer: str, ground_truth: str) -> dict:
    emb_a = st_model.encode(answer, convert_to_tensor=True)
    emb_b = st_model.encode(ground_truth, convert_to_tensor=True)
    score = float(util.cos_sim(emb_a, emb_b).item())
    score = round(max(0.0, min(1.0, score)), 2)
    return {
        "accuracy_score": score,
        "evidence": "Fallback: scored using semantic similarity because the LLM judge was unavailable.",
        "supporting_excerpt": "",
    }


# =====================================================
# Prompt
# =====================================================

SYSTEM_PROMPT = """You are a strict evaluation judge. You check whether an AI-generated answer \
is FACTUALLY CORRECT when compared against a ground truth source — either a reference answer or \
retrieved source content. You are not checking relevance or completeness, only factual accuracy.

Work claim by claim:
1. Break the answer down into its individual factual claims.
2. For each claim, search the ground truth carefully — including paraphrases, synonyms, and \
   related wording, not just exact word matches — for text that supports or contradicts it.
3. Quote the supporting or contradicting text directly from the ground truth where you find it \
   (short exact excerpt, not a paraphrase).
4. Classify each claim into ONE of these three categories, since they must be scored differently:
   - CONTRADICTED: the ground truth states something that directly conflicts with the claim.
   - UNSTATED: the ground truth does not mention this claim at all, but the claim is a reasonable, \
     widely-known, non-contradictory extension of the topic (e.g. common textbook knowledge about \
     the same subject). This is NOT the same as being wrong.
   - SUPPORTED: the ground truth directly states or clearly paraphrases the claim.

Respond with ONLY a JSON object, no other text, in exactly this format:
{
  "accuracy_score": <float between 0 and 1>,
  "evidence": "<2-3 sentences walking through the claims, their category, and what was found>",
  "supporting_excerpt": "<the single most relevant short exact quote from the ground truth, or empty string if none found>"
}

Scoring guide — score based on the MIX of claim categories above, not an all-or-nothing judgment:
- 1.0 = all claims are SUPPORTED
- 0.7-0.9 = most claims SUPPORTED, at most one UNSTATED-but-reasonable claim, nothing CONTRADICTED
- 0.4-0.6 = a mix of SUPPORTED and UNSTATED claims, nothing CONTRADICTED
- 0.2-0.3 = mostly UNSTATED claims with no supporting evidence, but still nothing CONTRADICTED
- 0.0-0.1 = reserved ONLY for claims that are CONTRADICTED by the ground truth, or answers with \
  no plausible connection to the topic at all

IMPORTANT: Do NOT score 0.0 just because a claim is not explicitly stated in the ground truth. \
An UNSTATED claim that is reasonable and not contradicted should still receive partial credit \
(0.2 or higher), never zero. Reserve 0.0 strictly for direct contradictions.

IMPORTANT: Keep "evidence" and "supporting_excerpt" SHORT — evidence under 40 words, \
supporting_excerpt under 20 words. Do not list every claim individually; summarize.

IMPORTANT: Your entire response must be valid JSON. Never use double-quote characters (") inside \
the "evidence" or "supporting_excerpt" string values — if you need to quote text, use single quotes \
(') instead. Do not include line breaks inside string values.
"""


def _build_user_prompt(answer: str, ground_truth: str, source_label: str) -> str:
    return (
        f"Answer to check: {answer}\n\n"
        f"{source_label}: {ground_truth}\n\n"
        f"Score the factual accuracy of the answer against the {source_label.lower()}."
    )


def _parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    candidate = match.group(0) if match else raw_text

    # First try strict parsing
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to a tolerant repair pass — handles stray quotes, trailing
    # commas, and other minor malformations small LLMs sometimes produce
    if _HAS_JSON_REPAIR:
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired

    raise ValueError("Could not parse a valid JSON object from the response")


def _call_llm_once(answer: str, ground_truth: str, source_label: str) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(answer, ground_truth, source_label)),
    ]
    response = groq_llm.invoke(messages)
    parsed = _parse_json_response(response.content)

    score = float(parsed["accuracy_score"])
    score = round(max(0.0, min(1.0, score)), 2)
    evidence = str(parsed.get("evidence", "")).strip() or "No evidence provided."
    supporting_excerpt = str(parsed.get("supporting_excerpt", "")).strip()

    return {
        "accuracy_score": score,
        "evidence": evidence,
        "supporting_excerpt": supporting_excerpt,
    }


# =====================================================
# Public function
# =====================================================

def judge_accuracy(
    answer: str,
    reference_answer: Optional[str],
    contexts: list[str],
) -> dict:
    """
    Scores the factual accuracy of `answer` against either the reference
    answer (if provided) or the retrieved context chunks (fallback source),
    with supporting evidence, using a single LLM call. Parsing tries strict
    JSON first, then a tolerant repair pass, then retries the LLM call once
    before falling back to local semantic similarity scoring.

    Returns:
        {
            "accuracy_score": float (0.0-1.0),
            "evidence": str,
            "supporting_excerpt": str,
            "used_reference_answer": bool
        }
    """

    has_reference = bool(reference_answer and reference_answer.strip())
    context_text = " ".join(contexts).strip() if contexts else ""

    if has_reference:
        ground_truth = reference_answer
        source_label = "Reference answer"
    elif context_text:
        ground_truth = context_text
        source_label = "Retrieved source content"
    else:
        return {
            "accuracy_score": 0.0,
            "evidence": "No reference answer or retrieved context was available to check accuracy against.",
            "supporting_excerpt": "",
            "used_reference_answer": False,
        }

    if not answer.strip():
        return {
            "accuracy_score": 0.0,
            "evidence": "Answer was empty.",
            "supporting_excerpt": "",
            "used_reference_answer": has_reference,
        }

    last_error = None
    for attempt in range(2):  # try once, retry once on parse failure
        try:
            result = _call_llm_once(answer, ground_truth, source_label)
            result["used_reference_answer"] = has_reference
            return result
        except Exception as e:
            last_error = e
            continue

    # Both attempts failed — fall back to local scoring
    result = _fallback_accuracy(answer, ground_truth)
    result["evidence"] += f" (LLM judge error after retry: {type(last_error).__name__})"
    result["used_reference_answer"] = has_reference
    return result


if __name__ == "__main__":
    # Quick standalone test
    result = judge_accuracy(
        answer="Artificial Intelligence (AI) is the ability of a computer or machine to perform tasks that normally require human intelligence. These tasks include learning, reasoning, understanding language, recognising images and speech, making decisions, and solving problems.",
        reference_answer=None,
        contexts=[
            "Artificial intelligence (AI) is the intelligence of machines or software, as opposed "
            "to the intelligence of humans or animals. It is also the field of study in computer "
            "science that develops and studies intelligent machines."
        ],
    )
    print(result)