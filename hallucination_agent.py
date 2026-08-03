from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import difflib
import string
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# =====================================================
# Groq LLM — one call per evaluation
# =====================================================

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# =====================================================
# Prompt
# =====================================================

SYSTEM_PROMPT = """You are a strict hallucination-detection judge. You check an AI-generated \
answer against retrieved source content to find claims that are NOT supported by that source \
content — i.e. things the answer states that cannot be verified from the given context.

Work claim by claim:
1. Break the answer down into its individual factual claims.
2. For each claim, search the ENTIRE retrieved source content carefully for matching or \
   paraphrased text — including exact or near-exact word-for-word matches. If the same wording, \
   or a clear paraphrase of it, appears anywhere in the source content, the claim IS supported.
3. Only flag a claim as unsupported if you cannot find any matching or paraphrased text for it \
   anywhere in the source content.
4. Before flagging a claim, double-check by re-reading the source content once more for that \
   specific claim's wording. Do not flag a claim that is directly quoted or closely paraphrased \
   from the source.

Respond with ONLY a JSON object, no other text, in exactly this format:
{
  "hallucination_risk": "<Low, Medium, or High>",
  "flagged_statements": ["<exact unsupported claim text from the answer>", "..."],
  "reasoning": "<1-2 sentences summarizing how many claims were checked and how many were unsupported>"
}

Risk guide:
- Low = all claims are supported by the source content (flagged_statements should be empty)
- Medium = some claims are supported, at least one is unsupported or unverifiable
- High = most or all claims are unsupported, or directly contradict the source content

If flagged_statements is empty, hallucination_risk must be "Low".
"""


def _build_user_prompt(answer: str, context_text: str) -> str:
    if not context_text.strip():
        return (
            f"Answer to check: {answer}\n\n"
            f"Retrieved source content: (none available)\n\n"
            f"No source content was retrieved, so no claims in the answer can be verified. "
            f"Flag every distinct factual claim in the answer as unsupported."
        )
    return (
        f"Answer to check: {answer}\n\n"
        f"Retrieved source content: {context_text}\n\n"
        f"Identify any claims in the answer that are not supported by the retrieved source content."
    )


def _parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))



def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _is_actually_grounded(statement: str, context_text: str, threshold: float = 0.82) -> bool:
    norm_stmt = _normalize(statement)
    norm_ctx = _normalize(context_text)

    if not norm_stmt or not norm_ctx:
        return False

    # Exact or near-exact substring match
    if norm_stmt in norm_ctx:
        return True

    # Fuzzy fallback: find the longest matching block between the
    # claim and the context, and see how much of the claim it covers
    matcher = difflib.SequenceMatcher(None, norm_ctx, norm_stmt)
    match = matcher.find_longest_match(0, len(norm_ctx), 0, len(norm_stmt))
    overlap_ratio = match.size / max(1, len(norm_stmt))

    return overlap_ratio >= threshold


def _verify_flagged_statements(flagged: list[str], context_text: str) -> tuple[list[str], int]:
    """Returns (corrected_flagged_list, number_removed_as_false_positive)."""
    if not context_text.strip():
        return flagged, 0  # nothing to verify against, trust the LLM as-is

    verified = []
    removed = 0
    for statement in flagged:
        if _is_actually_grounded(statement, context_text):
            removed += 1  # false positive — actually found in the source, drop it
        else:
            verified.append(statement)

    return verified, removed

#if llm fails

def _fallback_hallucination(context_text: str) -> dict:
    if not context_text.strip():
        return {
            "hallucination_risk": "High",
            "flagged_statements": ["No retrieved context was available to verify any claims."],
            "reasoning": "Fallback: no context available, so risk defaults to High.",
        }
    return {
        "hallucination_risk": "Medium",
        "flagged_statements": [],
        "reasoning": "Fallback: LLM judge was unavailable, defaulted to Medium risk since claims could not be checked.",
    }



# Public function


def detect_hallucination(answer: str, contexts: list[str], reference_answer: Optional[str] = None) -> dict:
    """
    Checks `answer` claim by claim against the retrieved `contexts` AND the
    reference answer (if provided), flagging any specific statements that
    are not supported by EITHER source. Combining both sources matters:
    retrieval can miss relevant chunks for general-knowledge questions that
    aren't well covered in the indexed dataset, and without this, a fully
    correct answer with a valid reference could get falsely flagged just
    because retrieval happened to return irrelevant context.

    LLM-flagged claims are cross-checked against the raw combined source
    text with string matching before being finalized, to catch LLM
    reasoning errors where it flags something that is actually present.

    Returns:
        {
            "hallucination_risk": "Low" | "Medium" | "High",
            "flagged_statements": list[str],
            "reasoning": str
        }
    """

    combined_sources = []
    if reference_answer and reference_answer.strip():
        combined_sources.append(reference_answer.strip())
    if contexts:
        combined_sources.extend(contexts)

    context_text = " ".join(combined_sources).strip()

    if not answer.strip():
        return {
            "hallucination_risk": "High",
            "flagged_statements": [],
            "reasoning": "Answer was empty, nothing to check.",
        }

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(answer, context_text)),
        ]

        response = groq_llm.invoke(messages)
        parsed = _parse_json_response(response.content)

        risk = str(parsed.get("hallucination_risk", "")).strip().title()
        if risk not in ("Low", "Medium", "High"):
            risk = "Medium"

        flagged = parsed.get("flagged_statements", [])
        if not isinstance(flagged, list):
            flagged = []
        flagged = [str(item).strip() for item in flagged if str(item).strip()]

        # Grounding safety net: remove any flagged claim that is actually
        # present in the combined context/reference (LLM false positive)
        flagged, num_removed = _verify_flagged_statements(flagged, context_text)

        reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."

        if num_removed > 0:
            reasoning += (
                f" (Note: {num_removed} claim(s) initially flagged were verified to be "
                f"directly present in the retrieved context or reference answer and removed as false positives.)"
            )

        # Recompute risk based on the corrected flagged list
        if len(flagged) == 0:
            risk = "Low"
        elif risk == "Low" and flagged:
            risk = "Medium"

        return {
            "hallucination_risk": risk,
            "flagged_statements": flagged,
            "reasoning": reasoning,
        }

    except Exception as e:
        result = _fallback_hallucination(context_text)
        result["reasoning"] += f" (LLM judge error: {type(e).__name__})"
        return result


if __name__ == "__main__":
    # Quick standalone test — reference_answer confirms "Paris" is correct
    # even though contexts (simulating a retrieval miss) has nothing relevant
    result = detect_hallucination(
        answer="Paris is the capital of France.",
        contexts=["This chunk is about something completely unrelated to France."],
        reference_answer="Paris",
    )
    print(result)