import os
from typing import List, Dict, Any
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def generate_llm_summary(
    query: str,
    retrieved_cases: List[Any],
    evidence: Dict[str, Any],
) -> str:
    """
    Generate cautious explanation from retrieved cases.
    """

    if not retrieved_cases:
        return (
            "No closely matching historical cases were found. "
            "The current signal may represent an uncommon pattern."
        )

    if client is None:
        return (
            "Historical patterns similar to the current signal were identified."
        )

    cases_text = ""
    for i, case in enumerate(retrieved_cases, start=1):
        try:
            _, timestamp, transcript, pc, sc, conf = case
            cases_text += (
                f"{i}. \"{transcript}\" "
                f"({pc}/{sc}, confidence={conf})\n"
            )
        except Exception:
            cases_text += f"{i}. Similar distress-related event.\n"

    prompt = f"""
You are an explanation assistant.

User context:
"{query}"

Retrieved historical ASR-style cases:
{cases_text}

System evidence summary:
- Emotion state: {evidence.get('emotion_state')}
- Severity: {evidence.get('severity')}

Task:
Explain in 3–4 lines what this pattern indicates.
Use cautious language such as "suggests" or "may indicate".
Do NOT claim live detection.
Do NOT invent facts.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=120,
    )

    return response.choices[0].message.content.strip()
