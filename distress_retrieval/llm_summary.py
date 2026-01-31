from groq import Groq

client = Groq(api_key="YOUR API KEY")


def generate_llm_summary(query, retrieved_cases, evidence):
   

    cases_text = ""
    for i, case in enumerate(retrieved_cases, start=1):
        _, timestamp, transcript, pc, sc, conf = case
        cases_text += (
            f"{i}. \"{transcript}\" "
            f"({pc}/{sc}, confidence={conf})\n"
        )

    prompt = f"""
You are an explanation assistant.

User context:
"{query}"

Retrieved historical ASR-style cases:
{cases_text}

System evidence summary:
- Emotion state: {evidence['emotion_state']}
- Severity: {evidence['severity']}

Task:
Explain in 3–4 lines what this pattern indicates.
Do NOT claim live detection.
Do NOT invent new facts.
Use cautious language like "suggests" or "indicates".
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=120,
    )


    return response.choices[0].message.content.strip()
