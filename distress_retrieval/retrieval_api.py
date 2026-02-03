from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .retrieve_sqlite import retrieve_similar_cases
from .llm_summary import generate_llm_summary

app = FastAPI(
    title="Distress Retrieval API",
    version="1.0.0"
)

class RetrievalRequest(BaseModel):
    emotion_state: str
    severity: int
    asr_text: Optional[str] = None

@app.post("/retrieve/similar")
def retrieve_similar(request: RetrievalRequest):
    """
    Retrieves similar historical distress cases
    and generates an LLM explanation.
    """

    # 1. Retrieve similar cases
    cases = retrieve_similar_cases(
        emotion=request.emotion_state,
        severity=request.severity
    )

    # 2. Generate explanation
    summary = generate_llm_summary(
        query=request.asr_text or "",
        retrieved_cases=cases,
        evidence={
            "emotion_state": request.emotion_state,
            "severity": request.severity
        }
    )

    return {
        "similar_cases": cases,
        "summary": summary
    }
