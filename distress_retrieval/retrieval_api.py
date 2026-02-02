from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from .retrieve_sqlite import retrieve_similar_cases
from .llm_summary import generate_llm_summary


app = FastAPI(
    title="Distress Retrieval API",
    description="Retrieve similar distress cases and generate explanations",
    version="1.0.0",
)


class RetrievalRequest(BaseModel):
    emotion_state: str = Field(..., example="high_arousal")
    severity: int = Field(..., ge=0, le=10, example=7)
    asr_text: Optional[str] = Field(None, example="child crying loudly")


@app.post("/retrieve/similar")
def retrieve_similar(request: RetrievalRequest):

    cases = retrieve_similar_cases(
        emotion=request.emotion_state,
        severity=request.severity,
    )

    summary = generate_llm_summary(
        query=request.asr_text or "",
        retrieved_cases=cases,
        evidence={
            "emotion_state": request.emotion_state,
            "severity": request.severity,
        },
    )

    return {
        "similar_cases": cases,
        "summary": summary,
    }
