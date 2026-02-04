import sqlite3
import re
import os

# ALWAYS resolve DB relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "distress.db")
# ------------------ TEXT UTILS ------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def semantic_score(query: str, text: str) -> int:
    stopwords = {
        "i","you","are","is","am","the","a","an","to","and",
        "of","in","on","it","this","that","we","he","she",
        "they","was","were","be"
    }

    q_tokens = set(normalize(query).split()) - stopwords
    t_tokens = set(normalize(text).split()) - stopwords

    return len(q_tokens.intersection(t_tokens))


# ------------------ CORE RETRIEVAL ------------------

def retrieve_distress(context: str, confidence_threshold: float = 0.6, top_k: int = 5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, transcript, primary_class, sub_class, confidence
        FROM distress_records
        WHERE distress_flag = 1
          AND confidence >= ?
        """,
        (confidence_threshold,),
    )

    rows = cursor.fetchall()
    conn.close()

    scored = []
    for timestamp, transcript, pc, sc, conf in rows:
        score = semantic_score(context, transcript)
        scored.append((score, timestamp, transcript, pc, sc, conf))

    scored.sort(key=lambda x: (x[0], x[5]), reverse=True)
    return scored[:top_k]


# ------------------ API-FACING WRAPPER ------------------

def retrieve_similar_cases(emotion: str, severity: int, top_k: int = 5):
    """
    Wrapper used by FastAPI.
    """
    context = emotion.replace("_", " ")
    return retrieve_distress(context=context, top_k=top_k)
