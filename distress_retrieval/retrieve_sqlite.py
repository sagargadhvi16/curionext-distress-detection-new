import sqlite3
import re
from llm_summary import generate_llm_summary

DB_PATH = "distress.db"


# ------------------ TEXT UTILS ------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def semantic_score(query, text):
    stopwords = {
        "i","you","are","is","am","the","a","an","to","and",
        "of","in","on","it","this","that","we","he","she",
        "they","was","were","be"
    }

    q_tokens = set(normalize(query).split()) - stopwords
    t_tokens = set(normalize(text).split()) - stopwords

    return len(q_tokens.intersection(t_tokens))


# ------------------ RETRIEVAL ------------------

def retrieve_distress(context, confidence_threshold=0.6, top_k=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, transcript, primary_class, sub_class, confidence
        FROM distress_records
        WHERE distress_flag = 1
          AND confidence >= ?
    """, (confidence_threshold,))

    rows = cursor.fetchall()
    conn.close()

    scored = []
    for timestamp, transcript, primary_class, sub_class, confidence in rows:
        score = semantic_score(context, transcript)
        scored.append(
            (score, timestamp, transcript, primary_class, sub_class, confidence)
        )

    scored.sort(key=lambda x: (x[0], x[5]), reverse=True)
    return scored[:top_k]


# ------------------ EVIDENCE SUMMARY ------------------

def generate_evidence_summary(results):
    if not results:
        return {
            "emotion_state": "neutral",
            "severity": "low",
            "summary": [
                "No distress signals detected",
                "No relevant historical matches found"
            ]
        }

    high_arousal = sum(1 for r in results if r[3] == "high_arousal")
    distress = sum(1 for r in results if r[3] == "distress")

    if high_arousal > distress:
        emotion_state = "high_arousal"
        severity = "high"
    else:
        emotion_state = "distress"
        severity = "medium"

    return {
        "emotion_state": emotion_state,
        "severity": severity,
        "summary": [
            f"{len(results)} similar distress cases retrieved",
            f"Dominant class: {emotion_state}",
            "High confidence historical patterns observed"
        ]
    }


# ------------------ INTERACTIVE LOOP ------------------

if __name__ == "__main__":
    print("\nDistress Retrieval System")
    print("Type a context query (type 'exit' to quit)\n")

    USE_LLM = True   # toggle here (safe place)

    while True:
        query = input("Enter context query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nExiting system. Bye.")
            break

        results = retrieve_distress(query)

        print("\n" + "=" * 30)
        print("Context Query")
        print("=" * 30)
        print(f"\"{query}\"")

        if not results:
            print("\nNo high-confidence distress cases found.")
            print("\n" + "-" * 50 + "\n")
            continue

        print("\n" + "=" * 30)
        print("Retrieved Distress Cases (Top 5)")
        print("=" * 30)

        for idx, (score, timestamp, transcript, pc, sc, conf) in enumerate(results, start=1):
            print(f"{idx}. Transcript : {transcript}")
            print(f"   Time       : {timestamp}")
            print(f"   Class      : {pc} / {sc}")
            print(f"   Confidence : {conf}")
            print()

        evidence = generate_evidence_summary(results)

        print("=" * 30)
        print("Evidence Summary")
        print("=" * 30)
        print(f"Emotion State : {evidence['emotion_state']}")
        print(f"Severity      : {evidence['severity']}\n")

        print("Key Observations:")
        for line in evidence["summary"]:
            print(f"- {line}")

        # -------- LLM EXPLANATION (POST-RETRIEVAL) --------
        if USE_LLM:
            explanation = generate_llm_summary(query, results, evidence)
            print("\n" + "=" * 30)
            print("LLM Explanation")
            print("=" * 30)
            print(explanation)

        print("\n" + "-" * 50 + "\n")
