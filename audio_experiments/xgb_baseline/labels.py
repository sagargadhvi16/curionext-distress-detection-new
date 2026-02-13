def fine_to_coarse(label: str) -> str:
    l = label.lower()
    if l in ["anger", "abuse", "scream"]:
        return "high_arousal"
    if l in ["fear", "cry"]:
        return "distress"
    return "neutral"
