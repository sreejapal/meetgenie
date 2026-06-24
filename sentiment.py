from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

def get_sentiment(text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = sentiment_pipeline(text[:512])[0]

    return {
        "label": label_map[result["label"]],
        "score": float(result["score"])
    }


# ✅ NEW (very small addition): helper for aggregation
from collections import Counter

def get_dominant_sentiment(sentiments):
    """
    sentiments: list of labels like ["POSITIVE", "NEGATIVE", ...]
    """
    if not sentiments:
        return "NEUTRAL"

    count = Counter(sentiments)
    return count.most_common(1)[0][0]