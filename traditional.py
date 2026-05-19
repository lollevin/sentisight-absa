from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_vader(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": round(compound, 4),
        "positive_score": round(scores["pos"], 4),
        "negative_score": round(scores["neg"], 4),
        "neutral_score": round(scores["neu"], 4),
        "label": label,
        "limitation": "VADER can only provide an overall sentiment score. It cannot identify which specific aspects (delivery, packaging, quality, etc.) are positive or negative — making it insufficient for detailed marketing decisions.",
    }


def analyze_batch_vader(texts: list[str]) -> list[dict]:
    return [analyze_vader(t) for t in texts]
