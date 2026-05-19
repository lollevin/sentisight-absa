import json
import os
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert marketing analyst specializing in Aspect-Based Sentiment Analysis (ABSA).
Your task is to analyze customer reviews and extract sentiment at the aspect level to generate actionable marketing insights.

For each review, identify specific aspects mentioned (such as: delivery, packaging, quality, price, customer_service, taste, design, usability, reliability, value) and their sentiment.

You MUST return ONLY valid JSON in exactly this format:
{
  "aspects": [
    {
      "name": "<aspect name>",
      "sentiment": "<positive|negative|neutral>",
      "confidence": <float 0.0-1.0>,
      "quote": "<exact phrase from review that supports this sentiment>"
    }
  ],
  "overall_sentiment": "<positive|negative|neutral>",
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "marketing_suggestions": [
    "<specific actionable marketing suggestion 1>",
    "<specific actionable marketing suggestion 2>",
    "<specific actionable marketing suggestion 3>"
  ]
}

Rules:
- Only include aspects actually mentioned in the review
- confidence must be between 0.0 and 1.0
- marketing_suggestions must be specific and actionable (not generic)
- If no clear aspect is identifiable, use "overall" as the aspect name
- Return ONLY the JSON object, no other text
"""


def get_client(provider: str, api_key: str) -> OpenAI:
    if provider == "DeepSeek":
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    elif provider == "OpenAI":
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def analyze_review(review: str, provider: str, api_key: str, model: str) -> dict:
    client = get_client(provider, api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this customer review:\n\n{review}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Normalize fields to avoid KeyError downstream
    result.setdefault("aspects", [])
    result.setdefault("overall_sentiment", "neutral")
    result.setdefault("keywords", [])
    result.setdefault("marketing_suggestions", [])

    return result


def analyze_batch(reviews: list[str], provider: str, api_key: str, model: str, progress_callback=None) -> list[dict]:
    results = []
    for i, review in enumerate(reviews):
        try:
            result = analyze_review(review, provider, api_key, model)
            result["review"] = review
            result["error"] = None
        except Exception as e:
            result = {
                "review": review,
                "aspects": [],
                "overall_sentiment": "error",
                "keywords": [],
                "marketing_suggestions": [],
                "error": str(e),
            }
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(reviews))
    return results


PROVIDER_MODELS = {
    "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
}
