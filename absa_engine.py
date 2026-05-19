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
    elif provider == "Groq":
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    elif provider == "Gemini":
        return OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Groq and Gemini don't support response_format=json_object for all models,
# so we handle JSON extraction differently
_JSON_FORMAT_PROVIDERS = {"DeepSeek", "OpenAI"}


def analyze_review(review: str, provider: str, api_key: str, model: str) -> dict:
    client = get_client(provider, api_key)

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this customer review:\n\n{review}"},
        ],
        temperature=0.3,
    )
    if provider in _JSON_FORMAT_PROVIDERS:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content or ""
    # Strip markdown code fences if present (e.g. ```json ... ```)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
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
    "OpenAI":   ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Groq":     ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    "Gemini":   ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
}

PROVIDER_INFO = {
    "DeepSeek": {"label": "DeepSeek 🇨🇳", "note": "Recommended · cheapest · ¥10 lasts long", "key_url": "https://platform.deepseek.com"},
    "OpenAI":   {"label": "OpenAI",        "note": "GPT-4o · most popular",                   "key_url": "https://platform.openai.com"},
    "Groq":     {"label": "Groq ⚡",        "note": "Free tier · fastest inference",            "key_url": "https://console.groq.com"},
    "Gemini":   {"label": "Gemini 🔵",     "note": "Free tier · Google AI",                    "key_url": "https://aistudio.google.com/apikey"},
}
