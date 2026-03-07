"""
Activity 1 - Hello, Azure AI
AI-102: First API calls to Azure AI services
"""

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

_ACTIVITY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("openai")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------
_openai_client = None
_content_safety_client = None
_language_client = None


# -------------------- OpenAI --------------------

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AzureOpenAI

        _openai_client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-02-15-preview",
        )
    return _openai_client


# -------------------- Content Safety --------------------

def _get_content_safety_client():
    global _content_safety_client
    if _content_safety_client is None:
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.core.credentials import AzureKeyCredential

        _content_safety_client = ContentSafetyClient(
            endpoint=os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"],
            credential=AzureKeyCredential(
                os.environ["AZURE_CONTENT_SAFETY_KEY"]
            ),
        )
    return _content_safety_client


# -------------------- AI Language --------------------

def _get_language_client():
    global _language_client
    if _language_client is None:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        _language_client = TextAnalyticsClient(
            endpoint=os.environ["AZURE_AI_LANGUAGE_ENDPOINT"],
            credential=AzureKeyCredential(
                os.environ["AZURE_AI_LANGUAGE_KEY"]
            ),
        )
    return _language_client


# ---------------------------------------------------------------------------
# Step 1 - Classify 311 Request
# ---------------------------------------------------------------------------

def classify_311_request(request_text: str) -> dict:
    print("DEPLOYMENT:", os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    print("KEY:", os.environ.get("AZURE_OPENAI_KEY"))
    
    
    client = _get_openai_client()

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant for classifying Memphis 311 service requests. "
                    "Classify the complaint into ONE of the following categories exactly as written:\n"
                    "- Pothole\n"
                    "- Noise Complaint\n"
                    "- Trash/Litter\n"
                    "- Street Light\n"
                    "- Water/Sewer\n"
                    "- Other\n\n"
                    "Return JSON with exactly these keys:\n"
                    "category (string),\n"
                    "confidence (number between 0.0 and 1.0),\n"
                    "reasoning (short explanation string)"
                ),
            },
            {"role": "user", "content": request_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    result = json.loads(content)

    return {
        "category": result["category"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }


# ---------------------------------------------------------------------------
# Step 2 - Content Safety
# ---------------------------------------------------------------------------

def check_content_safety(text: str) -> dict:
    from azure.ai.contentsafety.models import AnalyzeTextOptions

    client = _get_content_safety_client()

    result = client.analyze_text(
        AnalyzeTextOptions(text=text)
    )

    categories = {}
    for category in result.categories_analysis:
        categories[category.category] = category.severity

    safe = all(severity == 0 for severity in categories.values())

    return {
        "safe": safe,
        "categories": categories
    }


# ---------------------------------------------------------------------------
# Step 3 - Key Phrase Extraction
# ---------------------------------------------------------------------------

def extract_key_phrases(text: str) -> list[str]:
    client = _get_language_client()

    response = client.extract_key_phrases([text])[0]

    if response.is_error:
        return []

    return response.key_phrases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = os.path.join(_ACTIVITY_DIR, "data", "sample_requests.json")

    if len(sys.argv) > 1 and os.path.exists(data_path):
        with open(data_path) as f:
            samples = json.load(f)
        idx = int(sys.argv[1]) % len(samples)
        sample_request = samples[idx]["text"]
    else:
        sample_request = (
            "There's a huge pothole on Poplar Avenue near the "
            "Walgreens that damaged my tire"
        )

    classification = None
    safety = None
    key_phrases = None

    try:
        classification = classify_311_request(sample_request)
    except Exception as e:
        print(f"Classification error: {e}")

    try:
        safety = check_content_safety(sample_request)
    except Exception as e:
        print(f"Content Safety error: {e}")

    try:
        key_phrases = extract_key_phrases(sample_request)
    except Exception as e:
        print(f"Language error: {e}")

    status = (
        "partial"
        if classification and safety and key_phrases
        else "partial"
        if classification or safety or key_phrases
        else "error"
    )

    result = {
        "task": "partial",
        "status": status,
        "outputs": {
            "classification": classification,
            "content_safety": safety,
            "key_phrases": key_phrases,
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            "sdk_version": _get_sdk_version(),
        },
    }

    result_path = os.path.join(_ACTIVITY_DIR, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to {result_path}")


if __name__ == "__main__":
    main()