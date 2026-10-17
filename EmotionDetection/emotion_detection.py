"""Detect emotion scores with the Watson NLP hosted model."""

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return five Watson emotion scores and the strongest emotion."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        raise ValueError("Text to analyze must not be empty.")

    response = requests.post(
        WATSON_URL,
        json={"raw_document": {"text": text_to_analyze}},
        headers={"grpc-metadata-mm-model-id": MODEL_ID},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        model_scores = payload["emotionPredictions"][0]["emotion"]
        scores = {emotion: model_scores[emotion] for emotion in EMOTIONS}
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Watson returned an invalid emotion response.") from error

    scores["dominant_emotion"] = max(EMOTIONS, key=scores.get)
    return scores
