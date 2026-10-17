# Emotion Detection Web Application

This Coursera final project calls the hosted Watson NLP EmotionPredict model to score anger, disgust, fear, joy, and sadness, then identifies the strongest score.

## Setup

1. Enter the repository folder: cd oaqjp-final-project-emb-ai
2. Create a virtual environment: python3 -m venv .venv
3. Activate it: source .venv/bin/activate
4. Install dependencies: python3 -m pip install -r requirements.txt

## Run and verify

Run unit tests with: python3 -m unittest -v
Run static analysis with: python3 -m pylint server.py EmotionDetection/emotion_detection.py test_emotion_detection.py
Start Flask with: python3 server.py

Open the lab preview for port 5200. The starter page calls GET /emotionDetector?textToAnalyze=... For JSON clients, send POST /emotionDetector with application/json and {"text":"I am happy today."}. Empty input returns HTTP 400.

Watson NLP requires network access to the Skills Network model endpoint. Unit tests mock that service so they run consistently without a live model call.
