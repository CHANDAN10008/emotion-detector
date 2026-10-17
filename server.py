"""Flask web application for Watson emotion detection."""

from flask import Flask, jsonify, render_template, request
import requests

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.get("/")
def home():
    """Render the starter project's emotion detection page."""
    return render_template("index.html")


def _get_input_text():
    """Read text from the supported query, form, or JSON request."""
    if request.method == "GET":
        return request.args.get("textToAnalyze", "")
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None
        return data.get("textToAnalyze", data.get("text", ""))
    return request.form.get("textToAnalyze", request.form.get("text", ""))


def _format_result(scores):
    """Format scores for display in the starter web page."""
    score_text = ", ".join(
        f"'{emotion}': {scores[emotion]}"
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    )
    return (
        f"For the given statement, the system response is {score_text}. "
        f"The dominant emotion is {scores['dominant_emotion']}."
    )


@app.route("/emotionDetector", methods=["GET", "POST"])
def emotion_detector_route():
    """Analyze submitted text and return the emotion scores."""
    text_to_analyze = _get_input_text()
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return jsonify(error="Invalid text! Please try again!"), 400

    try:
        scores = emotion_detector(text_to_analyze)
    except requests.exceptions.RequestException:
        return jsonify(error="Emotion analysis service is unavailable."), 502
    except ValueError:
        return jsonify(error="Emotion analysis returned an invalid response."), 502
    except Exception:  # pylint: disable=broad-exception-caught  # Keep unexpected details out of the response.
        app.logger.exception("Unexpected emotion detection failure")
        return jsonify(error="Unable to analyze the supplied text."), 500

    if request.method == "POST" and request.is_json:
        return jsonify(scores), 200
    return _format_result(scores), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5200, debug=False)
