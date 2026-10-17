"""Unit tests for Watson emotion detection and Flask routes."""
import unittest
from unittest.mock import Mock, patch
import requests
from EmotionDetection.emotion_detection import emotion_detector
import server


class EmotionDetectorTests(unittest.TestCase):
    """Test model response handling and input validation."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_returns_five_scores_and_dominant_emotion(self, post_mock):
        """Checks all emotion fields and the dominant score."""
        scores = {"anger": 0.02, "disgust": 0.01, "fear": 0.03,
                  "joy": 0.91, "sadness": 0.04}
        response = Mock()
        response.json.return_value = {"emotionPredictions": [
            {"emotion": scores}
        ]}
        post_mock.return_value = response
        result = emotion_detector("I am delighted today.")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(set(result), set(scores) | {"dominant_emotion"})
        self.assertEqual(result["joy"], 0.91)
        post_mock.assert_called_once()
        self.assertEqual(post_mock.call_args.kwargs["json"], {
            "raw_document": {"text": "I am delighted today."}
        })

    def test_rejects_empty_input(self):
        """Rejects blank text before making a network request."""
        with self.assertRaises(ValueError):
            emotion_detector("   ")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_http_error_is_raised(self, post_mock):
        """Propagates HTTP failures from the hosted model."""
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("service error")
        post_mock.return_value = response
        with self.assertRaises(requests.HTTPError):
            emotion_detector("A test sentence.")


class FlaskRouteTests(unittest.TestCase):
    """Test HTTP handling without calling hosted Watson."""

    def setUp(self):
        server.app.config["TESTING"] = True
        self.client = server.app.test_client()

    @patch("server.emotion_detector")
    def test_get_route_formats_result(self, detector_mock):
        """Formats the page response with the dominant emotion."""
        detector_mock.return_value = {
            "anger": 0.01, "disgust": 0.02, "fear": 0.03,
            "joy": 0.9, "sadness": 0.04, "dominant_emotion": "joy",
        }
        response = self.client.get(
            "/emotionDetector?textToAnalyze=happy%20today"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"dominant emotion is joy", response.data)

    def test_empty_input_returns_400(self):
        """Returns a client error for empty text."""
        response = self.client.get("/emotionDetector?textToAnalyze=")
        self.assertEqual(response.status_code, 400)

    @patch("server.emotion_detector")
    def test_json_post_returns_scores(self, detector_mock):
        """Returns structured scores to JSON clients."""
        detector_mock.return_value = {
            "anger": 0.01, "disgust": 0.02, "fear": 0.03,
            "joy": 0.9, "sadness": 0.04, "dominant_emotion": "joy",
        }
        response = self.client.post(
            "/emotionDetector", json={"text": "happy today"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["dominant_emotion"], "joy")

    @patch("server.emotion_detector", side_effect=requests.ConnectionError)
    def test_service_error_returns_502(self, _detector_mock):
        """Hides upstream service details from callers."""
        response = self.client.get("/emotionDetector?textToAnalyze=hello")
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
