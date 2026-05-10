import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web_app.main import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_analyze_endpoint_uses_service_layer(self):
        payload = {
            "ticker": "TST",
            "company": {"name": "Test Systems"},
            "composite": {"overallScore": 70},
            "modules": {},
        }

        with patch("web_app.main.analyze_ticker", return_value=payload):
            response = self.client.get("/api/analyze/TST")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ticker"], "TST")


if __name__ == "__main__":
    unittest.main()

