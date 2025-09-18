import pytest
import requests
import os
import sys

# Add src to path for imports (if needed later)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFraudDetectionAPI:
    """Integration tests for the Fraud Detection Flask API."""

    BASE_URL = "http://127.0.0.1:5000"

    def _server_up(self) -> bool:
        try:
            r = requests.get(f"{self.BASE_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def test_predict_endpoint_valid_data(self):
        if not self._server_up():
            pytest.skip("API server not running; skipping integration test")
        sample_data = {
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530534538215,
            "Amount": 149.62,
            "Time": 0.0,
        }
        response = requests.post(
            f"{self.BASE_URL}/predict", json=sample_data, timeout=5
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_predict_endpoint_missing_data(self):
        if not self._server_up():
            pytest.skip("API server not running; skipping integration test")
        incomplete_data = {
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            # Missing the rest of the required fields
        }
        response = requests.post(
            f"{self.BASE_URL}/predict", json=incomplete_data, timeout=5
        )
        assert response.status_code in [400, 500]

    def test_batch_predict_endpoint(self):
        if not self._server_up():
            pytest.skip("API server not running; skipping integration test")
        response = requests.get(f"{self.BASE_URL}/batch_predict", timeout=5)
        assert response.status_code == 405  # Method not allowed (expects POST)


if __name__ == "__main__":
    pytest.main([__file__])
