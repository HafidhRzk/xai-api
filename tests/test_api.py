from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_predict_and_explain():
    # Use a dummy example close to dataset mean (adjust length)
    # Make sure the app has artifacts loaded before running tests
    sample = {"features": [0.0]*30}
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    r2 = client.post("/explain", json=sample)
    assert r2.status_code == 200
    data = r2.json()
    assert "contributions" in data and "base_value" in data
