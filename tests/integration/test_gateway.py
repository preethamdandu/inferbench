from fastapi.testclient import TestClient
from src.gateway.app import app

client = TestClient(app)


def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()
