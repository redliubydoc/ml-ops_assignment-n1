from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

