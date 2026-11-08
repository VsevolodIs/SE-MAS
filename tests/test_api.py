import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "TutorFlow" in data.get("service", "")


def test_check_solution_invalid_payload(client):
    response = client.post("/api/v1/check-solution", json={"prompt": "hi"})
    assert response.status_code == 422


def test_check_solution_missing_field(client):
    response = client.post("/api/v1/check-solution", json={})
    assert response.status_code == 422


@pytest.mark.slow
def test_check_solution_pipeline_success(client):
    payload = {
        "prompt": "Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3."
    }
    response = client.post("/api/v1/check-solution", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_correct") is True
    assert "3.0" in data.get("math_solution", "")
    assert data.get("latency_sec") > 0