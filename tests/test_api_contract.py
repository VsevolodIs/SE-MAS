from fastapi.testclient import TestClient
from src.api import main as api


class FakeGraph:
    def invoke(self, payload):
        assert payload["user_prompt"] == "Составь 3 задания на дроби"
        return {"task_type": "pedagogical", "final_output": "3 задания", "validation_status": "NEEDS_REVIEW"}
    def stream(self, payload):
        yield {"coordinator": {"task_type": "solve_check"}}
        yield {"solution_checker": {"is_correct": None, "math_verified": True, "math_solution": "не подтверждено"}}


def test_generic_api(monkeypatch):
    monkeypatch.setattr(api, "tutor_app", FakeGraph())
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "tutor_app", FakeGraph())
        response = client.post("/api/v1/agent", json={"user_prompt": "Составь 3 задания на дроби"})
    assert response.status_code == 200
    assert response.json()["task_type"] == "pedagogical"


def test_checker_does_not_infer_truth_from_math_verified(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "tutor_app", FakeGraph())
        response = client.post("/api/v1/check-solution", json={"prompt": "Проверь решение по задаче 2x=10"})
    assert response.status_code == 200
    assert response.json()["is_correct"] is None
