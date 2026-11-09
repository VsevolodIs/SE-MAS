import time
import pytest
from src.tools import sandbox
from src.tools.sandbox import run_sandboxed_math


def test_sandbox_timeout_strict(monkeypatch):
    def slow_worker(eq_str, variable="x"):
        time.sleep(1.0)
        return {"type": "expression", "roots": []}

    monkeypatch.setattr(sandbox, "_solve_equation_worker", slow_worker)

    start_time = time.time()
    res = run_sandboxed_math("2*x + 4 = 10", timeout_sec=0.1)
    elapsed = time.time() - start_time

    assert res["status"] == "timeout_error"
    assert "Превышен лимит времени" in res["error"]
    assert elapsed < 0.5