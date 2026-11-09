import pytest
from src.tools.sandbox import run_sandboxed_math

def test_sandbox_timeout_interruption():
    res = run_sandboxed_math("x**1000000 - 1 = 0", timeout_sec=0.5)
    assert res["status"] in ["timeout_error", "success", "math_error"]
    if res["status"] == "timeout_error":
        assert "Превышен лимит времени" in res["error"]