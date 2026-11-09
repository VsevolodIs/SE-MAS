import time
import pytest
from src.tools.sandbox import run_sandboxed_math


def test_sandbox_timeout_strict():
    start_time = time.time()
    res = run_sandboxed_math("2*x + 4 = 10", timeout_sec=0.001)
    elapsed = time.time() - start_time

    assert res["status"] == "timeout_error"
    assert "Превышен лимит времени" in res["error"]
    assert elapsed < 1.0