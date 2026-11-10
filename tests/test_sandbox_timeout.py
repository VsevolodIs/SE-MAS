import subprocess
import time
from src.tools import sandbox


def test_sandbox_timeout_strict(monkeypatch):
    def simulated_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=kwargs["timeout"])

    monkeypatch.setattr(sandbox.subprocess, "run", simulated_timeout)
    start_time = time.perf_counter()
    result = sandbox.run_sandboxed_math("2*x + 4 = 10", timeout_sec=0.1)
    assert result["status"] == "timeout_error"
    assert "Превышен лимит времени" in result["error"]
    assert time.perf_counter() - start_time < 0.5
