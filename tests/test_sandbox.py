import pytest
from src.tools.sandbox import run_sandboxed_math


def test_sandbox_linear_equation():
  result = run_sandboxed_math("2*x + 4 = 10", variable="x", timeout_sec=2.0)
  assert result["status"] == "success"
  assert len(result["roots"]) == 1
  assert result["roots"][0]["numeric"] == 3.0


def test_sandbox_security_rejection():
  malicious_input = "__import__('os').system('dir')"
  result = run_sandboxed_math(malicious_input, timeout_sec=2.0)
  assert result["status"] == "security_error"
  assert "запрещенная инструкция" in result["error"]


def test_sandbox_syntax_error_handling():
  bad_input = "2 * x +++"
  result = run_sandboxed_math(bad_input, timeout_sec=2.0)
  assert result["status"] == "math_error"