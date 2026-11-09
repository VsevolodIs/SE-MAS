from typing import Any, Dict
from src.tools.sandbox import run_sandboxed_math


def execute_symbolic_math(expression: str, variable: str = "x") -> Dict[str, Any]:
    return run_sandboxed_math(expression, variable=variable, timeout_sec=3.0)


SKILL_METADATA = {
    "math_solver": {
        "name": "math_solver",
        "description": "Символьное аналитическое вычисление и решение уравнений",
        "parameters": {"expression": "string", "variable": "string"},
    },
    "solution_verifier": {
        "name": "solution_verifier",
        "description": "Строгая проверка корней и ограничений ОДЗ",
        "parameters": {"equation": "string", "proposed_roots": "list"},
    },
}