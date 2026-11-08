import concurrent.futures
import re
from typing import Any, Dict, List, Optional
import sympy as sp


class MathExecutionTimeoutError(Exception):
    pass


class MathSecurityError(Exception):
    pass


FORBIDDEN_KEYWORDS = [
    "__",
    "import",
    "eval",
    "exec",
    "open",
    "os",
    "sys",
    "subprocess",
    "lambda",
    "globals",
    "locals",
    "compile",
]


def validate_expression_safety(expr_str: str) -> None:
    expr_lower = expr_str.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in expr_lower:
            raise MathSecurityError(f"Обнаружена запрещенная инструкция: '{keyword}'")


def extract_pure_equation(raw_str: str) -> str:
    match = re.search(r"([0-9a-zA-Z_\*\+\-\/\^\(\)\.\s]+=\s*[-+]?[0-9a-zA-Z_\*\+\-\/\^\(\)\.]+)", raw_str)
    if match:
        eq = match.group(1).strip()
        return eq.rstrip(".;, ")
    return raw_str.strip()


def _solve_equation_worker(eq_str: str, variable: str = "x") -> List[Dict[str, Any]]:
    validate_expression_safety(eq_str)
    clean_eq = extract_pure_equation(eq_str)

    var = sp.Symbol(variable)

    if "=" in clean_eq:
        left_str, right_str = clean_eq.split("=", 1)
        left_expr = sp.sympify(left_str.strip())
        right_expr = sp.sympify(right_str.strip())
        expr = left_expr - right_expr
    else:
        expr = sp.sympify(clean_eq.strip())

    solutions = sp.solve(expr, var)
    parsed_roots = []
    for sol in solutions:
        parsed_roots.append({
            "symbolic": str(sol),
            "numeric": float(sol.evalf()) if sol.is_real else None,
        })
    return parsed_roots


def run_sandboxed_math(
        eq_str: str, variable: str = "x", timeout_sec: float = 3.0
) -> Dict[str, Any]:
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_solve_equation_worker, eq_str, variable)
            roots = future.result(timeout=timeout_sec)
            return {
                "status": "success",
                "roots": roots,
                "error": None,
            }
    except concurrent.futures.TimeoutError:
        return {
            "status": "timeout_error",
            "roots": [],
            "error": f"Превышен лимит времени вычисления ({timeout_sec} сек)",
        }
    except MathSecurityError as e:
        return {
            "status": "security_error",
            "roots": [],
            "error": str(e),
        }
    except Exception as e:
        return {
            "status": "math_error",
            "roots": [],
            "error": f"Ошибка парсинга/вычисления: {str(e)}",
        }