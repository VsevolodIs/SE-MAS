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


def preprocess_math_string(raw_str: str) -> str:
    expr = raw_str.strip()

    # Замена степени ^ на **
    expr = expr.replace("^", "**")

    # Замена десятичной запятой: 5,5 -> 5.5
    expr = re.sub(r"(\d+),(\d+)", r"\1.\2", expr)

    # 2x -> 2*x
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)

    # 3(x+1) -> 3*(x+1)
    expr = re.sub(r"(\d)(\()", r"\1*\2", expr)

    # (x+1)(x-1) -> (x+1)*(x-1)
    expr = re.sub(r"(\))(\()", r"\1*\2", expr)

    # (x+1)x -> (x+1)*x
    expr = re.sub(r"(\))([a-zA-Z])", r"\1*\2", expr)

    # Поддержка процентов: 20% от 250 = 250 * 0.20
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:от|of)\s*(\d+(?:\.\d+)?)", expr, flags=re.IGNORECASE)
    if pct_match:
        pct = float(pct_match.group(1)) / 100.0
        val = float(pct_match.group(2))
        return f"{val} * {pct}"

    return expr


def extract_math_relation(raw_str: str) -> Optional[str]:
    # Поиск знаков отношения: =, <=, >=, <, >
    pattern = r"([0-9a-zA-Z_\*\+\-\/\^\(\)\.\s]+(?:\<=|\>=|\<|\>|=)\s*[-+]?[0-9a-zA-Z_\*\+\-\/\^\(\)\.]+)"
    match = re.search(pattern, raw_str)
    if match:
        eq = match.group(1).strip()
        return eq.rstrip(".;, ")
    return None


def _solve_equation_worker(raw_input: str, variable: str = "x") -> Dict[str, Any]:
    validate_expression_safety(raw_input)
    clean_str = preprocess_math_string(raw_input)
    relation = extract_math_relation(clean_str)

    var = sp.Symbol(variable)

    target = relation if relation else clean_str

    for op in ["<=", ">=", "<", ">"]:
        if op in target:
            left_s, right_s = target.split(op, 1)
            left_e = sp.sympify(left_s.strip())
            right_e = sp.sympify(right_s.strip())

            if op == "<":
                ineq = left_e < right_e
            elif op == ">":
                ineq = left_e > right_e
            elif op == "<=":
                ineq = left_e <= right_e
            else:
                ineq = left_e >= right_e

            sol = sp.solve(ineq, var)
            return {
                "type": "inequality",
                "solution_str": str(sol),
                "roots": [],
                "domain_valid": True,
            }

    if "=" in target:
        left_s, right_s = target.split("=", 1)
        left_e = sp.sympify(left_s.strip())
        right_e = sp.sympify(right_s.strip())
        expr = left_e - right_e

        denoms = sp.denom(sp.together(expr))
        forbidden_roots = sp.solve(denoms, var) if denoms != 1 else []

        raw_solutions = sp.solve(expr, var)
        parsed_roots = []
        for sol in raw_solutions:
            if sol in forbidden_roots:
                continue
            parsed_roots.append({
                "symbolic": str(sol),
                "numeric": float(sol.evalf()) if sol.is_real else None,
            })

        return {
            "type": "equation",
            "roots": parsed_roots,
            "forbidden_roots": [str(r) for r in forbidden_roots],
            "solution_str": ", ".join(str(r["symbolic"]) for r in parsed_roots),
            "domain_valid": len(forbidden_roots) == 0 or any(s in forbidden_roots for s in raw_solutions),
        }

    try:
        expr = sp.sympify(target)
        val = expr.evalf()
        return {
            "type": "expression",
            "roots": [{"symbolic": str(val), "numeric": float(val) if val.is_real else None}],
            "solution_str": str(val),
            "domain_valid": True,
        }
    except Exception as e:
        if re.search(r"[а-яА-ЯёЁ]", raw_input):
            return {
                "type": "unsupported_format",
                "roots": [],
                "solution_str": None,
                "domain_valid": True,
            }
        raise e


def run_sandboxed_math(
        eq_str: str, variable: str = "x", timeout_sec: float = 3.0
) -> Dict[str, Any]:
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_solve_equation_worker, eq_str, variable)
            res = future.result(timeout=timeout_sec)

            if res.get("type") == "unsupported_format":
                return {
                    "status": "unsupported_format",
                    "roots": [],
                    "details": res,
                    "error": "Инструмент не поддерживает текстовый формат задачи; требуется обработка LLM-валидатором",
                }

            return {
                "status": "success",
                "roots": res.get("roots", []),
                "details": res,
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