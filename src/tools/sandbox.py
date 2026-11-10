import json
import subprocess
import sys
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
    if len(expr_str) > 400:
        raise MathSecurityError("Слишком длинное математическое выражение")
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
    # Ищем выражение с оператором, исключая захват посторонних слов и концевых точек
    pattern = r"((?:[0-9a-zA-Z_\*\+\-\/\^\(\)]|\s(?=[0-9a-zA-Z_\*\+\-\/\^\(\)]))+(?:<=|>=|<|>|=)\s*[-+]?[0-9a-zA-Z_\*\+\-\/\^\(\)\.]+)"
    match = re.search(pattern, raw_str)
    if match:
        eq = match.group(1).strip()
        return re.sub(r"[.;,\s]+$", "", eq)
    return None


def _solve_equation_worker(raw_input: str, variable: str = "x") -> Dict[str, Any]:
    validate_expression_safety(raw_input)
    clean_str = preprocess_math_string(raw_input)
    # Only a small algebraic grammar is accepted by SymPy. Natural-language
    # questions are routed to the LLM and are never passed to sympify().
    if re.search(r"[^0-9a-zA-Z.,+\-*/^()=<>\s%]", clean_str):
        return {"type": "unsupported_format", "roots": [], "solution_str": None}
    identifiers = re.findall(r"[A-Za-z]+", clean_str)
    if any(name not in {"x", "y", "a", "b", "c", "d", "n", "sqrt"} for name in identifiers):
        return {"type": "unsupported_format", "roots": [], "solution_str": None}
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
            if sol in forbidden_roots or sol.is_real is False:
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
            "domain_valid": True,  # only admissible real roots are retained
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
    """Run SymPy in a killable child process, not a ThreadPoolExecutor.

    Important: process termination is NOT a VM/container security boundary.
    This supports bounded time and prevents a hung calculation from keeping a
    Python worker thread alive. Untrusted inputs still require an OS sandbox.
    """
    try:
        validate_expression_safety(eq_str)
    except MathSecurityError as exc:
        return {"status": "security_error", "roots": [], "error": str(exc)}

    request = json.dumps({"expression": eq_str, "variable": variable}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.tools.sandbox", "--worker"],
            input=request,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout_error", "roots": [],
            "error": f"Превышен лимит времени вычисления ({timeout_sec} сек)",
        }
    except OSError as exc:
        return {"status": "math_error", "roots": [], "error": f"Запуск вычисления невозможен: {exc}"}

    if proc.returncode != 0:
        return {"status": "math_error", "roots": [], "error": "Дочерний процесс вычисления завершился с ошибкой"}
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "math_error", "roots": [], "error": "Некорректный JSON от математического процесса"}
    if output.get("error_type") == "MathSecurityError":
        return {"status": "security_error", "roots": [], "error": output.get("error", "Запрещённый ввод")}
    if "error" in output:
        return {"status": "math_error", "roots": [], "error": output["error"]}
    if output.get("type") == "unsupported_format":
        return {"status": "unsupported_format", "roots": [], "details": output,
                "error": "Текстовая задача не поддерживается символьным парсером"}
    return {"status": "success", "roots": output.get("roots", []),
            "details": output, "error": None}


def _worker_main() -> None:
    """Private JSON-in/JSON-out entrypoint for the subprocess."""
    try:
        payload = json.loads(sys.stdin.read())
        res = _solve_equation_worker(payload["expression"], payload.get("variable", "x"))
    except Exception as exc:
        # Return only a sanitized exception category, not a traceback.
        res = {"error": str(exc)[:300], "error_type": type(exc).__name__}
    sys.stdout.write(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker_main()
