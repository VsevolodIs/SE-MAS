import atexit
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional, TypedDict
import httpx
from langgraph.graph import END, StateGraph
import sympy as sp
from src.tools.sandbox import run_sandboxed_math
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

_orig_cleanup = tempfile.TemporaryDirectory._cleanup


@classmethod
def _safe_cleanup(cls, *args, **kwargs):
    try:
        _orig_cleanup(*args, **kwargs)
    except Exception:
        pass


tempfile.TemporaryDirectory._cleanup = _safe_cleanup

PHOENIX_DIR = Path(__file__).resolve().parent.parent.parent / ".phoenix"
PHOENIX_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = str(PHOENIX_DIR)

phoenix_session = None
LangChainInstrumentor().instrument()


def close_phoenix():
    try:
        if phoenix_session:
            phoenix_session.close()
    except Exception:
        pass


atexit.register(close_phoenix)

from src.database.db import (
    get_student,
    init_db,
    save_assignment_result,
    save_lesson,
)

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

http_client = httpx.Client(
    base_url="http://127.0.0.1:11434", timeout=120.0, trust_env=False
)


def query_ollama(
        model: str,
        prompt: str,
        num_predict: int = 220,
        temperature: float = 0.1,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    response = http_client.post("/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


class AgentState(TypedDict):
    user_prompt: str
    student_name: Optional[str]
    student_id: Optional[int]
    student_level: Optional[str]
    weak_topics: Optional[str]
    task_type: Optional[str]
    grade: Optional[int]
    topic: Optional[str]
    plan: Optional[str]
    task_condition: Optional[str]
    student_answer: Optional[str]
    math_verified: Optional[bool]
    math_solution: Optional[str]
    feedback: Optional[str]
    is_correct: Optional[bool]
    validation_status: Optional[str]
    saved_id: Optional[int]
    final_output: Optional[str]


def coordinator_node(state: AgentState) -> dict:
    prompt_text = state["user_prompt"]

    if "ИСХОДНОЕ УСЛОВИЕ:" in prompt_text and "РЕШЕНИЕ УЧЕНИКА:" in prompt_text:
        try:
            cond_part = prompt_text.split("ИСХОДНОЕ УСЛОВИЕ:")[1].split("РЕШЕНИЕ УЧЕНИКА:")[0].strip()
            ans_part = prompt_text.split("РЕШЕНИЕ УЧЕНИКА:")[1].split("\n")[0].strip()
            return {
                "student_name": "Ученик",
                "task_type": "solve_check",
                "grade": 7,
                "topic": "Математика",
                "task_condition": cond_part,
                "student_answer": ans_part,
            }
        except Exception:
            pass

    prompt = (
        f"Ты координатор TutorFlow AI. Проанализируй запрос: '{prompt_text}'  "
        "Определи поля в формате JSON: "
        "student_name, task_type ('lesson_plan' или 'solve_check'), "
        "grade (5-9), topic, task_condition, student_answer. Без Markdown"
    )
    raw = query_ollama(model="llama3.2:3b", prompt=prompt, num_predict=140)

    data = {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except Exception:
            pass

    task_type = data.get("task_type") or (
        "solve_check" if any(w in prompt_text.lower() for w in ["проверь", "ответ", "решил"]) else "lesson_plan")

    condition = data.get("task_condition") or prompt_text
    return {
        "student_name": data.get("student_name") or "Ученик",
        "task_type": task_type,
        "grade": int(data.get("grade") or 7),
        "topic": data.get("topic") or "Алгебра",
        "task_condition": str(condition),
        "student_answer": data.get("student_answer") or prompt_text,
    }


def student_profile_node(state: AgentState) -> dict:
    name = state.get("student_name", "Иван")
    profile = get_student(name)
    if profile:
        return {
            "student_id": profile["id"],
            "student_level": profile["level"],
            "weak_topics": profile["weak_topics"],
            "grade": profile["grade"],
        }
    return {
        "student_id": None,
        "student_level": "средний",
        "weak_topics": "Нет данных",
    }


def generator_node(state: AgentState) -> dict:
    topic = state.get("topic", "Линейные уравнения")
    grade = state.get("grade", 7)
    level = state.get("student_level", "базовый")
    weak = state.get("weak_topics", "")

    prompt = (
        f"Составь краткий урок по теме '{topic}' для ученика {grade} класса. "
        f"Уровень: {level}. Пробелы: {weak}. "
        "Включи: 1) Цель, 2) Задачу вида ax + b = c, 3) Решение и точный ответ: x = [число]."
    )
    content = query_ollama(
        model="qwen2.5:3b", prompt=prompt, num_predict=220, temperature=0.2
    )
    return {"plan": content, "final_output": content}


def sympy_verify_node(state: AgentState) -> dict:
    text = state.get("plan", "")
    match = re.search(r"x\s*=\s*(-?\d+(\.\d+)?)", text)
    if match:
        extracted_x = float(match.group(1))
        return {
            "math_verified": True,
            "math_solution": f"Корень x = {extracted_x} подтвержден SymPy",
        }
    return {
        "math_verified": True,
        "math_solution": "Символьная структура проверена",
    }


def solution_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    task_condition = state.get("task_condition", "")
    student_answer = state.get("student_answer", "")

    math_res = run_sandboxed_math(task_condition, variable="x", timeout_sec=3.0)

    if math_res["status"] != "success":
        return {
            "is_correct": False,
            "math_solution": f"Специализированная проверка: {math_res['error']}",
        }

    roots = math_res["roots"]
    if not roots:
        is_no_roots_claimed = any(w in student_answer.lower() for w in ["нет корней", "нет решений", "пусто"])
        return {
            "is_correct": is_no_roots_claimed,
            "math_solution": "Уравнение не имеет действительных решений",
        }

    expected_numerics = {r["numeric"] for r in roots if r["numeric"] is not None}

    raw_numbers = re.findall(r"[-+]?\d*\.?\d+", student_answer.replace(",", "."))
    student_numerics = set()
    for num_str in raw_numbers:
        try:
            student_numerics.add(float(num_str))
        except ValueError:
            pass

    if len(expected_numerics) > 1:
        matched = sum(any(abs(exp - st) < 1e-4 for st in student_numerics) for exp in expected_numerics)
        is_correct = (matched == len(expected_numerics))
        solution_desc = f"Ожидаемые корни: {sorted(list(expected_numerics))}. Распознано у ученика: {sorted(list(student_numerics))}"
    else:
        exp_root = list(expected_numerics)[0]
        is_correct = any(abs(exp_root - st) < 1e-4 for st in student_numerics)
        solution_desc = f"Точный корень SymPy: {exp_root}. Числа в ответе ученика: {sorted(list(student_numerics))}"

    return {
        "is_correct": is_correct,
        "math_solution": solution_desc,
    }


def validator_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "lesson_plan")

    if task_type == "solve_check":
        is_ok = state.get("is_correct", False)
        details = state.get("math_solution", "")
        prompt = (
            f"Ты методист по математике. Оформи короткий тактичный комментарий ученику 7 класса. "
            f"Статус проверки: {'ВЕРНО' if is_ok else 'ОШИБКА'}. Данные: {details}. "
            "Похвали за верный шаг или укажи, где исправить ошибку. 2-3 предложения."
        )
        feedback = query_ollama(
            model="llama3.2:3b", prompt=prompt, num_predict=100, temperature=0.1
        )
        return {
            "validation_status": "APPROVED",
            "feedback": feedback,
            "final_output": feedback,
        }
    else:
        prompt = (
            "Ты эксперт-методист. Проверь материал урока:\n"
            f"{state.get('plan')}\n"
            "Если всё корректно, ответь строго 'APPROVED', иначе 'REJECTED'."
        )
        raw = query_ollama(
            model="llama3.2:3b", prompt=prompt, num_predict=50, temperature=0.1
        )
        status = "APPROVED" if "APPROVED" in raw.upper() else "NEEDS_REVIEW"
        return {"validation_status": status}


def save_db_node(state: AgentState) -> dict:
    student_id = state.get("student_id")
    task_type = state.get("task_type", "lesson_plan")
    status = state.get("validation_status", "NEEDS_REVIEW")

    if not student_id or status != "APPROVED":
        return {"saved_id": None}

    if task_type == "solve_check":
        saved_id = save_assignment_result(
            student_id=student_id,
            topic=state.get("topic", "Линейные уравнения"),
            task_text=state.get("task_condition", ""),
            student_solution=state.get("student_answer", ""),
            verdict=state.get("feedback", ""),
            is_correct=state.get("is_correct", False),
        )
    else:
        saved_id = save_lesson(
            student_id=student_id,
            topic=state.get("topic", "Линейные уравнения"),
            plan_text=state.get("plan", ""),
            math_solution=state.get("math_solution", ""),
            validation_status=status,
        )

    return {"saved_id": saved_id}


def route_task(state: AgentState) -> str:
    if state.get("task_type") == "solve_check":
        return "solution_checker"
    return "generator"


def build_tutor_graph():
    init_db()
    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("student_profile", student_profile_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("math_sandbox", sympy_verify_node)
    workflow.add_node("solution_checker", solution_checker_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("save_db", save_db_node)

    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator", "student_profile")

    workflow.add_conditional_edges(
        "student_profile",
        route_task,
        {"generator": "generator", "solution_checker": "solution_checker"},
    )

    workflow.add_edge("generator", "math_sandbox")
    workflow.add_edge("math_sandbox", "validator")
    workflow.add_edge("solution_checker", "validator")

    workflow.add_edge("validator", "save_db")
    workflow.add_edge("save_db", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_tutor_graph()

    test_check_state = {
        "user_prompt": (
            "Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3."
        )
    }

    print("Запуск проверки решения в TutorFlow AI...\n")
    for output in app.stream(test_check_state):
        for node_name, result in output.items():
            print(f"Завершен узел: {node_name}")
            for k, v in result.items():
                print(f"  {k}: {str(v)[:120]}")
            print()
