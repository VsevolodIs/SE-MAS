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
from src.database.db import get_session_history, save_session_message


PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return ""

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
    base_url="http://127.0.0.1:11434", timeout=180.0, trust_env=False
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
    session_id: Optional[str]
    chat_history: Optional[list]
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
    prompt = state.get("prompt") or state.get("user_prompt") or ""
    session_id = state.get("session_id", "default_session")
    student_name = state.get("student_name", "Ученик")
    grade = state.get("grade", 7)
    topic = state.get("topic", "Математика")

    task_condition = state.get("task_condition") or ""
    student_answer = state.get("student_answer") or ""

    if not task_condition:
        cond_match = re.search(r"задач[еиу]?\s+(.+?)(?=\.\s+[А-ЯЁ]|\s+ученик|\s+ответ|$)", prompt, re.IGNORECASE)
        if cond_match:
            task_condition = cond_match.group(1).strip().rstrip(".;, ")
        else:
            eq_match = re.search(r"([0-9a-zA-Z_\*\+\-\/\^\(\)\.\s]+=\s*[-+]?[0-9a-zA-Z_\*\+\-\/\^\(\)\.]+)", prompt)
            if eq_match:
                task_condition = eq_match.group(1).strip().rstrip(".;, ")

    if not student_answer:
        ans_match = re.search(
            r"(?:ученик\s+ответил|ответ(?:ом)?\s+является|ответ\s*[:=])\s*([^\n.]+)",
            prompt,
            re.IGNORECASE,
        )
        if ans_match:
            student_answer = ans_match.group(1).strip().rstrip(".;, ")
        else:
            ans_val_match = re.search(r"x\s*=\s*[-+]?\d*\.?\d+", prompt, re.IGNORECASE)
            if ans_val_match:
                student_answer = ans_val_match.group(0).strip()

    p_lower = prompt.lower()
    if "проверь" in p_lower or "решение" in p_lower or student_answer:
        task_type = "solve_check"
    elif any(w in p_lower for w in ["персонализ", "подбери под уровень", "слабые темы"]):
        task_type = "personalize"
    elif any(w in p_lower for w in ["поясни", "объясни подробнее", "уточни", "не понял"]):
        task_type = "clarify"
    elif any(w in p_lower for w in ["по шагам", "пошагов", "в несколько действий"]):
        task_type = "multi_step"
    elif any(w in p_lower for w in ["бомба", "взлом", "политика", "код на c++"]):
        task_type = "safety"
    elif any(w in p_lower for w in ["реши", "найди корень", "вычисли", "сколько будет"]):
        task_type = "solve_math"
    else:
        task_type = "pedagogical"

    history = list(state.get("chat_history") or [])
    if prompt:
        history.append({"role": "user", "content": prompt})

    return {
        "prompt": prompt,
        "user_prompt": prompt,
        "session_id": session_id,
        "student_name": student_name,
        "grade": grade,
        "topic": topic,
        "task_condition": task_condition,
        "student_answer": student_answer,
        "task_type": task_type,
        "chat_history": history,
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
    topic = state.get("topic") or "Математика"
    grade = state.get("grade") or 7
    level = state.get("student_level") or "базовый"
    weak = state.get("weak_topics") or "не указаны"
    user_request = state.get("user_prompt") or ""

    prompt = (
        "Ты — преподаватель математики для учеников 5–9 классов.\n"
        "Твоя задача — выполнить конкретный запрос пользователя.\n\n"

        f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_request}\n\n"

        "ДАННЫЕ ОБ УЧЕНИКЕ:\n"
        f"Класс: {grade}\n"
        f"Тема из профиля: {topic}\n"
        f"Уровень: {level}\n"
        f"Слабые темы: {weak}\n\n"

        "ПРАВИЛА:\n"
        "1. Следуй прежде всего явному запросу пользователя.\n"
        "2. Если пользователь указал тему, используй именно её.\n"
        "3. Если пользователь указал класс, учитывай его.\n"
        "4. Если указано количество заданий, создай ровно столько.\n"
        "5. Если пользователь просит ответы, добавь ответы к каждому заданию.\n"
        "6. Если пользователь просит план урока, составь план урока, "
        "а не просто список упражнений.\n"
        "7. Если пользователь просит объяснение, дай пошаговое объяснение.\n"
        "8. Проверяй математическую правильность решений и ответов.\n"
        "9. Не подменяй запрошенную тему линейными уравнениями.\n"
        "10. Отвечай на русском языке, понятно для указанного класса.\n\n"

        "Выдай только результат выполнения запроса."
    )

    content = query_ollama(
        model="qwen2.5:3b",
        prompt=prompt,
        num_predict=600,
        temperature=0.2,
    )

    return {
        "plan": content,
        "final_output": content,
    }

def sympy_verify_node(state: AgentState) -> dict:
    text = state.get("plan", "")
    task_cond = state.get("task_condition", "")

    if task_cond:
        res = run_sandboxed_math(task_cond, timeout_sec=3.0)
        if res.get("roots"):
            root_val = res["roots"][0].get("numeric")
            return {
                "math_verified": True,
                "math_solution": f"Корень x = {root_val} подтвержден SymPy",
            }

    match = re.search(r"x\s*=\s*(-?\d+(\.\d+)?)", text)
    if match:
        extracted_x = float(match.group(1))
        return {
            "math_verified": True,
            "math_solution": f"Точный корень SymPy: {extracted_x}",
        }
    return {
        "math_verified": True,
        "math_solution": "Символьная структура проверена",
    }


def solution_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    task_condition = state.get("task_condition", "")
    student_answer = state.get("student_answer", "")

    if not student_answer or not str(student_answer).strip():
        return {
            "is_correct": False,
            "math_verified": False,
            "math_solution": "Ответ ученика отсутствует или не был извлечён.",
            "requires_llm_fallback": True,
        }

    math_res = run_sandboxed_math(task_condition, variable="x", timeout_sec=5.0)
    status = math_res.get("status")
    details = math_res.get("details", {})

    if status in ["unsupported_format", "math_error", "timeout_error", "security_error"]:
        return {
            "is_correct": None,
            "math_verified": False,
            "math_solution": (
                f"Инструментальная песочница не смогла однозначно вычислить задачу ({math_res.get('error')}) "
                "Требуется семантическая проверка шагов валидатором"
            ),
            "requires_llm_fallback": True,
        }

    if details.get("type") == "inequality":
        solution_str = details.get("solution_str", "")
        if ("<" in student_answer and ">" in student_answer) and ("&" in solution_str):
            return {
                "is_correct": False,
                "math_verified": False,
                "math_solution": f"Ошибка: решение неравенства — интервал {solution_str}, а не объединение лучей",
                "requires_llm_fallback": False,
            }
        return {
            "is_correct": None,
            "math_verified": False,
            "math_solution": f"Аналитическое решение неравенства SymPy: {solution_str}",
            "requires_llm_fallback": True,
        }

    forbidden_roots = details.get("forbidden_roots", [])
    clean_student_ans = re.sub(r"=\s*0\b", "", student_answer)
    raw_numbers = re.findall(r"[-+]?\d*\.?\d+", clean_student_ans.replace(",", "."))
    student_numerics = set()
    for num_str in raw_numbers:
        try:
            val = float(num_str)
            student_numerics.add(val)
        except ValueError:
            pass

    for forb in forbidden_roots:
        try:
            forb_val = float(sp.sympify(forb).evalf())
            if any(abs(forb_val - st) < 1e-4 for st in student_numerics):
                return {
                    "is_correct": False,
                    "math_verified": False,
                    "math_solution": f"Ошибка ОДЗ: корень x = {forb} обращает знаменатель в ноль и является посторонним",
                    "requires_llm_fallback": False,
                }
        except Exception:
            pass

    roots = math_res.get("roots", [])
    if not roots:
        is_no_roots_claimed = any(w in student_answer.lower() for w in ["нет корней", "нет решений", "пусто", "решений нет"])
        return {
            "is_correct": is_no_roots_claimed,
            "math_verified": is_no_roots_claimed,
            "math_solution": "Уравнение не имеет действительных решений в области допустимых значений.",
            "requires_llm_fallback": False,
        }

    expected_numerics = {r["numeric"] for r in roots if r["numeric"] is not None}

    if len(expected_numerics) > 1:
        matched = sum(any(abs(exp - st) < 1e-4 for st in student_numerics) for exp in expected_numerics)
        is_correct = (matched == len(expected_numerics)) and (len(student_numerics) <= len(expected_numerics))
        solution_desc = f"Ожидаемые корни SymPy: {sorted(list(expected_numerics))}. Числа в ответе ученика: {sorted(list(student_numerics))}"
    elif len(expected_numerics) == 1:
        exp_root = list(expected_numerics)[0]
        is_correct = any(abs(exp_root - st) < 1e-4 for st in student_numerics)
        solution_desc = f"Точный корень SymPy: {exp_root}. Числа в ответе ученика: {sorted(list(student_numerics))}"
    else:
        is_correct = None
        solution_desc = f"Символьный ответ SymPy: {details.get('solution_str')}"

    return {
        "is_correct": is_correct,
        "math_verified": bool(is_correct) if is_correct is not None else False,
        "math_solution": solution_desc,
        "requires_llm_fallback": is_correct is None,
    }


def validator_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "lesson_plan")

    def _query_validator_llm(prompt_text: str, max_tokens: int = 140, temp: float = 0.1) -> str:
        try:
            return query_ollama(model="mistral:7b", prompt=prompt_text, num_predict=max_tokens, temperature=temp)
        except Exception:
            pass
        try:
            return query_ollama(model="llama3.2:3b", prompt=prompt_text, num_predict=max_tokens, temperature=temp)
        except Exception:
            pass
        return query_ollama(model="qwen2.5:3b", prompt=prompt_text, num_predict=max_tokens, temperature=temp)

    if task_type == "solve_check":
        is_ok = state.get("is_correct")
        details = state.get("math_solution", "")
        task_condition = state.get("task_condition", "")
        student_answer = state.get("student_answer", "")
        requires_fallback = state.get("requires_llm_fallback", False)

        if requires_fallback or is_ok is None:
            validator_spec = load_prompt("validator.md")
            prompt = (
                f"{validator_spec}\n\n"
                f"УСЛОВИЕ ЗАДАЧИ: {task_condition}\n"
                f"ОТВЕТ УЧЕНИКА: {student_answer}\n"
                f"КОНТЕКСТ/РЕШЕНИЕ: {details}\n\n"
                "Оцени ответ строго по правилам выше"
            )
            raw_response = _query_validator_llm(prompt, max_tokens=140, temp=0.0)

            lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
            first_line = lines[0].upper() if lines else ""

            if "ВЕРДИКТ: ОШИБКА" in first_line or "ОШИБКА" in first_line:
                evaluated_is_correct = False
                status = "APPROVED"
            elif "ВЕРДИКТ: ВЕРНО" in first_line or "ВЕРНО" in first_line:
                evaluated_is_correct = True
                status = "APPROVED"
            else:
                evaluated_is_correct = None
                status = "MANUAL_VERIFICATION_REQUIRED"

            feedback_text = "\n".join(lines[1:]) if len(lines) > 1 else raw_response

            return {
                "is_correct": evaluated_is_correct,
                "validation_status": status,
                "feedback": feedback_text,
                "final_output": feedback_text,
            }

        prompt = (
            "Ты методист по математике. Оформи короткий тактичный комментарий ученику 7 класса.\n"
            f"Статус проверки: {'ВЕРНО' if is_ok else 'ОШИБКА'}.\n"
            f"Математические данные проверки: {details}.\n"
            f"Ответ ученика: {student_answer}.\n"
            "Похвали за верный шаг или дружелюбно объясни, где допущена ошибка. 2-3 предложения."
        )
        feedback = _query_validator_llm(prompt, max_tokens=120, temp=0.1)
        status = "APPROVED" if is_ok is not None else "MANUAL_VERIFICATION_REQUIRED"

        return {
            "is_correct": is_ok,
            "validation_status": status,
            "feedback": feedback,
            "final_output": feedback,
        }

    else:
        prompt = (
            "Ты эксперт-методист. Проверь материал урока:\n"
            f"{state.get('plan')}\n"
            "Если всё корректно, ответь строго 'APPROVED', иначе 'REJECTED'."
        )
        raw = _query_validator_llm(prompt, max_tokens=50, temp=0.1)
        status = "APPROVED" if "APPROVED" in raw.upper() else "NEEDS_REVIEW"
        return {"validation_status": status}

def solver_agent_node(state: AgentState) -> dict:
    task = state.get("task_condition") or state.get("user_prompt", "")
    grade = state.get("grade", 7)
    topic = state.get("topic", "Алгебра")

    math_res = run_sandboxed_math(task, variable="x", timeout_sec=3.0)
    details = math_res.get("details", {})
    roots = math_res.get("roots", [])

    if math_res.get("status") == "success":
        if details.get("type") == "inequality":
            solution_summary = (
                f"Решение неравенства: {details.get('solution_str')}"
            )
        elif roots:
            roots_str = ", ".join(str(r["symbolic"]) for r in roots)
            solution_summary = f"Точные корни уравнения: {roots_str}"
        else:
            solution_summary = (
                "Уравнение не имеет действительных корней (пустое множество)"
            )
    else:
        solution_summary = (
            "Символьный расчет недоступен, требуется текстовый разбор"
        )

    solver_spec = load_prompt("solver.md")
    prompt = (
        f"{solver_spec}\n\n"
        f"Тема: {topic}, класс: {grade}.\n"
        f"Условие: {task}\n"
        f"Точный математический ответ: {solution_summary}\n\n"
        "Сформируй понятное пошаговое решение ученику:"
    )

    explanation = query_ollama(
        model="llama3.2:3b", prompt=prompt, num_predict=250, temperature=0.2
    )

    return {
        "math_solution": solution_summary,
        "plan": explanation,
        "final_output": explanation,
    }

def safety_node(state: AgentState) -> dict:
    prompt = state.get("user_prompt", "")

    system_msg = (
        "Ты виртуальный помощник репетитора по математике 5-9 классов TutorFlow AI\n"
        "Пользователь задал вопрос не по теме математики или нарушающий границы репетиторства\n"
        f"Запрос пользователя: '{prompt}'\n\n"
        "Инструкция:\n"
        "1. Вежливо и кратко откажись отвечать на нерелевантную тему\n"
        "2. Напомни, что ты специализируешься исключительно на школьной алгебре и геометрии 5–9 классов\n"
        "3. Предложи разобрать математическую задачу или тему урока (1-2 предложения)"
    )

    response = query_ollama(
        model="llama3.2:3b", prompt=system_msg, num_predict=80, temperature=0.1
    )

    return {
        "validation_status": "APPROVED",
        "final_output": response,
        "feedback": "Запрос обработан фильтром безопасности / оффтопа",
    }

def save_db_node(state: AgentState) -> dict:
    session_id = state.get("session_id") or "default_session"
    final_output = state.get("final_output") or state.get("feedback") or ""
    if final_output:
        save_session_message(session_id, "assistant", final_output)

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
    task_type = state.get("task_type", "pedagogical")

    if task_type == "solve_check":
        return "solution_checker"
    elif task_type == "solve_math":
        return "solver_agent"
    elif task_type == "safety":
        return "safety_node"
    elif task_type in ["personalize", "clarify", "multi_step", "pedagogical"]:
        return "pedagogical_generator"
    return "pedagogical_generator"


def build_tutor_graph():
    init_db()
    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("student_profile", student_profile_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("solver_agent", solver_agent_node)  # Новый узел
    workflow.add_node("safety_node", safety_node)  # Новый узел
    workflow.add_node("math_sandbox", sympy_verify_node)
    workflow.add_node("solution_checker", solution_checker_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("save_db", save_db_node)

    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator", "student_profile")

    workflow.add_conditional_edges(
        "student_profile",
        route_task,
        {
            "solution_checker": "solution_checker",
            "solver_agent": "solver_agent",
            "pedagogical_generator": "generator",
            "safety_node": "safety_node",
        },
    )

    workflow.add_edge("solution_checker", "validator")
    workflow.add_edge("solver_agent", "validator")
    workflow.add_edge("generator", "math_sandbox")
    workflow.add_edge("math_sandbox", "validator")

    workflow.add_edge("safety_node", "save_db")
    workflow.add_edge("validator", "save_db")
    workflow.add_edge("save_db", END)

    return workflow.compile()


if __name__ == "__main__":
    phoenix_session = px.launch_app()
    LangChainInstrumentor().instrument()

    app = build_tutor_graph()
    test_input = {
        "prompt": "Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3.",
        "student_name": "Иван",
        "task_condition": "2*x + 4 = 10",
        "student_answer": "x = 3",
        "task_type": "solve_check"
    }
    result = app.invoke(test_input)
    print("\nИтоговый результат:")
    print(result)
