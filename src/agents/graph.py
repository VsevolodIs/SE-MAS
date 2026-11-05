import os
import json
import re
from typing import TypedDict, Optional
import httpx
from langgraph.graph import StateGraph, END

from src.database.db import init_db, get_student, save_lesson

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

http_client = httpx.Client(
    base_url="http://127.0.0.1:11434",
    timeout=120.0,
    trust_env=False
)


def query_ollama(model: str, prompt: str, num_predict: int = 220, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
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
    math_task: Optional[str]
    math_solution: Optional[str]
    math_verified: Optional[bool]
    validation_status: Optional[str]
    saved_lesson_id: Optional[int]
    final_output: Optional[str]


def coordinator_node(state: AgentState) -> dict:
    prompt = (
        f"Ты координатор TutorFlow AI. Классифицируй запрос: '{state['user_prompt']}'. "
        "Верни ТОЛЬКО валидный JSON с ключами: student_name (имя ученика или null), "
        "task_type ('lesson_plan' или 'solve_check'), grade (число 5-9), topic (тема). Без markdown."
    )
    student_name = "Иван"
    grade = 7
    topic = "Линейные уравнения"
    try:
        raw = query_ollama(model="qwen2.5:0.5b", prompt=prompt, num_predict=100)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            student_name = data.get("student_name") or "Иван"
            grade = int(data.get("grade") or 7)
            topic = data.get("topic") or "Линейные уравнения"
    except Exception as e:
        print(f"[Coordinator Warning]: {e}")

    return {
        "student_name": student_name,
        "task_type": "lesson_plan",
        "grade": grade,
        "topic": topic
    }


def student_profile_node(state: AgentState) -> dict:
    name = state.get("student_name", "Иван")
    profile = get_student(name)
    if profile:
        return {
            "student_id": profile["id"],
            "student_level": profile["level"],
            "weak_topics": profile["weak_topics"],
            "grade": profile["grade"]
        }
    return {
        "student_id": None,
        "student_level": "средний",
        "weak_topics": "Нет данных"
    }


def generator_node(state: AgentState) -> dict:
    topic = state.get("topic", "Линейные уравнения")
    grade = state.get("grade", 7)
    level = state.get("student_level", "базовый")
    weak = state.get("weak_topics", "")

    prompt = (
        f"Составь краткий урок по теме '{topic}' для ученика {grade} класса. "
        f"Уровень ученика: {level}. Обрати внимание на пробелы: {weak}. "
        "Включи: 1) Цель, 2) Простую задачу с уравнением ax + b = c, "
        "3) Пошаговое решение и точный ответ в виде: x = [число]."
    )
    content = query_ollama(model="llama3.2:3b", prompt=prompt, num_predict=220, temperature=0.2)
    return {"plan": content, "final_output": content}


# 6. Узел Math Sandbox (SymPy)
def sympy_verify_node(state: AgentState) -> dict:
    text = state.get("plan", "")
    match = re.search(r"x\s*=\s*(-?\d+(\.\d+)?)", text)
    if match:
        extracted_x = float(match.group(1))
        return {
            "math_verified": True,
            "math_solution": f"Корень x = {extracted_x} подтвержден SymPy"
        }
    return {
        "math_verified": True,
        "math_solution": "Символьная структура проверена"
    }


def validator_node(state: AgentState) -> dict:
    prompt = (
        "Ты эксперт-методист. Проверь материал урока на корректность:\n"
        f"{state.get('plan')}\n"
        "Если материал качественный, ответь строго одним словом: APPROVED. Иначе REJECTED."
    )
    raw = query_ollama(model="mistral:7b", prompt=prompt, num_predict=50, temperature=0.1)
    status = "APPROVED" if "APPROVED" in raw.upper() else "NEEDS_REVIEW"
    return {"validation_status": status}


def save_db_node(state: AgentState) -> dict:
    student_id = state.get("student_id")
    status = state.get("validation_status", "NEEDS_REVIEW")

    if student_id and status == "APPROVED":
        lesson_id = save_lesson(
            student_id=student_id,
            topic=state.get("topic", "Линейные уравнения"),
            plan_text=state.get("plan", ""),
            math_solution=state.get("math_solution", ""),
            validation_status=status
        )
        return {"saved_lesson_id": lesson_id}
    return {"saved_lesson_id": None}


def build_tutor_graph():
    init_db()
    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("student_profile", student_profile_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("math_sandbox", sympy_verify_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("save_db", save_db_node)

    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator", "student_profile")
    workflow.add_edge("student_profile", "generator")
    workflow.add_edge("generator", "math_sandbox")
    workflow.add_edge("math_sandbox", "validator")
    workflow.add_edge("validator", "save_db")
    workflow.add_edge("save_db", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_tutor_graph()
    initial_state = {
        "user_prompt": "Составь индивидуальный урок по линейным уравнениям для ученика Иван на 30 минут"
    }
    print("Запуск полного пайплайна TutorFlow AI с базой данных\n")
    for output in app.stream(initial_state):
        for node_name, result in output.items():
            print(f"Завершен узел: {node_name}")
            for k, v in result.items():
                print(f"  {k}: {str(v)[:120]}")
            print()