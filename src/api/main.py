import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from src.agents.graph import build_tutor_graph
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tutor_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tutor_app
    tutor_app = build_tutor_graph()
    yield
    tutor_app = None


app = FastAPI(
    title="TutorFlow AI API",
    description="REST API для автоматической проверки математических решений с использованием LangGraph и SymPy.",
    version="1.0.0",
    lifespan=lifespan,
)


class CheckSolutionRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=5,
        description="Текст задачи с ответом ученика",
        example="Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3.",
    )


class CheckSolutionResponse(BaseModel):
    is_correct: Optional[bool] = Field(
        None, description="Вердикт валидатора по решению"
    )
    math_solution: Optional[str] = Field(
        None, description="Результат аналитической песочницы SymPy"
    )
    feedback: Optional[str] = Field(
        None, description="Педагогическое объяснение для ученика"
    )
    latency_sec: float = Field(
        ..., description="Время обработки запроса в секундах"
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "TutorFlow AI"}


@app.post(
    "/api/v1/check-solution",
    response_model=CheckSolutionResponse,
    status_code=status.HTTP_200_OK,
)
def check_solution(request: CheckSolutionRequest):
    if tutor_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Граф агентов TutorFlow не инициализирован",
        )

    start_time = time.perf_counter()
    final_state: Dict[str, Any] = {}

    try:
        for output in tutor_app.stream({"user_prompt": request.prompt}):
            for _, result in output.items():
                final_state.update(result)

        elapsed = round(time.perf_counter() - start_time, 2)

        return CheckSolutionResponse(
            is_correct=final_state.get("is_correct"),
            math_solution=final_state.get("math_solution"),
            feedback=final_state.get("feedback"),
            latency_sec=elapsed,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка выполнения пайплайна: {str(exc)}",
        )