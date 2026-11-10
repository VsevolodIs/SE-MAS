import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from src.agents.graph import build_tutor_graph
from src.observability.telemetry import record_request
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


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
        json_schema_extra={"example": "Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3."},
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


class AgentRequest(BaseModel):
    user_prompt: str = Field(..., min_length=2, max_length=4000)
    grade: Optional[int] = Field(None, ge=5, le=9)
    student_level: Optional[str] = Field(None, max_length=100)
    weak_topics: Optional[str] = Field(None, max_length=500)


class AgentResponse(BaseModel):
    trace_id: Optional[str] = None
    task_type: Optional[str] = None
    validation_status: Optional[str] = None
    final_output: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_sec: float


@app.post("/api/v1/agent", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    """Public entrypoint to the entire multi-agent workflow, not just answer-checking."""
    if tutor_app is None:
        raise HTTPException(status_code=503, detail="Граф агентов не инициализирован")
    started = time.perf_counter()
    # Public access to a named student profile requires authentication and authorization.
    payload = request.model_dump(exclude_none=True)
    try:
        output = tutor_app.invoke(payload)
    except Exception:
        raise HTTPException(status_code=503, detail="Не удалось выполнить агентный запрос")
    elapsed = round(time.perf_counter() - started, 3)
    record_request(endpoint="/api/v1/agent", task_type=output.get("task_type"),
                   validation_status=output.get("validation_status"), latency_sec=elapsed,
                   trace_id=output.get("trace_id"))
    return AgentResponse(
        trace_id=output.get("trace_id"), task_type=output.get("task_type"),
        validation_status=output.get("validation_status"),
        final_output=output.get("final_output"),
        is_correct=output.get("is_correct"),
        latency_sec=elapsed,
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
        input_payload = {
            "user_prompt": request.prompt,
        }

        for output in tutor_app.stream(input_payload):
            for _, result in output.items():
                if isinstance(result, dict):
                    final_state.update(result)

        elapsed = round(time.perf_counter() - start_time, 2)

        if final_state.get("task_type") != "solve_check":
            raise HTTPException(status_code=422,
                                detail="Запрос не распознан как проверка решения")
        is_correct = final_state.get("is_correct")
        feedback = final_state.get("feedback")
        record_request(endpoint="/api/v1/check-solution", task_type=final_state.get("task_type"),
                       validation_status=final_state.get("validation_status"), latency_sec=elapsed,
                       trace_id=final_state.get("trace_id"))
        return CheckSolutionResponse(
            is_correct=is_correct,
            math_solution=final_state.get("math_solution"),
            feedback=feedback,
            latency_sec=elapsed,
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка обработки запроса",
        )