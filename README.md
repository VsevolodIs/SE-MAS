# TutorFlow AI

Мультиагентная система на базе **LangGraph** и **FastAPI** для автоматизированной проверки решений и генерации учебных материалов по математике (5–9 классы). 

Проект объединяет детерминированную песочницу **SymPy** для символьных алгебраических вычислений, локальный инференс LLM через **Ollama**, слой персистентности на **SQLite**, модуль рекомендаций **Two-Tower (PyTorch)** и платформу наблюдаемости **Arize Phoenix**.

---

## 1. Стек технологий

- **Оркестрация агентов:** LangGraph / LangChain
- **Локальный инференс LLM:** Ollama (`llama3.2:3b`, `qwen2.5:3b`)
- **Символьные вычисления:** SymPy
- **Сервисный слой:** FastAPI, Uvicorn, Pydantic
- **База данных:** SQLite
- **Рекомендательная система:** PyTorch (Two-Tower architecture)
- **Мониторинг и Observability:** Arize Phoenix, OpenTelemetry
- **Тестирование и CI/CD:** pytest, GitHub Actions

---

## 2. Быстрый старт и установка

### Клонирование и настройка окружения (PowerShell / Windows)

```powershell
git clone https://github.com/VsevolodIs/SE-MAS.git
cd SE-MAS

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Запуск и загрузка моделей в Ollama

Убедитесь, что демон Ollama запущен:
```powershell
ollama serve
```

В отдельном окне терминала загрузите рабочие модели:
```powershell
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

---

## 3. Запуск веб-сервера (FastAPI)

Запуск сервиса в режиме разработки:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Интерактивная документация (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Проверка работоспособности (Healthcheck):** `http://127.0.0.1:8000/health`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## 4. Примеры вызовов API

### 4.1. Проверка решения ученика (`POST /api/v1/check-solution`)

```powershell
$body = @{
  prompt = "Проверь решение задачи 2*x + 4 = 10. Ученик ответил x = 3. Верно ли это?"
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/check-solution" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

**Пример ответа:**
```json
{
  "is_correct": true,
  "math_solution": "Точный корень SymPy: 3.0. Числа в ответе ученика: [3.0]",
  "feedback": "Отлично, ты прав! Точное решение имеет корень 3.",
  "latency_sec": 0.12
}
```

### 4.2. Универсальный маршрут агента (`POST /api/v1/agent`)

```powershell
$body = @{
  user_prompt = "Реши уравнение 2x + 5 = 15"
  grade = 7
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/agent" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

---

## 5. Трассировка и наблюдаемость (Arize Phoenix)

Для включения сквозной трассировки графа LangGraph:

1. Установите зависимости телеметрии (с фиксацией совместимости GraphQL):
```powershell
python -m pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain "graphql-core==3.2.3"
```

2. Запустите веб-интерфейс Phoenix в отдельном терминале:
```powershell
phoenix serve
```
Интерфейс доступен по адресу: `http://127.0.0.1:6006`

3. Запустите API с включенным сборщиком телеметрии:
```powershell
$env:PHOENIX_ENABLED = "1"
$env:PHOENIX_COLLECTOR_ENDPOINT = "http://127.0.0.1:6006"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## 6. Запуск тестов и валидация

### Модульные и интеграционные тесты (pytest)
```powershell
python -m pytest tests -v
```

### Автономный запуск графа агентов
```powershell
python -m src.agents.graph
```

### Обучение и оценка модели Two-Tower RecSys
```powershell
python src/models/recsys_two_tower.py
```

---

## 7. Структура проекта

```text
├── src/
│   ├── agents/               # Мультиагентный граф LangGraph и системные инструкции
│   ├── api/                  # FastAPI сервис, схемы Pydantic и эндпоинты
│   ├── database/             # Слой работы с базой данных SQLite
│   ├── models/               # Нейросетевая рекомендательная система Two-Tower (PyTorch)
│   ├── observability/        # Настройка трассировки Arize Phoenix и OpenTelemetry
│   ├── skills/               # Спецификации навыков агентов (SKILL.md)
│   └── tools/                # Песочница SymPy с AST-валидацией и таймаутами
├── tests/                    # Набор автотестов (API, песочница, метрики, Negative Sampling)
├── benchmarks/               # Эталонный датасет сценариев (dataset.jsonl)
├── scripts/                  # Скрипты интеграционного тестирования и аудита
├── requirements.txt          # Зависимости проекта
└── README.md
```
