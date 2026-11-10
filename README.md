# SE-MAS / TutorFlow AI

Мультиагентный учебный прототип для математики 5–9 классов. **Не production-ready.**

## Запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
python -m uvicorn src.api.main:app --reload
```

`GET /health` — здоровье API; `POST /api/v1/agent` — все маршруты графа; `POST /api/v1/check-solution` — только проверка решения.

## Проверки

```powershell
python -m pytest -q -m "not slow"
python -m scripts.audit_dataset_routes
python scripts/run_graph_dataset.py --limit 3
```

Скрипт `audit_dataset_routes` **не вызывает Ollama**, а проверяет лишь маршрутизацию. Полный `run_graph_dataset.py` использует реальные локальные модели; результаты требуют отдельной содержательной оценки.

## Модели

Генератор: `qwen2.5:3b`, решатель: `llama3.2:3b`. Валидатор выбирает *другое семейство* относительно модели решателя/генератора. Модель можно задать через `TUTORFLOW_VALIDATOR_MODEL`, но выбрать ту же семью нельзя.

## Границы безопасности и зрелость

- SymPy исполняется в отдельном дочернем процессе с тайм-аутом; **это не полноценная изоляция VM/MicroVM**.
- Публичный API не принимает `student_name` и `session_id`: подключение к именным профилям требует аутентификации и проверки прав.
- Локальные события и узловые спаны без содержания запросов пишутся в `.runtime/events.jsonl`. Это **не полноценная система алертинга и централизованной observability**.
- `APPROVED` для педагогических материалов — предварительное решение LLM, не строгая математическая гарантия.
- Two-Tower RecSys обучается на синтетических взаимодействиях и не подключена к публичному API; её метрики нельзя считать продуктовой эффективностью.

Актуальные ограничения и этапы доработки: `docs/TECHNICAL_AUDIT_TZ.md`.
