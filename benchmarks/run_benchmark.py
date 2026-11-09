import json
import os
from pathlib import Path
import time
import traceback
from ollama import Client

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

client = Client(host="http://127.0.0.1:11434", timeout=120)

MODELS = ["qwen2.5:0.5b", "llama3.2:3b", "mistral:7b"]

TEST_PROMPTS = [
    (
        "Ты — координатор TutorFlow AI. Классифицируй запрос репетитора: "
        "'Подготовь проверочную работу по дробям для 6 класса на 20 минут'. "
        "Верни только валидный JSON с ключами: task_type, grade, topic, duration_min. "
        "Без Markdown и лишнего текста."
    ),
    (
        "Составь краткий план урока математики для 7 класса по теме 'Линейные уравнения' "
        "на 45 минут. Назови ровно 3 этапа и распредели время."
    ),
    (
        "Найди корень уравнения: 3*(x - 4) = 2*x + 5. "
        "Напиши шаги раскрытия скобок и финальный ответ x = ..."
    ),
]

results = []

for model in MODELS:
  print(f"Тестируем модель: {model}", flush=True)

  for idx, prompt in enumerate(TEST_PROMPTS):
    print(f"  -> Запрос {idx + 1}/3 выполняется...", end="", flush=True)
    start_time = time.perf_counter()

    try:
      response = client.chat(
          model=model,
          messages=[{"role": "user", "content": prompt}],
          options={
              "temperature": 0.1,
              "num_predict": 220,
              "repeat_penalty": 1.2,
          },
      )
      elapsed = time.perf_counter() - start_time
      eval_count = response.get("eval_count", 0)
      eval_duration = response.get("eval_duration", 1) / 1e9
      tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0

      print(
          f" Готово! ({elapsed:.2f} с | {tokens_per_sec:.1f} tok/s)",
          flush=True,
      )

      results.append({
          "model": model,
          "prompt_id": idx + 1,
          "prompt": prompt,
          "latency_sec": round(elapsed, 3),
          "tokens_per_sec": round(tokens_per_sec, 1),
          "output": response["message"]["content"].strip(),
      })
    except Exception as e:
      print(f"\n[Ошибка на запросе {idx + 1}]: {e}", flush=True)
      traceback.print_exc()

BASE_DIR = Path(__file__).resolve().parent
output_file = BASE_DIR / "results.json"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
  json.dump(results, f, ensure_ascii=False, indent=2)

print(
    f"\nЗаписано записей: {len(results)}. Файл сохранён: {output_file}",
    flush=True,
)