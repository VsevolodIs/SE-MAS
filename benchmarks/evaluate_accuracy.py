import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import build_tutor_graph

DATA_PATH = PROJECT_ROOT / "benchmarks" / "benchmark_sample.json"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "accuracy_results.json"


def run_accuracy_evaluation():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл с тестовыми кейсами не найден: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Запуск оценки точности TutorFlow AI на выборке из {len(test_cases)} задач")

    app = build_tutor_graph()

    detailed_results = []
    correct_verdicts = 0
    math_sandbox_success = 0
    total_time = 0.0

    for idx, item in enumerate(test_cases, start=1):
        print(f"\n[{idx}/{len(test_cases)}] Тест: {item['id']} | {item['prompt']}")

        start_t = time.perf_counter()
        final_state = {}

        try:
            for output in app.stream({"user_prompt": item["prompt"]}):
                for node_name, result in output.items():
                    final_state.update(result)

            elapsed = time.perf_counter() - start_t
            total_time += elapsed

            predicted_correct = final_state.get("is_correct")
            math_sol = final_state.get("math_solution", "")

            is_match = (predicted_correct == item["expected_is_correct"])
            if is_match:
                correct_verdicts += 1

            sandbox_ok = final_state.get("math_verified") is True
            if sandbox_ok:
                math_sandbox_success += 1

            print(
                f"Вердикт системы: {predicted_correct} (Ожидалось: {item['expected_is_correct']}) | Совпадение: {is_match}")
            print(f"Решение песочницы: {math_sol}")
            print(f"Время выполнения: {elapsed:.2f} с")

            detailed_results.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "expected_is_correct": item["expected_is_correct"],
                "predicted_is_correct": predicted_correct,
                "is_match": is_match,
                "math_solution": math_sol,
                "latency_sec": round(elapsed, 2)
            })

        except Exception as e:
            print(f"Ошибка кейса {item['id']}: {e}")
            detailed_results.append({
                "id": item["id"],
                "error": str(e),
                "is_match": False
            })

    total_count = len(test_cases)
    accuracy = round((correct_verdicts / total_count) * 100, 2) if total_count else 0.0
    sandbox_accuracy = round((math_sandbox_success / total_count) * 100, 2) if total_count else 0.0
    avg_latency = round(total_time / total_count, 2) if total_count else 0.0

    summary = {
        "total_cases": total_count,
        "correct_verdicts": correct_verdicts,
        "verdict_accuracy_percent": accuracy,
        "confirmed_math_verified_percent": sandbox_accuracy,
        "avg_case_latency_sec": avg_latency,
        "results": detailed_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("ИТОГИ ОЦЕНКИ ТОЧНОСТИ:")
    print(f"  Всего тестов: {total_count}")
    print(f"  Верных вердиктов: {correct_verdicts}/{total_count} ({accuracy}%)")
    print(f"  Строго подтверждено математических проверок: {math_sandbox_success}/{total_count} ({sandbox_accuracy}%)")
    print(f"  Среднее время на задачу: {avg_latency} с")
    print(f"  Результаты сохранены в: {RESULTS_PATH}")


if __name__ == "__main__":
    run_accuracy_evaluation()