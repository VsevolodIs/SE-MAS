import json
import sys
import time
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import build_tutor_graph
from src.tools.sandbox import run_sandboxed_math

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = BENCHMARK_DIR / "infra_results.json"


def benchmark_ollama_model(model: str, prompt: str, runs: int = 3) -> dict:
    client = httpx.Client(base_url="http://127.0.0.1:11434", timeout=120.0, trust_env=False)
    latencies = []

    for _ in range(runs):
        start = time.perf_counter()
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50},
            },
        )
        response.raise_for_status()
        latencies.append(time.perf_counter() - start)

    return {
        "model": model,
        "runs": runs,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
        "min_latency_sec": round(min(latencies), 3),
        "max_latency_sec": round(max(latencies), 3),
    }


def benchmark_sympy_sandbox(equation: str, runs: int = 5) -> dict:
    latencies = []
    status = "unknown"

    for _ in range(runs):
        start = time.perf_counter()
        res = run_sandboxed_math(equation, variable="x", timeout_sec=3.0)
        latencies.append(time.perf_counter() - start)
        status = res.get("status", "unknown")

    return {
        "equation": equation,
        "status": status,
        "runs": runs,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 5),
    }


def benchmark_end_to_end_graph(app, test_prompt: str) -> dict:
    start = time.perf_counter()
    nodes_executed = []

    for output in app.stream({"user_prompt": test_prompt}):
        for node_name in output.keys():
            nodes_executed.append(node_name)

    total_latency = time.perf_counter() - start
    return {
        "test_prompt": test_prompt,
        "total_latency_sec": round(total_latency, 3),
        "nodes_flow": " -> ".join(nodes_executed),
    }


def run_all_benchmarks():
    print("=" * 60)
    print("Запуск инфраструктурных бенчмарков TutorFlow AI...")
    print("=" * 60)

    results = {}

    print("\n1. Тестирование производительности SymPy Sandbox...")
    sympy_bench = benchmark_sympy_sandbox("2*x + 4 = 10")
    print(f"  Среднее время расчета: {sympy_bench['avg_latency_sec']} c (Статус: {sympy_bench['status']})")
    results["sympy_sandbox"] = sympy_bench

    print("\n2. Тестирование задержек инференса моделей Ollama...")
    prompt = "Реши уравнение 2*x + 4 = 10"

    print("  -> Оценка легковесной qwen2.5:0.5b...")
    qwen_bench = benchmark_ollama_model("qwen2.5:0.5b", prompt, runs=2)
    print(f"     Avg Latency: {qwen_bench['avg_latency_sec']} c")

    print("  -> Оценка тяжёлой mistral:7b...")
    mistral_bench = benchmark_ollama_model("mistral:7b", prompt, runs=2)
    print(f"     Avg Latency: {mistral_bench['avg_latency_sec']} c")

    results["models_latency"] = {
        "qwen2.5:0.5b": qwen_bench,
        "mistral:7b": mistral_bench,
    }

    print("\n3. Тестирование сквозной задержки LangGraph Pipeline...")
    app = build_tutor_graph()
    e2e_bench = benchmark_end_to_end_graph(
        app, "Проверь решение Ивана по задаче 2*x + 4 = 10. Ученик ответил x = 3."
    )
    print(f"  Цепочка узлов: {e2e_bench['nodes_flow']}")
    print(f"  Полное время выполнения: {e2e_bench['total_latency_sec']} c")
    results["end_to_end_pipeline"] = e2e_bench

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Бенчмарк успешно завершен, результаты сохранены в:\n{RESULTS_PATH}")


if __name__ == "__main__":
    run_all_benchmarks()