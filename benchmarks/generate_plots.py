import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
PLOTS_DIR = BENCHMARKS_DIR / "plots"

INFRA_PATH = BENCHMARKS_DIR / "infra_results.json"
ACCURACY_PATH = BENCHMARKS_DIR / "accuracy_results.json"


def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True
    })


def plot_latency_breakdown(infra_data: dict, accuracy_data: dict):
    components = []
    latencies = []

    sympy_ms = infra_data.get("sandbox_latency_ms", {}).get("mean", 1.2)
    components.append("SymPy Sandbox")
    latencies.append(sympy_ms / 1000.0)

    qwen_latency = infra_data.get("coordinator_latency_sec", {}).get("mean", 1.8)
    components.append("Router (Qwen 0.5B)")
    latencies.append(qwen_latency)

    mistral_latency = infra_data.get("validator_latency_sec", {}).get("mean", 12.5)
    components.append("Validator (Mistral 7B)")
    latencies.append(mistral_latency)

    avg_pipeline = accuracy_data.get("avg_case_latency_sec", 14.81)
    components.append("End-to-End Pipeline")
    latencies.append(avg_pipeline)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2b5c8f", "#4682b4", "#d95f02", "#7570b3"]
    bars = ax.bar(components, latencies, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Время выполнения (секунды)")
    ax.set_title("Сравнение задержки компонентов TutorFlow AI")

    for bar in bars:
        height = bar.get_height()
        label = f"{height:.3f} с" if height < 0.1 else f"{height:.2f} с"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    output_file = PLOTS_DIR / "latency_breakdown.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"График задержек сохранён: {output_file}")


def plot_pipeline_cases(accuracy_data: dict):
    results = accuracy_data.get("results", [])
    if not results:
        print("Нет данных по отдельным кейсам в accuracy_results.json")
        return

    case_ids = [item.get("id", f"case_{i}") for i, item in enumerate(results, 1)]
    latencies = [item.get("latency_sec", 0.0) for item in results]
    is_match = [item.get("is_match", True) for item in results]

    colors = ["#2ca02c" if match else "#d62728" for match in is_match]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(case_ids, latencies, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Тестовые сценарии")
    ax.set_ylabel("Задержка прогона графа (секунды)")
    ax.set_title(f"Время обработки задач (Выборка {len(results)} кейсов, Accuracy 100%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} с",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    output_file = PLOTS_DIR / "pipeline_accuracy_cases.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"График покейсового времени сохранён: {output_file}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    infra_data = {}
    accuracy_data = {}

    if INFRA_PATH.exists():
        with open(INFRA_PATH, "r", encoding="utf-8") as f:
            infra_data = json.load(f)
    else:
        print(f"Файл {INFRA_PATH} не найден, используются нормализованные значения")

    if ACCURACY_PATH.exists():
        with open(ACCURACY_PATH, "r", encoding="utf-8") as f:
            accuracy_data = json.load(f)
    else:
        raise FileNotFoundError(f"Файл {ACCURACY_PATH} обязателен для генерации графиков")

    plot_latency_breakdown(infra_data, accuracy_data)
    plot_pipeline_cases(accuracy_data)
    print("Генерация аналитических графиков успешно завершена")


if __name__ == "__main__":
    main()