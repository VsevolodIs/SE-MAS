"""Run benchmark dataset cases against the *real* TutorFlow LangGraph.

Usage (from repo root):
    python scripts/run_graph_dataset.py --limit 3
    python scripts/run_graph_dataset.py

Outputs go to results/graph_dataset/<timestamp>_cases.jsonl and _summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def choose(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        val = obj.get(key)
        if val is not None and val != "":
            return val
    return None


def get_prompt(case: dict[str, Any]) -> str:
    val = choose(case, "user_prompt", "prompt", "query", "input", "user_input", "request", "text", "question", "instruction")
    if isinstance(val, dict):
        val = choose(val, "user_prompt", "prompt", "query", "text", "content")
    if isinstance(val, list):
        val = "\n".join(str(x.get("content", "")) if isinstance(x, dict) else str(x) for x in val)
    if isinstance(val, str) and val.strip():
        return val.strip()
    raise ValueError("Не найден текст запроса: проверьте ключи JSONL и функцию get_prompt()")


def normalize_expected(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = choose(raw, "task_type", "type", "name")
    if not isinstance(raw, str):
        return ""
    key = raw.lower().strip().replace("-", "_").replace(" ", "_")
    mapping = {
        "solve": "solve_math", "solve_math": "solve_math", "math": "solve_math",
        "check_solution": "solve_check", "solve_check": "solve_check", "check": "solve_check",
        "generate_exercises": "pedagogical", "exercise_generation": "pedagogical",
        "plan_lesson": "pedagogical", "lesson_plan": "pedagogical",
        "pedagogical": "pedagogical", "generate": "pedagogical",
        "personalize": "personalize", "personalization": "personalize",
        "multi_step": "multi_step", "multistep": "multi_step",
        "clarify": "clarify", "clarification": "clarify",
        "safety": "safety",
    }
    return mapping.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/dataset.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = все кейсы")
    parser.add_argument("--start", type=int, default=1, help="Номер первого кейса, начиная с 1")
    parser.add_argument("--out-dir", default="results/graph_dataset")
    args = parser.parse_args()

    from src.agents.graph import build_tutor_graph

    dataset = (ROOT / args.dataset).resolve()
    if not dataset.is_file():
        raise SystemExit(f"Датасет не найден: {dataset}")
    cases = []
    with dataset.open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Ожидался JSON-объект")
                cases.append((lineno, obj))
            except Exception as exc:
                raise SystemExit(f"Некорректная строка JSONL {lineno}: {exc}") from exc

    selected = cases[max(0, args.start - 1):]
    if args.limit > 0:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("Нет кейсов для запуска")

    outdir = ROOT / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_file = outdir / f"{stamp}_cases.jsonl"
    csv_file = outdir / f"{stamp}_summary.csv"
    graph = build_tutor_graph()
    rows = []

    print(f"Датасет: {dataset} | всего: {len(cases)} | запускаем: {len(selected)}", flush=True)
    try:
        with jsonl_file.open("w", encoding="utf-8") as report:
            for idx, (lineno, case) in enumerate(selected, 1):
                case_id = str(choose(case, "id", "case_id", "test_id") or f"line_{lineno}")
                category = str(choose(case, "category", "scenario", "task_type", "type", "intent") or "")
                expected = normalize_expected(choose(case, "task_type", "category", "scenario", "type", "intent"))
                start = time.perf_counter()
                try:
                    prompt = get_prompt(case)
                    state = {"user_prompt": prompt, "session_id": f"benchmark_{stamp}_{idx}", "student_name": "Тестовый ученик"}
                    for key in ("grade", "topic", "student_level", "weak_topics", "student_answer", "task_condition"):
                        if key in case and case[key] is not None:
                            state[key] = case[key]
                    result = graph.invoke(state)
                    actual = str(result.get("task_type") or "")
                    status = str(result.get("validation_status") or "")
                    output = result.get("final_output")
                    row = {"id": case_id, "line": lineno, "category": category, "expected_route": expected,
                           "actual_route": actual, "route_match": (expected == actual if expected else None),
                           "validation_status": status, "is_correct": result.get("is_correct"),
                           "elapsed_s": round(time.perf_counter() - start, 2), "error": "", "prompt": prompt,
                           "final_output": output if isinstance(output, str) else str(output or "")}
                except Exception as exc:
                    row = {"id": case_id, "line": lineno, "category": category, "expected_route": expected,
                           "actual_route": "", "route_match": None, "validation_status": "",
                           "is_correct": None, "elapsed_s": round(time.perf_counter() - start, 2),
                           "error": f"{type(exc).__name__}: {exc}", "prompt": str(choose(case, "user_prompt", "prompt", "query", "input", "text") or ""),
                           "final_output": ""}
                rows.append(row)
                report.write(json.dumps({"case": case, **row}, ensure_ascii=False, default=str) + "\n")
                report.flush()
                mark = "ERROR" if row["error"] else ("ROUTE?" if row["route_match"] is False else "DONE")
                print(f"[{idx}/{len(selected)}] {case_id} {mark} route={row['actual_route'] or '-'} "
                      f"status={row['validation_status'] or '-'} {row['elapsed_s']}s", flush=True)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем. Завершённые кейсы сохранены.", flush=True)
    finally:
        if rows:
            columns = ["id", "line", "category", "expected_route", "actual_route", "route_match",
                       "validation_status", "is_correct", "elapsed_s", "error", "prompt", "final_output"]
            with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            print("\nИТОГ:", len(rows), "кейсов, ошибок выполнения:", sum(bool(r["error"]) for r in rows))
            print("Маршруты:", dict(Counter(r["actual_route"] or "ERROR" for r in rows)))
            print("Статусы:", dict(Counter(r["validation_status"] or "NO_STATUS" for r in rows)))
            print("Несовпадений маршрута:", sum(r["route_match"] is False for r in rows))
            print("Отчёт JSONL:", jsonl_file)
            print("Отчёт CSV:", csv_file)
            print("ВАЖНО: APPROVED не доказывает математическую корректность ответа.")


if __name__ == "__main__":
    main()
