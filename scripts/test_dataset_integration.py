"""Check dataset/API coverage, optionally exercising real SE-MAS agents.

Examples (from the repository root):
    py scripts/test_dataset_integration.py
    py scripts/test_dataset_integration.py --live --id TF-020 --limit 1
    py scripts/test_dataset_integration.py --live --limit 7

HTTP 200 does not mean the agent's mathematical judgment is correct.
"""

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {"check_solution": "/api/v1/check-solution"}
FIELDS = [
    "id", "task_type", "endpoint", "status", "http_status", "elapsed_s",
    "is_correct", "feedback", "math_solution", "error",
]

# Explicit, reviewed fixtures for the seven check_solution tasks.
# The currently running API expects {"prompt": ...}, not {"problem", "student_solution"}.
# Review fixture wording against benchmarks/dataset.jsonl before interpreting results.
CHECK_SOLUTION_PAYLOADS = {
    "TF-020": {
        "problem": "2(x+3)=14",
        "student_solution": "2x+3=14; 2x=11; x=5,5",
    },
    "TF-021": {
        "problem": "x^2 - 9 = 0",
        "student_solution": "(x-3)(x+3)=0; x=3 или x=-3",
    },
    "TF-022": {
        "problem": "(x^2 - 4)/(x - 2) = 0",
        "student_solution": "x=2 или x=-2",
    },
    "TF-023": {
        "problem": "Найди 20% от 250",
        "student_solution": "20% от 250 — это 50",
    },
    "TF-024": {
        "problem": "x^2 - 4 < 0",
        "student_solution": "x < -2 или x > 2",
    },
    "TF-025": {
        "problem": "Два числа в сумме дают 25, одно на 7 больше другого.",
        "student_solution": "Числа 16 и 9",
    },
    "TF-026": {
        "problem": "Существует ли треугольник со сторонами 3, 4 и 8?",
        "student_solution": "Да, третья сторона может быть 8",
    },
}


def request_json(url, method="GET", payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(body)
        except ValueError:
            decoded = {"error": body}
        return exc.code, decoded


def load_dataset(path):
    rows = []
    ids = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not isinstance(item.get("input"), dict)
            or not isinstance(item["input"].get("prompt"), str)
        ):
            raise ValueError(f"Invalid record on line {number}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate test ID on line {number}: {item['id']}")
        ids.add(item["id"])
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Audit SE-MAS dataset coverage and optionally call the real check-solution API"
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks" / "dataset.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "integration")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--live", action="store_true", help="Call real agents; may write to the API database")
    parser.add_argument("--id", dest="test_id", help="Run just this dataset case, e.g. TF-020")
    parser.add_argument("--limit", type=int, default=0, help="Maximum supported API calls (0 = all)")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    dataset = load_dataset(args.dataset)
    if args.test_id:
        dataset = [item for item in dataset if item["id"] == args.test_id]
        if not dataset:
            parser.error(f"Unknown test ID: {args.test_id}")

    # Never overwrite an earlier run (including historical integration reports).
    args.out.mkdir(parents=True, exist_ok=True)
    run_type = "live" if args.live else "audit"
    run_dir = args.out / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{run_type}"
    run_dir.mkdir(parents=True, exist_ok=False)

    results = []
    raw_responses = []
    attempted = 0
    base = args.base_url.rstrip("/")

    if args.live:
        try:
            status, health = request_json(base + "/health", timeout=10)
            print(f"API /health: HTTP {status} | {health}")
            if status != 200:
                parser.error("API health check did not return HTTP 200")
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            parser.error(f"API not available: {exc}. Start the FastAPI server first.")

    for record in dataset:
        task = record.get("task_type", "unknown")
        rec_id = record["id"]
        endpoint = SUPPORTED.get(task)
        item = {
            "id": rec_id,
            "task_type": task,
            "endpoint": endpoint or "",
            "status": "unsupported",
            "http_status": "",
            "elapsed_s": "",
            "is_correct": "",
            "feedback": "",
            "math_solution": "",
            "error": "",
        }
        if endpoint:
            fixture = CHECK_SOLUTION_PAYLOADS.get(rec_id)
            if fixture is None:
                item["status"] = "missing_payload"
                item["error"] = "No reviewed payload for this test ID; request not sent"
            else:
                # The API takes a single prompt. Keep the original condition distinct
                # from the student's working so the coordinator has less ambiguity.
                payload = {
                    "prompt": (
                        "Проверь решение ученика по математике.\n"
                        f"ИСХОДНОЕ УСЛОВИЕ: {fixture['problem']}\n"
                        f"РЕШЕНИЕ УЧЕНИКА: {fixture['student_solution']}\n"
                        "Исходным условием является ТОЛЬКО строка после «ИСХОДНОЕ УСЛОВИЕ». "
                        "Шаги ученика не являются исходным уравнением. "
                        "Определи правильность, проверь шаги и укажи первую ошибку, если она есть."
                    )
                }
                item["status"] = "ready_for_api"
                if args.live and (args.limit == 0 or attempted < args.limit):
                    attempted += 1
                    started = time.perf_counter()
                    try:
                        http_status, response = request_json(
                            base + endpoint, "POST", payload, args.timeout
                        )
                        item["elapsed_s"] = round(time.perf_counter() - started, 3)
                        item["http_status"] = http_status
                        item["status"] = "api_ok" if http_status == 200 else "api_error"
                        raw_responses.append({
                            "id": rec_id,
                            "request": payload,
                            "http_status": http_status,
                            "response": response,
                            "expected_checks": record.get("checks"),
                        })
                        if http_status == 200 and isinstance(response, dict):
                            item["is_correct"] = response.get("is_correct", "")
                            item["math_solution"] = response.get("math_solution", "")
                            item["feedback"] = response.get("feedback", "")
                        elif http_status != 200:
                            item["error"] = json.dumps(response, ensure_ascii=False)[:1000]
                    except (URLError, TimeoutError, OSError, ValueError) as exc:
                        item["status"] = "api_error"
                        item["elapsed_s"] = round(time.perf_counter() - started, 3)
                        item["error"] = str(exc)[:1000]
                        raw_responses.append({
                            "id": rec_id,
                            "request": payload,
                            "error": str(exc),
                            "expected_checks": record.get("checks"),
                        })
                elif args.live:
                    item["status"] = "not_run_limit"
        results.append(item)
        print(f"{item['id']}: {item['task_type']} -> {item['status']}", flush=True)

    with (run_dir / "integration_results.jsonl").open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (run_dir / "integration_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    if args.live:
        with (run_dir / "raw_api_responses.jsonl").open("w", encoding="utf-8") as file:
            for record in raw_responses:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset_count": len(dataset),
        "task_types": dict(Counter(r.get("task_type", "unknown") for r in dataset)),
        "statuses": dict(Counter(r["status"] for r in results)),
        "live": args.live,
        "api_calls": attempted,
        "note": "api_ok means HTTP 200, NOT semantic correctness. Review raw_api_responses.jsonl.",
    }
    (run_dir / "integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
