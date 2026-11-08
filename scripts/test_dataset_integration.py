import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {"check_solution": "/api/v1/check-solution"}
FIELDS = ["id", "task_type", "endpoint", "status", "http_status", "elapsed_s", "is_correct", "feedback", "math_solution", "error"]


def request_json(url, method="GET", payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, {"error": body}


def load_dataset(path):
    rows = []
    ids = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("input"), dict) or not isinstance(item["input"].get("prompt"), str):
            raise ValueError(f"Invalid record on line {number}")
        if item.get("id") in ids:
            raise ValueError(f"Duplicate test ID on line {number}: {item.get('id')}")
        ids.add(item.get("id"))
        rows.append(item)
    return rows


def main():
    p = argparse.ArgumentParser(description="Audit SE-MAS coverage and optionally run real check-solution cases against FastAPI")
    p.add_argument("--dataset", type=Path, default=ROOT / "benchmarks" / "dataset.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "integration")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--live", action="store_true", help="Call the real API (may trigger LLM inference and DB writes)")
    p.add_argument("--limit", type=int, default=0, help="Max supported real API calls; 0=all supported")
    p.add_argument("--timeout", type=int, default=240)
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit must not be negative")
    dataset = load_dataset(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    output = []
    attempted = 0
    base = args.base_url.rstrip("/")
    if args.live:
        try:
            status, health = request_json(base + "/health", timeout=10)
            print(f"API /health: HTTP {status} | {health}")
        except (URLError, TimeoutError, OSError) as exc:
            p.error(f"API not available: {exc}. Run uvicorn first.")
    for record in dataset:
        task = record.get("task_type", "unknown")
        endpoint = SUPPORTED.get(task)
        item = {"id": record.get("id"), "task_type": task, "endpoint": endpoint or "", "status": "unsupported", "http_status": "", "elapsed_s": "", "is_correct": "", "feedback": "", "math_solution": "", "error": ""}
        if endpoint:
            item["status"] = "ready_for_api"
            if args.live and (args.limit == 0 or attempted < args.limit):
                attempted += 1
                t0 = time.perf_counter()
                try:
                    http_status, response = request_json(base + endpoint, "POST", {"prompt": record["input"]["prompt"]}, args.timeout)
                    item["elapsed_s"] = round(time.perf_counter() - t0, 3)
                    item["http_status"] = http_status
                    item["status"] = "api_ok" if http_status == 200 else "api_error"
                    if http_status == 200:
                        item["is_correct"] = response.get("is_correct")
                        item["math_solution"] = response.get("math_solution", "")
                        item["feedback"] = response.get("feedback", "")
                    else:
                        item["error"] = str(response.get("error", response))[:500]
                except (URLError, TimeoutError, OSError, ValueError) as exc:
                    item["status"] = "api_error"
                    item["elapsed_s"] = round(time.perf_counter() - t0, 3)
                    item["error"] = str(exc)[:500]
        output.append(item)
        print(f"{item['id']}: {item['task_type']} -> {item['status']}", flush=True)

    with (args.out / "integration_results.jsonl").open("w", encoding="utf-8") as file:
        for item in output:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.out / "integration_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    summary = {"dataset_count": len(dataset), "task_types": dict(Counter(r.get("task_type", "unknown") for r in dataset)), "statuses": dict(Counter(r["status"] for r in output)), "live": args.live, "api_calls": attempted, "note": "api_ok means HTTP 200, NOT semantic correctness. Human/automatic scoring remains separate."}
    (args.out / "integration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
