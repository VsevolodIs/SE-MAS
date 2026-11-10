"""Offline route coverage audit for the official 42-case dataset."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.graph import coordinator_node
MAPPING = {
    "solve": "solve_math", "generate_exercises": "pedagogical",
    "plan_lesson": "pedagogical", "check_solution": "solve_check",
    "personalize": "personalize", "multi_step": "multi_step",
    "clarify": "clarify", "safety": "safety",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/dataset.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/route_audit.csv")
    args = parser.parse_args()
    rows = []
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        expected = MAPPING.get(case.get("task_type"))
        prompt = case["input"]["prompt"]
        actual = coordinator_node({"user_prompt": prompt}).get("task_type")
        rows.append({"id": case["id"], "dataset_type": case.get("task_type"),
                     "expected_route": expected or "UNMAPPED", "actual_route": actual,
                     "match": expected == actual, "prompt": prompt})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    count = Counter("match" if r["match"] else "mismatch" for r in rows)
    print(f"Cases: {len(rows)} | matched: {count['match']} | mismatched: {count['mismatch']}")
    for row in rows:
        if not row["match"]:
            print(row["id"], row["expected_route"], "->", row["actual_route"])
    print("CSV:", args.out)

if __name__ == "__main__":
    main()
