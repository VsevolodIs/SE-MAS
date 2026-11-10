import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with path.open('r', encoding='utf-8-sig') as handle:
        for lineno, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{lineno}: {exc}') from exc


def normalize_verdict(value):
    if value is True:
        return 'correct'
    if value is False:
        return 'incorrect'
    return 'unknown'


def detect_tool_error(response):
    solution = str(response.get('math_solution') or '')
    markers = (
        'не удалось распарсить', 'песочница вернула статус',
        'ошибка парсинга', 'could not parse', 'could not convert',
        'object is not iterable',
    )
    return any(marker in solution.lower() for marker in markers)


def main():
    parser = argparse.ArgumentParser(description='Score saved check_solution integration results')
    parser.add_argument('--run', type=Path, required=True, help='Directory containing raw_api_responses.jsonl')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'benchmarks' / 'dataset.jsonl')
    args = parser.parse_args()

    raw_file = args.run / 'raw_api_responses.jsonl'
    if not raw_file.is_file():
        parser.error(f'File not found: {raw_file}')
    if not args.dataset.is_file():
        parser.error(f'Dataset not found: {args.dataset}')

    dataset = {row['id']: row for row in read_jsonl(args.dataset)}
    rows = []
    for item in read_jsonl(raw_file):
        test_id = item['id']
        expected = item.get('expected_checks') or dataset.get(test_id, {}).get('checks', {})
        response = item.get('response') or {}
        http_ok = item.get('http_status') == 200
        predicted = normalize_verdict(response.get('is_correct')) if http_ok else 'unknown'
        target = expected.get('verdict', 'unknown')
        verdict_pass = predicted == target if predicted != 'unknown' and target != 'unknown' else None
        error = detect_tool_error(response) if http_ok else False
        rows.append({
            'id': test_id,
            'http_ok': http_ok,
            'expected_verdict': target,
            'actual_verdict': predicted,
            'verdict_pass': verdict_pass,
            'tool_error': error,
            'semantic_review': 'required',
            'expected_checks': json.dumps(expected, ensure_ascii=False),
            'math_solution': response.get('math_solution', ''),
            'feedback': response.get('feedback', ''),
            'latency_sec': response.get('latency_sec', ''),
        })

    if not rows:
        parser.error('No API responses in the file')

    out_csv = args.run / 'semantic_evaluation.csv'
    fields = list(rows[0].keys())
    with out_csv.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'responses': len(rows),
        'http_ok': sum(row['http_ok'] for row in rows),
        'verdict_pass': sum(row['verdict_pass'] is True for row in rows),
        'verdict_fail': sum(row['verdict_pass'] is False for row in rows),
        'verdict_unknown': sum(row['verdict_pass'] is None for row in rows),
        'tool_errors': sum(row['tool_error'] for row in rows),
        'full_semantic_pass': 'not automatically evaluated; manual review required',
    }
    out_json = args.run / 'semantic_summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('RESULTS')
    for row in rows:
        status = ('PASS' if row['verdict_pass'] else 'FAIL') if row['verdict_pass'] is not None else 'UNKNOWN'
        suffix = ' | tool_error' if row['tool_error'] else ''
        print(f"{row['id']}: verdict {status}{suffix} | feedback: review required")
    print('SUMMARY', json.dumps(summary, ensure_ascii=False))
    print('Saved:', out_csv)
    print('Saved:', out_json)


if __name__ == '__main__':
    main()
