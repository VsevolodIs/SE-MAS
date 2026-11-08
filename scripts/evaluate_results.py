"""TutorFlow A03: reproducible technical metrics + conservative mathematics checks.

Requires: pip install sympy
No LLM judge, no API calls. This program never labels creative tasks as automatically correct.
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    import sympy as sp
except ImportError:
    raise SystemExit('Missing dependency. Run: py -m pip install sympy')

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {'solve': 'math', 'check_solution': 'solution_check',
              'generate_exercises': 'generation', 'plan_lesson': 'lesson',
              'personalize': 'personalization', 'multi_step': 'multi_step',
              'clarify': 'ambiguity', 'safety': 'safety'}


def read_jsonl(path):
    with open(path, encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]


def pick_files(folder):
    paths = []
    for name in ('qwen3_full', 'gemma3_full', 'llama3_full'):
        matches = sorted((folder / name).glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
        if not matches:
            raise SystemExit(f'Missing full benchmark JSONL under {folder / name}')
        paths.append(matches[-1])
    return paths


def user_answer(row):
    # The benchmark writes this field for Qwen's post-thinking content.
    # Retain the raw answer elsewhere for audit.
    value = row.get('user_facing_answer')
    return value if isinstance(value, str) and value.strip() else (row.get('answer') or '')


def terminal_text(answer):
    # Try an explicit answer label before falling back to last line.
    matches = list(re.finditer(r'(?:ответ\s*[:：]|\*\*ответ\*\*\s*[:：]?)', answer, re.I))
    if matches:
        return answer[matches[-1].end():].strip()[:300]
    chunks = [line.strip() for line in answer.splitlines() if line.strip()]
    return '\n'.join(chunks[-3:])[-450:] if chunks else ''


def numbers(text):
    text = text.replace('−', '-').replace('–', '-').replace(',', '.')
    # Fractions and decimals, including negative numbers, outside words.
    values = []
    for match in re.finditer(r'(?<![\w])(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)(?![\w])', text):
        try:
            val = sp.sympify(match.group(1).replace(' ', ''), evaluate=True)
            if bool(val.is_number):
                values.append(val)
        except (ValueError, TypeError, SyntaxError, sp.SympifyError):
            pass
    return values


def final_value(text):
    vals = numbers(terminal_text(text))
    return vals[-1] if vals else None


def eq(a, b):
    return a is not None and sp.simplify(a - sp.sympify(b)) == 0


def variable_assignments(text):
    # Match x = 3, x₁ = 2, х = 4, and y = 3. Works for Cyrillic х as well.
    text = text.replace('−', '-').replace('₁', '1').replace('₂', '2')
    out = defaultdict(list)
    for m in re.finditer(r'(?<![\w])([xyх])\s*[12]?\s*=\s*(-?\d+(?:[.,]\d+)?)', text, re.I):
        var = 'x' if m.group(1).lower() == 'х' else m.group(1).lower()
        out[var].append(sp.sympify(m.group(2).replace(',', '.')))
    return out


def objective_check(record, answer, status):
    """Review *candidates* for 9 simple math tasks, not a complete proof.

    Other tasks need mathematical step validation / a human-written rubric.
    A candidate pass does not certify all intermediate steps or safe handling.
    """
    if status != 'ok':
        return 'not_evaluated', 'generation not complete'
    case = record['id']
    if case == 'TF-006':
        return 'manual_review', 'symbolic simplification requires domain condition a != 0'
    if record['task_type'] != 'solve':
        return 'manual_review', 'rubric / semantic validation required'
    checks = record.get('checks', {})
    target = checks.get('exact_numeric', checks.get('exact_answer'))
    if target is not None:
        val = final_value(answer)
        if val is None:
            return 'manual_review', 'no unambiguous terminal numeric result'
        good = eq(val, target)
        return ('candidate_pass' if good else 'candidate_fail', f'last numeric value={val}; expected={target}')
    if 'equivalent_expression' in checks:
        target = checks['equivalent_expression']
        if target == '17/12':
            val = final_value(answer)
            if val is None:
                return 'manual_review', 'no unambiguous terminal fraction'
            return ('candidate_pass' if eq(val, sp.Rational(17, 12)) else 'candidate_fail', f'last value={val}; expected=17/12')
        if target == 'x=3':
            xs = variable_assignments(terminal_text(answer)).get('x', [])
            if not xs:
                return 'manual_review', 'no terminal x assignment'
            return ('candidate_pass' if eq(xs[-1], 3) else 'candidate_fail', f'x={xs[-1]}; expected=3')
        return 'manual_review', 'unsupported expression checker'
    if 'root_set' in checks:
        xs = variable_assignments(terminal_text(answer)).get('x', [])
        expected = {sp.sympify(v) for v in checks['root_set']}
        if not xs:
            return 'manual_review', 'no terminal root set'
        observed = set(xs)
        return ('candidate_pass' if observed == expected else 'candidate_fail', f'roots={sorted(map(str,observed))}, expected={sorted(map(str,expected))}')
    if 'variables' in checks:
        observed = variable_assignments(terminal_text(answer))
        if not all(observed.get(v) for v in checks['variables']):
            return 'manual_review', 'missing terminal assignments'
        good = all(eq(observed[v][-1], val) for v, val in checks['variables'].items())
        return ('candidate_pass' if good else 'candidate_fail', 'terminal x/y assignments checked')
    return 'manual_review', 'no supported deterministic checker'


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'benchmarks'/'dataset.jsonl')
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--results', type=Path, nargs='+', help='Explicit JSONL files (overrides default full-run folders)')
    parser.add_argument('--out', type=Path, default=ROOT/'results'/'evaluation')
    args = parser.parse_args()
    tests = read_jsonl(args.dataset)
    reference = {r['id']: r for r in tests}
    if len(reference) != len(tests):
        raise SystemExit('Duplicate IDs in dataset')
    sources = args.results or pick_files(args.results_dir)
    rows = []
    models = set()
    for src in sources:
        results = read_jsonl(src)
        for run in results:
            task_id = run.get('id')
            if task_id not in reference:
                raise SystemExit(f'Unknown test ID {task_id} in {src}')
            model = run.get('model', '')
            key = (model, task_id)
            if key in models:
                raise SystemExit(f'Duplicate result {key}; select only one run per model')
            models.add(key)
            record = reference[task_id]
            answer = user_answer(run)
            status = run.get('status', 'unknown')
            if status == 'ok' and run.get('done_reason') == 'length':
                status = 'truncated'
            auto, why = objective_check(record, answer, status)
            rows.append({
                'id': task_id, 'model': model, 'category': CATEGORIES.get(record['task_type'],record['task_type']),
                'task_type': record['task_type'], 'generation_status': status,
                'done_reason': run.get('done_reason'), 'thinking_in_answer': run.get('thinking_in_answer', ''),
                'ttft_s': run.get('ttft_s'), 'tps': run.get('tps'), 'wall_s': run.get('wall_s'),
                'automatic_result': auto, 'automatic_reason': why,
                'human_verdict': '', 'human_comment': '',
                'expected_behavior': record.get('expected_behavior',''), 'answer': answer,
                'source_file': str(src)
            })
    args.out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: (x['model'],x['id']))
    details = args.out/'evaluation_details.csv'
    write_csv(details, rows, list(rows[0]) if rows else [])
    summary = []
    for model in sorted({r['model'] for r in rows}):
        subset = [r for r in rows if r['model'] == model]
        complete = [r for r in subset if r['generation_status'] == 'ok']
        mean = lambda k: round(statistics.mean(float(r[k]) for r in subset if r[k] is not None), 3) if any(r[k] is not None for r in subset) else None
        summary.append({'model': model, 'total': len(subset), 'completed': len(complete),
                        'truncated': sum(r['generation_status']=='truncated' for r in subset),
                        'errors': sum(r['generation_status'] not in ('ok','truncated') for r in subset),
                        'completion_percent': round(100*len(complete)/len(subset), 2) if subset else 0,
                        'avg_ttft_s': mean('ttft_s'), 'avg_tps': mean('tps'), 'avg_wall_s': mean('wall_s'),
                        'candidate_pass': sum(r['automatic_result']=='candidate_pass' for r in subset),
                        'candidate_fail': sum(r['automatic_result']=='candidate_fail' for r in subset),
                        'manual_or_unrated': sum(r['automatic_result'] in ('manual_review','not_evaluated') for r in subset)})
    report = args.out/'evaluation_summary.csv'
    write_csv(report, summary, list(summary[0]) if summary else [])
    readme = args.out/'README.md'
    readme.write_text('''# TutorFlow AI — A03 evaluation

This script produces **technical benchmark summaries** and **conservative objective-check candidates**.

- `evaluation_summary.csv`: completed/truncated counts, TTFT/TPS, plus counts of candidate checks.
- `evaluation_details.csv`: all answers and preliminary automatic flags; fill `human_verdict` with `pass`, `partial`, or `fail`, and explain in `human_comment` after manual checking.
- `candidate_pass` is **not** verified mathematical accuracy: it only checks selected final values, not the validity of each step.
- `candidate_fail` flags a likely wrong final answer; confirm manually when formatting is ambiguous.
- `manual_review` means no safe automated verifier exists for this test yet.
- `not_evaluated` means the generation ended prematurely or failed.
- The tool does **not** compute an overall educational quality percentage; a separate, human-reviewed rubric is needed.
- SymPy is used for deterministic comparisons of numbers, rational expressions, and root values in a narrow whitelist of straightforward tasks. It does not validate all algebraic steps.
- These results do not measure RAM or VRAM.

To add deeper automatic validation, develop a dedicated verifier for each task type and cross-check its decisions against independently hand-reviewed examples.
''', encoding='utf-8')
    print(f'Processed {len(rows)} answers from {len(sources)} run files.')
    for s in summary:
        print(f"{s['model']}: completed {s['completed']}/{s['total']}; truncated {s['truncated']}; "
              f"TTFT={s['avg_ttft_s']}s; TPS={s['avg_tps']}; "
              f"candidates pass/fail={s['candidate_pass']}/{s['candidate_fail']}")
    print(f'Saved: {details}\nSaved: {report}\nSaved: {readme}')

if __name__ == '__main__':
    main()