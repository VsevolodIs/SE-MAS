"""TutorFlow AI: reproducible Ollama inference benchmark (quality scored separately)."""
import argparse
import csv
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    'Ты помощник репетитора по математике 5–9 классов. '
    'Отвечай по-русски, точно и понятно. Не выдумывай данные об ученике. '
    'Если информации не хватает, уточни. Если не уверен в решении, сообщи об этом. '
    'Выводи только ответ пользователю, без черновых рассуждений.'
)


def format_prompt(record):
    data = record['input']
    parts = [data['prompt']]
    if data.get('student_profile') is not None:
        parts.append('Профиль ученика:\n' + json.dumps(data['student_profile'], ensure_ascii=False))
    if data.get('context') is not None:
        parts.append('Дополнительный контекст:\n' + json.dumps(data['context'], ensure_ascii=False))
    return '\n\n'.join(parts)


def stream_generate(base_url, model, prompt, timeout, seed, num_predict, qwen_no_think=False, warmup=False):
    request_prompt = ('/no_think\n' + prompt) if qwen_no_think and 'qwen3' in model.lower() else prompt
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request_prompt}],
        'stream': True,
        'think': False,
        'options': {'temperature': 0, 'seed': seed, 'num_predict': num_predict},
        'keep_alive': '10m',
    }
    req = Request(
        base_url.rstrip('/') + '/api/chat',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json; charset=utf-8'},
        method='POST',
    )
    start = time.perf_counter()
    pieces, thinking_pieces = [], []
    first_token_s = None
    last = {}
    with urlopen(req, timeout=timeout) as response:
        for raw in response:
            if not raw.strip():
                continue
            event = json.loads(raw.decode('utf-8'))
            if event.get('error'):
                raise RuntimeError(event['error'])
            msg = event.get('message') or {}
            token = msg.get('content') or ''
            thought = msg.get('thinking') or ''
            if token or thought:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                if not warmup:
                    pieces.append(token)
                    thinking_pieces.append(thought)
            last = event
    wall_s = time.perf_counter() - start
    if not last.get('done'):
        raise RuntimeError('Stream ended without done=true')
    answer = ''.join(pieces)
    thinking = ''.join(thinking_pieces)
    leaked_thinking = bool(re.search(r'</?think>', answer, flags=re.IGNORECASE))
    user_facing_answer = re.split(r'</think>', answer, flags=re.IGNORECASE)[-1].strip() if leaked_thinking and '</think>' in answer.lower() else answer.strip()
    eval_dur = last.get('eval_duration') or 0
    eval_count = last.get('eval_count') or 0
    return {
        'answer': answer,
        'user_facing_answer': user_facing_answer,
        'thinking_chars': len(thinking),
        'thinking_in_answer': leaked_thinking,
        'ttft_s': round(first_token_s, 4) if first_token_s is not None else None,
        'wall_s': round(wall_s, 4),
        'tps': round(eval_count * 1e9 / eval_dur, 3) if eval_dur else None,
        'output_tokens': eval_count,
        'input_tokens': last.get('prompt_eval_count'),
        'load_s': round((last.get('load_duration') or 0) / 1e9, 4),
        'done_reason': last.get('done_reason'),
        'truncated': last.get('done_reason') == 'length',
    }


def main():
    p = argparse.ArgumentParser(description='Benchmark local Ollama models; do not score correctness.')
    p.add_argument('--models', nargs='+', required=True, help='Already downloaded Ollama model tags')
    p.add_argument('--dataset', type=Path, default=ROOT / 'benchmarks' / 'dataset.jsonl')
    p.add_argument('--out', type=Path, default=ROOT / 'results', help='Output DIRECTORY, not a filename')
    p.add_argument('--limit', type=int, default=0, help='Number of first dataset rows; 0 means all')
    p.add_argument('--timeout', type=int, default=600, help='Seconds per request')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-predict', type=int, default=1024, help='Maximum generated tokens')
    p.add_argument('--qwen-no-think', action='store_true', help='Prefix /no_think on Qwen3 requests only')
    p.add_argument('--base-url', default='http://localhost:11434')
    args = p.parse_args()
    if args.limit < 0 or args.num_predict <= 0 or args.timeout <= 0:
        p.error('--limit must be >= 0; --num-predict and --timeout must be > 0')
    if args.out.exists() and not args.out.is_dir():
        p.error('--out must be a directory')
    records = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw_path = args.out / f'ollama_{stamp}.jsonl'
    csv_path = args.out / f'ollama_{stamp}.csv'
    fields = ['id', 'model', 'task_type', 'grade', 'status', 'ttft_s', 'tps', 'wall_s',
              'input_tokens', 'output_tokens', 'load_s', 'done_reason', 'truncated',
              'thinking_chars', 'thinking_in_answer', 'error']
    summary = []
    for model in args.models:
        print(f'\nMODEL {model} — warm-up', flush=True)
        try:
            stream_generate(args.base_url, model, 'Ответь одним словом: готово.',
                            args.timeout, args.seed, 64, args.qwen_no_think, warmup=True)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f'Warm-up failed: {exc}. Will attempt cases anyway.', flush=True)
        for index, record in enumerate(records, 1):
            item = {'id': record['id'], 'model': model,
                    'task_type': record['task_type'], 'grade': record.get('grade'),
                    'status': 'ok', 'error': None, 'prompt': format_prompt(record), 'answer': None}
            try:
                item.update(stream_generate(args.base_url, model, item['prompt'],
                                            args.timeout, args.seed, args.num_predict,
                                            args.qwen_no_think))
                if item['truncated']:
                    item['status'] = 'truncated'
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                item['status'] = 'error'
                item['error'] = str(exc)
            summary.append(item)
            with raw_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'[{index}/{len(records)}] {record["id"]}: {item["status"]} | '
                  f'TTFT={item.get("ttft_s")}s | TPS={item.get("tps")} | '
                  f'thinking_in_answer={item.get("thinking_in_answer")}', flush=True)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        'timestamp_utc': stamp, 'models': args.models, 'samples_per_model': len(records),
        'seed': args.seed, 'temperature': 0, 'num_predict': args.num_predict,
        'think': False, 'qwen_no_think': args.qwen_no_think,
        'api': '/api/chat', 'python': platform.python_version(),
        'platform': platform.platform(), 'base_url': args.base_url,
        'dataset': str(args.dataset),
        'note': 'Quality not scored. TTFT=first nonempty message chunk (content or thinking). '
                'TPS=Ollama eval_count/eval_duration. Warm-up excluded. '
                'Raw answer retained; user_facing_answer is diagnostic only. '
                'Use same flags/hardware for controlled comparisons.',
    }
    (args.out / f'ollama_{stamp}_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved: {raw_path}\nSaved: {csv_path}', flush=True)
    print('Reminder: correctness, RAM, and VRAM require separate evaluation.', flush=True)


if __name__ == '__main__':
    main()
