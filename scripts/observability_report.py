from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    if not path.is_file():
        return {"status": "no_events", "event_file": str(path), "events": 0,
                "message": "No telemetry recorded yet. Run a real API request first."}
    requests, spans, llms = [], [], []
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    malformed += 1
                    continue
            except (ValueError, TypeError):
                malformed += 1
                continue
            kind = event.get("kind")
            if kind == "node_span":
                spans.append(event)
            elif kind == "llm_metric":
                llms.append(event)
            elif "endpoint" in event:
                requests.append(event)
    latencies = [float(x["latency_sec"]) for x in requests
                 if isinstance(x.get("latency_sec"), (float, int)) and x["latency_sec"] >= 0]
    slow = sum(x.get("slow") is True or
               (isinstance(x.get("latency_sec"), (int, float)) and x["latency_sec"] > 60)
               for x in requests)
    failed_spans = sum(x.get("ok") is False for x in spans)
    statuses = Counter(str(x.get("validation_status") or "UNKNOWN") for x in requests)
    models = Counter(str(x.get("model") or "UNKNOWN") for x in llms)
    tps = [float(x["generation_tps"]) for x in llms
           if isinstance(x.get("generation_tps"), (int, float)) and x["generation_tps"] > 0]
    result = {
        "status": "ok", "event_file": str(path),
        "events": len(requests) + len(spans) + len(llms), "malformed_lines": malformed,
        "requests": {
            "total": len(requests), "by_endpoint": dict(Counter(str(x["endpoint"]) for x in requests)),
            "by_task_type": dict(Counter(str(x.get("task_type") or "UNKNOWN") for x in requests)),
            "validation_statuses": dict(statuses),
            "latency_avg_sec": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_max_sec": round(max(latencies), 3) if latencies else None,
            "slow_over_60_sec": slow,
        },
        "graph": {"node_spans": len(spans), "failed_spans": failed_spans,
                  "by_node": dict(Counter(str(x.get("node") or "UNKNOWN") for x in spans))},
        "llm": {"calls": len(llms), "by_model": dict(models),
                "mean_generation_tps": round(statistics.mean(tps), 3) if tps else None},
        "local_alerts": [],
    }
    if slow:
        result["local_alerts"].append(f"SLOW_REQUESTS: {slow} requests exceeded 60 sec")
    if failed_spans:
        result["local_alerts"].append(f"FAILED_NODES: {failed_spans} node executions failed")
    if malformed:
        result["local_alerts"].append(f"MALFORMED_EVENTS: {malformed} lines could not be parsed")
    return result


def markdown(r: dict) -> str:
    if r["status"] != "ok":
        return f"# TutorFlow AI — observability\n\n{r['message']}\n"
    q = r["requests"]
    g = r["graph"]
    l = r["llm"]
    items = ["# TutorFlow AI — observability report", "",
             f"Source: `{r['event_file']}`", "",
             f"- API requests: **{q['total']}**", f"- Average API latency: **{q['latency_avg_sec']} s**",
             f"- Max API latency: **{q['latency_max_sec']} s**",
             f"- Slow API requests (>60 s): **{q['slow_over_60_sec']}**",
             f"- Agent node spans: **{g['node_spans']}**, failed: **{g['failed_spans']}**",
             f"- LLM calls: **{l['calls']}**, mean generation TPS: **{l['mean_generation_tps']}**",
             "", "## Validation statuses", ""]
    items += [f"- {k}: {v}" for k,v in sorted(q['validation_statuses'].items())]
    items += ["", "## Models", ""]
    items += [f"- {k}: {v} calls" for k,v in sorted(l['by_model'].items())]
    items += ["", "## Local alert conditions", ""]
    items += [f"- {x}" for x in r["local_alerts"]] or ["- None observed"]
    items += ["", "**Limitations:** These are local file-based events, not distributed tracing or externally delivered alerts. "
              "Status APPROVED does not prove mathematical correctness. Non-streaming Ollama calls do not provide TTFT."]
    return "\n".join(items) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path(".runtime/events.jsonl"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    result = summarize(args.events)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for out, data in ((args.json_out, json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                      (args.md_out, markdown(result))):
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(data, encoding="utf-8")


if __name__ == "__main__":
    main()
