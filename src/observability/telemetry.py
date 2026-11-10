"""Non-PII request metrics/events. Not a replacement for distributed tracing or alerting."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOCK = Lock()
LOGGER = logging.getLogger("tutorflow.telemetry")
EVENT_PATH = Path(os.getenv("TUTORFLOW_EVENTS_PATH", str(Path(__file__).resolve().parents[2] / ".runtime" / "events.jsonl")))


def record_request(*, endpoint: str, task_type: str | None, validation_status: str | None,
                   latency_sec: float, trace_id: str | None = None) -> None:
    """No prompts, student names, session IDs or raw model output in persisted events."""
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint, "task_type": task_type, "trace_id": trace_id,
        "validation_status": validation_status, "latency_sec": latency_sec,
        "slow": latency_sec > 60.0,
    }
    try:
        with _LOCK:
            EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        LOGGER.exception("Could not persist request metrics")
    if event["slow"]:
        LOGGER.warning("Slow agent call: %s (%s seconds)", endpoint, latency_sec)


def record_span(*, name: str, trace_id: str | None, latency_sec: float, ok: bool) -> None:
    """Append one non-content span to reconstruct ordered graph trajectories."""
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "kind": "node_span", "node": name, "trace_id": trace_id,
        "latency_sec": latency_sec, "ok": ok,
    }
    try:
        with _LOCK:
            EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        LOGGER.exception("Could not persist node span")
    if not ok:
        LOGGER.error("Node failure: %s", name)


def record_llm(*, model: str, latency_sec: float,
               generated_tokens: int | None, eval_duration_ns: int | None) -> None:
    """Ollama token metrics. TTFT cannot be measured from non-streaming calls."""
    tps = (generated_tokens / (eval_duration_ns / 1e9)
           if generated_tokens and eval_duration_ns else None)
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "kind": "llm_metric", "model": model,
        "latency_sec": latency_sec, "generated_tokens": generated_tokens,
        "generation_tps": round(tps, 3) if tps else None,
    }
    try:
        with _LOCK:
            EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        LOGGER.exception("Could not persist LLM performance")
