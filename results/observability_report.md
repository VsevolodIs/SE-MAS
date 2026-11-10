# TutorFlow AI — observability report

Source: `.runtime\events.jsonl`

- API requests: **3**
- Average API latency: **52.955 s**
- Max API latency: **158.866 s**
- Slow API requests (>60 s): **1**
- Agent node spans: **5**, failed: **0**
- LLM calls: **2**, mean generation TPS: **4.258**

## Validation statuses

- APPROVED: 1
- NEEDS_REVIEW: 1
- UNKNOWN: 1

## Models

- llama3.2:3b: 1 calls
- qwen2.5:3b: 1 calls

## Local alert conditions

- SLOW_REQUESTS: 1 requests exceeded 60 sec

**Limitations:** These are local file-based events, not distributed tracing or externally delivered alerts. Status APPROVED does not prove mathematical correctness. Non-streaming Ollama calls do not provide TTFT.
