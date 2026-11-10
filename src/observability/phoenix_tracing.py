import os

_initialized = False


def setup_phoenix_tracing() -> None:
    global _initialized

    if _initialized:
        return

    if os.getenv("PHOENIX_ENABLED", "0") != "1":
        return

    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    register(
        project_name="TutorFlow-AI",
        endpoint=os.getenv(
            "PHOENIX_ENDPOINT",
            "http://127.0.0.1:6006/v1/traces",
        ),
        protocol="http/protobuf",
        batch=False,
    )

    LangChainInstrumentor().instrument()

    _initialized = True