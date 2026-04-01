import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.web_app import app, get_pipeline


def _warmup_pipeline_background() -> None:
    # Прогрев в фоне: веб стартует сразу, а индексы поднимаются без блокировки старта.
    try:
        get_pipeline()
        print("RAG pipeline warmup finished.", flush=True)
    except Exception as exc:
        print(f"RAG warmup failed: {exc}", flush=True)


if __name__ == "__main__":
    print("Starting background warmup...", flush=True)
    threading.Thread(target=_warmup_pipeline_background, daemon=True).start()
    print("Starting web demo on http://127.0.0.1:8000", flush=True)
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
