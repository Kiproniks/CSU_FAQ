import asyncio
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.telegram_bot import get_pipeline, main


def _warmup_pipeline_background() -> None:
    # Прогрев в фоне: polling стартует сразу, а индекс готовится параллельно.
    try:
        get_pipeline()
        print("Telegram RAG warmup finished.", flush=True)
    except Exception as exc:
        print(f"Telegram RAG warmup failed: {exc}", flush=True)


if __name__ == "__main__":
    print("Starting Telegram bot...", flush=True)
    threading.Thread(target=_warmup_pipeline_background, daemon=True).start()
    asyncio.run(main())
