import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.telegram_bot import main


if __name__ == "__main__":
    # Легковесный запуск цикла polling для Telegram-бота.
    print("Starting Telegram bot...", flush=True)
    asyncio.run(main())
