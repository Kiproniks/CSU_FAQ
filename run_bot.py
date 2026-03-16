from app.telegram_bot import main
import asyncio

if __name__ == '__main__':
    # Легковесный запуск цикла polling для Telegram-бота.
    print("Starting Telegram bot...", flush=True)
    asyncio.run(main())

