import os
from dotenv import load_dotenv

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

# Загружаем ТОЛЬКО наш .env.mybot ПЕРВЫМ и с перезаписью
load_dotenv(".env.mybot", override=True)

print("🚀 Запускаю ТВОЕГО отдельного бота...")
token = os.getenv("TELEGRAM_BOT_TOKEN")
print(f"Токен: {token[:15]}... (скрыто)")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")

# Теперь импортируем остальное
from app.telegram_bot import main
import asyncio

if __name__ == '__main__':
    asyncio.run(main())