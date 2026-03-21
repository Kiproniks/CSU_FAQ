from __future__ import annotations

import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from app.config import settings
from app.rag_pipeline import RAGPipeline

# Единый экземпляр пайплайна и хранение режима для каждого чата.
_pipeline: RAGPipeline | None = None
_chat_modes: dict[int, str] = {}

# Кнопки для пользователя и внутреннее сопоставление режимов.
CHUNK_MODE_BUTTON = "ChunkBased"
ENTITY_MODE_BUTTON = "EntityBased"
MODE_BUTTONS = {
    CHUNK_MODE_BUTTON: "chunk",
    ENTITY_MODE_BUTTON: "entity",
}


def get_pipeline() -> RAGPipeline:
    # Ленивая инициализация для быстрого старта бота.
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def build_mode_keyboard() -> ReplyKeyboardMarkup:
    # Постоянная клавиатура для быстрого переключения режима в чате.
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=CHUNK_MODE_BUTTON),
                KeyboardButton(text=ENTITY_MODE_BUTTON),
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def get_chat_mode(chat_id: int) -> str:
    # Для новых чатов по умолчанию используем режим chunk.
    return _chat_modes.get(chat_id, "chunk")


def set_chat_mode(chat_id: int, mode: str) -> None:
    # Сохраняем выбранный режим поиска для каждого чата/сессии.
    _chat_modes[chat_id] = mode


def format_hits(hits: list[dict], limit: int = 5) -> str:
    # Короткий список источников, добавляемый после ответа.
    if not hits:
        return ""

    lines = ["\nSources:"]
    for hit in hits[:limit]:
        strategy = hit.get("strategy", "unknown")
        score = float(hit.get("score", 0.0))
        source = hit.get("source", "unknown")
        lines.append(f"- {strategy} | score={score:.3f} | {source}")

    return "\n".join(lines)


def _snippet(text: str, max_len: int = 420) -> str:
    # Короткий фрагмент для fallback-сообщений.
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def build_bot_answer(result: dict) -> str:
    # Основной ответ от LLM (всегда показываем)
    llm_part = result.get("answer", "Нет ответа от LLM.")

    hits = result.get("hits", [])
    if not hits:
        return llm_part

    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # Полные чанки ВСЕГДА (как ты просил)
    chunks_block = ["\n📚 **Полные чанки из базы (ChunkBased):**"]
    for i, hit in enumerate(hits, 1):
        source = hit.get("source", "unknown.pdf")
        score = float(hit.get("score", 0.0))
        full_text = hit.get("text", "").strip()          # полный текст без обрезки
        chunks_block.append(
            f"[{i}] **{source}** (score={score:.3f})\n"
            f"{full_text}\n"
            f"{'─' * 50}"
        )
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

    return llm_part + "\n\n" + "\n".join(chunks_block)


async def start_handler(message: Message) -> None:
    # Сбрасываем режим на стандартный и показываем выбор режима.
    chat_id = message.chat.id
    set_chat_mode(chat_id, "chunk")
    await message.answer(
        "Select answer mode, then send your question.\n"
        "Current mode: ChunkBased.",
        reply_markup=build_mode_keyboard(),
    )


async def mode_handler(message: Message) -> None:
    # Обработка нажатий кнопок режима.
    mode = MODE_BUTTONS.get((message.text or "").strip())
    if not mode:
        return
    set_chat_mode(message.chat.id, mode)
    await message.answer(f"Mode switched to: {message.text}")


async def question_handler(message: Message) -> None:
    # Обрабатываем обычный текст как вопрос в текущем режиме чата.
    chat_id = message.chat.id
    question = (message.text or "").strip()
    if not question:
        await message.answer("Please send a non-empty question.")
        return

    mode = get_chat_mode(chat_id)
    # Первый запрос может быть медленнее, потому что пайплайн инициализируется лениво.
    await message.answer(f"Processing your question in {mode} mode...")

    try:
        result = await asyncio.to_thread(lambda: get_pipeline().answer(question, mode=mode))
    except Exception as exc:
        await message.answer(f"Processing error: {exc}")
        return

    answer = build_bot_answer(result)
    response_text = f"Mode: {mode}\n\n{answer}"
    # Жесткий лимит сообщения Telegram — 4096 символов.
    await message.answer(response_text[:4000])


async def main() -> None:
    # Точка входа бота: проверка токена, регистрация хендлеров, запуск polling.
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

    bot = Bot(settings.telegram_bot_token)
    dp = Dispatcher()
    dp.message.register(start_handler, CommandStart())
    dp.message.register(mode_handler, F.text.in_(tuple(MODE_BUTTONS.keys())))
    dp.message.register(
        question_handler,
        F.text & ~F.text.startswith("/") & ~F.text.in_(tuple(MODE_BUTTONS.keys())),
    )
    print("Telegram bot polling started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

