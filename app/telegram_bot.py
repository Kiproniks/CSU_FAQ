from __future__ import annotations

import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from app.config import settings
from app.rag_pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None
_chat_modes: dict[int, str] = {}

CHUNK_MODE_BUTTON = "ChunkBased"
ENTITY_MODE_BUTTON = "EntityBased"
MODE_BUTTONS = {
    CHUNK_MODE_BUTTON: "chunk",
    ENTITY_MODE_BUTTON: "entity",
}


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def build_mode_keyboard() -> ReplyKeyboardMarkup:
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
    return _chat_modes.get(chat_id, "chunk")


def set_chat_mode(chat_id: int, mode: str) -> None:
    _chat_modes[chat_id] = mode


def format_hits(hits: list[dict], limit: int = 5) -> str:
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
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def build_bot_answer(result: dict) -> str:
    if result.get("provider") != "echo":
        return result.get("answer", "")

    hits = result.get("hits", [])
    if not hits:
        return "No relevant fragments were found in indexed files."

    lines = [
        "LLM is unavailable, showing best matching fragments from indexed files:",
    ]
    for hit in hits[:2]:
        source = hit.get("source", "unknown")
        score = float(hit.get("score", 0.0))
        lines.append(f"[{source}] score={score:.3f}")
        lines.append(_snippet(hit.get("text", "")))

    return "\n\n".join(lines)


async def start_handler(message: Message) -> None:
    chat_id = message.chat.id
    set_chat_mode(chat_id, "chunk")
    await message.answer(
        "Select answer mode, then send your question.\n"
        "Current mode: ChunkBased.",
        reply_markup=build_mode_keyboard(),
    )


async def mode_handler(message: Message) -> None:
    mode = MODE_BUTTONS.get((message.text or "").strip())
    if not mode:
        return
    set_chat_mode(message.chat.id, mode)
    await message.answer(f"Mode switched to: {message.text}")


async def question_handler(message: Message) -> None:
    chat_id = message.chat.id
    question = (message.text or "").strip()
    if not question:
        await message.answer("Please send a non-empty question.")
        return

    mode = get_chat_mode(chat_id)
    # First query may be slower because retrieval pipeline is initialized lazily.
    await message.answer(f"Processing your question in {mode} mode...")

    try:
        result = await asyncio.to_thread(lambda: get_pipeline().answer(question, mode=mode))
    except Exception as exc:
        await message.answer(f"Processing error: {exc}")
        return

    answer = build_bot_answer(result)
    response_text = f"Mode: {mode}\n\n{answer}{format_hits(result.get('hits', []))}"
    # Telegram message hard limit is 4096 chars.
    await message.answer(response_text[:4000])


async def main() -> None:
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
