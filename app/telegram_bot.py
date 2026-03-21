from __future__ import annotations

import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from app.config import settings
from app.rag_pipeline import RAGPipeline   # ← остаётся как было

# ==================== ТРИ РЕЖИМА ====================
_pipeline: RAGPipeline | None = None
_chat_modes: dict[int, str] = {}

CHUNK_MODE_BUTTON = "ChunkBased"
ENTITY_MODE_BUTTON = "EntityBased"
HYBRID_MODE_BUTTON = "Hybrid"          # ← НОВОЕ

MODE_BUTTONS = {
    CHUNK_MODE_BUTTON: "chunk",
    ENTITY_MODE_BUTTON: "entity",
    HYBRID_MODE_BUTTON: "hybrid",
}


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def build_mode_keyboard() -> ReplyKeyboardMarkup:
    """Три кнопки — как ты просил"""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=CHUNK_MODE_BUTTON)],
            [KeyboardButton(text=ENTITY_MODE_BUTTON)],
            [KeyboardButton(text=HYBRID_MODE_BUTTON)],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def get_chat_mode(chat_id: int) -> str:
    return _chat_modes.get(chat_id, "chunk")


def set_chat_mode(chat_id: int, mode: str) -> None:
    _chat_modes[chat_id] = mode


# def build_bot_answer(result: dict) -> str:
#     """Твой текущий метод (оставил без изменений — он уже показывает полные чанки)"""
#     llm_part = result.get("answer", "Нет ответа от LLM.")
#     hits = result.get("hits", [])

#     if not hits:
#         return llm_part

#     chunks_block = ["\n📚 **Полные чанки из базы:**"]
#     for i, hit in enumerate(hits, 1):
#         source = hit.get("source", "unknown.pdf")
#         score = float(hit.get("score", 0.0))
#         full_text = hit.get("text", "").strip()
#         chunks_block.append(
#             f"[{i}] **{source}** (score={score:.3f})\n"
#             f"{full_text}\n"
#             f"{'─' * 50}"
#         )

#     return llm_part + "\n\n" + "\n".join(chunks_block)


async def start_handler(message: Message) -> None:
    chat_id = message.chat.id
    set_chat_mode(chat_id, "chunk")
    await message.answer(
        "Выбери режим ответа ниже 👇\n"
        "По умолчанию: ChunkBased",
        reply_markup=build_mode_keyboard(),
    )


async def mode_handler(message: Message) -> None:
    mode = MODE_BUTTONS.get((message.text or "").strip())
    if not mode:
        return

    set_chat_mode(message.chat.id, mode)
    await message.answer(f"✅ Режим изменён на: **{message.text}**")


async def question_handler(message: Message) -> None:
    chat_id = message.chat.id
    question = (message.text or "").strip()
    if not question:
        await message.answer("Отправь непустой вопрос.")
        return

    mode = get_chat_mode(chat_id)

    await message.answer(
        f"🔄 Processing your question in **{mode}** mode...\n"
        "(если вопрос сложный — будет несколько ответов)"
    )

    try:
        result = await asyncio.to_thread(
            lambda: get_pipeline().answer(question, mode=mode, top_k=6)
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка обработки: {exc}")
        return

    results = result if isinstance(result, list) else [result]

    # === 1. Шапка ===
    first = results[0]
    await message.answer(
        f"Mode: **{first.get('mode', mode).upper()}**\n"
        f"✅ **Ответ от локальной LLM** (`{first.get('model', 'llama3.2:3b')}`)"
    )

    # === 2. Ответы ===
    show_subtitles = len(results) > 1
    for i, r in enumerate(results, 1):
        sub_q = r.get("question", question)
        answer_text = r.get("answer", "Нет ответа от LLM.")

        if show_subtitles:
            await message.answer(f"**Под-вопрос {i}:** {sub_q}\n\n**Ответ:**\n{answer_text}")
        else:
            await message.answer(answer_text)

    # === 3. ЧАНКИ — ТОЛЬКО СПИСОК (без текста!) ===
    all_chunks = []
    for r in results:
        all_chunks.extend(r.get("chunk_results", []))
        all_chunks.extend(r.get("entity_results", []))

    relevant = [c for c in all_chunks if c.get("score", 0) > 0.1]

    if relevant:
        await message.answer("📚 **Найденные чанки (score + источник):**")
        for i, chunk in enumerate(relevant, 1):
            score = chunk.get("score", 0.0)
            source = chunk.get("source", "unknown")
            await message.answer(f"[{i}] score={score:.3f} | {source}")
    else:
        await message.answer("⚠️ Чанки не найдены.")


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