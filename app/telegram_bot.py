from __future__ import annotations

import asyncio
import os
import time
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from app.config import settings
from app.db import get_database
from app.rag_pipeline import RAGPipeline
from app.source_attribution import SourceAttributionFormatter

# Единый экземпляр пайплайна и хранение режима для каждого чата.
_pipeline: RAGPipeline | None = None
_chat_modes: dict[int, str] = {}

# Кнопки для пользователя и внутреннее сопоставление режимов.
CHUNK_MODE_BUTTON = "ChunkBased"
ENTITY_MODE_BUTTON = "EntityBased"
FAQ_BUTTON = "FAQ"
MODE_BUTTONS = {
    CHUNK_MODE_BUTTON: "chunk",
    ENTITY_MODE_BUTTON: "entity",
}
BOT_RETRY_DELAY_SEC = 15
REQUESTS_POLL_TIMEOUT_SEC = 30
FAQ_ANSWER_MAX_LEN = 220
FAQ_MESSAGE_MAX_LEN = 3900
VERIFICATION_DISABLED_TEXT = "Верификация отключена: токены больше не требуются."


def get_pipeline() -> RAGPipeline:
    # Ленивая инициализация для быстрого старта бота.
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_db():
    return get_database()


def is_admin(chat_id: int) -> bool:
    chat_id_int = int(chat_id)
    if chat_id_int in set(settings.admin_telegram_ids or []):
        return True

    db_user = get_db().get_user(external_id=str(chat_id_int), source="telegram")
    return bool((db_user or {}).get("is_admin", False))


def build_mode_keyboard() -> ReplyKeyboardMarkup:
    # Постоянная клавиатура для быстрого переключения режима в чате.
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=CHUNK_MODE_BUTTON),
                KeyboardButton(text=ENTITY_MODE_BUTTON),
                KeyboardButton(text=FAQ_BUTTON),
            ],
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


def format_primary_hit(hits: list[dict]) -> str:
    # Добавляем только подпись лучшего источника без длинной цитаты.
    if not hits:
        return ""

    primary = hits[0]
    source = SourceAttributionFormatter.format_short(primary)
    if not source:
        return ""
    return f"\n\nИсточник: {source}"


def build_admin_help_text(public_base_url: str = "") -> str:
    # /admin всегда отдает локальную ссылку по запросу пользователя.
    return "Админ-панель: http://127.0.0.1:8000/admin"


def build_faq_text(last_n: int = 100, top_n: int = 10, mode: str = "chunk") -> str:
    # Топ часто задаваемых вопросов + сохраненные ответы из query_logs.
    try:
        rows = get_db().top_faq_questions(last_n=last_n, top_n=top_n)
    except Exception as exc:
        return f"FAQ временно недоступен: {exc}"
    if not rows:
        return "FAQ пока пуст. Задай несколько вопросов."

    lines = ["FAQ: топ вопросов"]
    for idx, row in enumerate(rows, start=1):
        question = " ".join(str(row.get("question", "")).split())
        if not question:
            continue

        saved_answer = " ".join(str(row.get("answer", "")).split())
        if saved_answer:
            answer = _snippet(saved_answer, max_len=FAQ_ANSWER_MAX_LEN)
        else:
            answer = "Ответ появится после следующего запроса этого вопроса в QA."

        block = f"{idx}. Вопрос: {question}\nОтвет: {answer}"
        if len("\n\n".join(lines + [block])) > FAQ_MESSAGE_MAX_LEN:
            lines.append("... список обрезан по лимиту сообщения Telegram.")
            break
        lines.append(block)

    return "\n\n".join(lines)


def _snippet(text: str, max_len: int = 420) -> str:
    # Короткий фрагмент для fallback-сообщений.
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def build_bot_answer(result: dict) -> str:
    # Возвращаем ответ LLM; при echo сообщаем о недоступности без цитирования контекста.
    if result.get("provider") != "echo":
        return result.get("answer", "")

    return "LLM временно недоступна. Повтори вопрос позже."


def _command_arg(text: str) -> str:
    parts = (text or "").strip().split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _parse_source_external(text: str) -> tuple[str, str] | None:
    # Ожидаем аргументы в формате: "<source> <external_id>".
    parts = (text or "").strip().split(maxsplit=2)
    if len(parts) < 3:
        return None
    source = parts[1].strip().lower()
    external_id = parts[2].strip()
    if not source or not external_id:
        return None
    return source, external_id


def _parse_source_external_amount(text: str) -> tuple[str, str, int] | None:
    # Ожидаем аргументы в формате: "<source> <external_id> <amount>".
    parts = (text or "").strip().split(maxsplit=3)
    if len(parts) < 4:
        return None
    source = parts[1].strip().lower()
    external_id = parts[2].strip()
    try:
        amount = int(parts[3].strip())
    except ValueError:
        return None
    if not source or not external_id or amount <= 0:
        return None
    return source, external_id, amount

def _reply_keyboard_payload() -> dict:
    # JSON-клавиатура для requests backend.
    return {
        "keyboard": [[{"text": CHUNK_MODE_BUTTON}, {"text": ENTITY_MODE_BUTTON}, {"text": FAQ_BUTTON}]],
        "resize_keyboard": True,
        "one_time_keyboard": False,
    }


def _tg_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{settings.telegram_bot_token}/{method}"


def _tg_send_message(chat_id: int, text: str, reply_markup: dict | None = None) -> None:
    payload: dict = {"chat_id": int(chat_id), "text": (text or "")[:4000]}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        response = requests.post(_tg_api_url("sendMessage"), json=payload, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"Telegram sendMessage error for chat_id={chat_id}: {exc}")


def _sync_start_handler(chat_id: int) -> None:
    set_chat_mode(chat_id, "chunk")
    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")

    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)
        limit_line = "Monthly limit: unlimited (admin)."
    else:
        limit_line = f"Monthly limit: {settings.user_monthly_request_limit} requests."

    _tg_send_message(
        chat_id,
        "Select answer mode, then send your question.\n"
        "Current mode: ChunkBased.\n\n"
        "Useful commands:\n"
        "/admin - admin panel link\n"
        "/faq - show FAQ with answers\n"
        f"{limit_line}",
        reply_markup=_reply_keyboard_payload(),
    )


def _sync_token_handler(chat_id: int) -> None:
    _tg_send_message(chat_id, VERIFICATION_DISABLED_TEXT)


def _sync_verify_handler(chat_id: int, text: str) -> None:
    _tg_send_message(chat_id, VERIFICATION_DISABLED_TEXT)


def _sync_admin_handler(chat_id: int) -> None:
    # /admin для requests-backend: просто ссылка на веб-админку.
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    admin_text = build_admin_help_text()
    _tg_send_message(chat_id, admin_text)


def _sync_admin_stats_handler(chat_id: int, text: str) -> None:
    # Статистика активности за N дней (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    raw = _command_arg(text)
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().activity_series(days=days)
    if not rows:
        _tg_send_message(chat_id, "No activity data (DB unavailable or empty).")
        return

    lines = [f"Activity for last {days} days:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_new_users_handler(chat_id: int, text: str) -> None:
    # Статистика новых пользователей за N дней (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    raw = _command_arg(text)
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().new_users_series(days=days)
    if not rows:
        _tg_send_message(chat_id, "No user data (DB unavailable or empty).")
        return

    lines = [f"New users for last {days} days:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_tokens_handler(chat_id: int, text: str) -> None:
    # Список последних токенов (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    raw = _command_arg(text)
    try:
        limit = max(1, min(50, int(raw))) if raw else 10
    except ValueError:
        limit = 10

    rows = get_db().list_tokens(limit=limit)
    if not rows:
        _tg_send_message(chat_id, "No token data (DB unavailable or empty).")
        return

    lines = [f"Latest {limit} tokens:"]
    for row in rows:
        status = "active" if row.get("active") else "disabled"
        lines.append(f"{row.get('source')}:{row.get('external_id')} | {status} | {row.get('token')}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_activate_handler(chat_id: int, text: str) -> None:
    # Активация токена (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    token = _command_arg(text)
    if not token:
        _tg_send_message(chat_id, "Usage: /admin_activate <token>")
        return

    ok = get_db().set_token_active(token=token, active=True)
    _tg_send_message(chat_id, "Token activated." if ok else "Token not found / DB unavailable.")


def _sync_admin_deactivate_handler(chat_id: int, text: str) -> None:
    # Деактивация токена (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    token = _command_arg(text)
    if not token:
        _tg_send_message(chat_id, "Usage: /admin_deactivate <token>")
        return

    ok = get_db().set_token_active(token=token, active=False)
    _tg_send_message(chat_id, "Token deactivated." if ok else "Token not found / DB unavailable.")


def _sync_admin_make_admin_handler(chat_id: int, text: str) -> None:
    # Назначение admin роли пользователю (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    parsed = _parse_source_external(text)
    if not parsed:
        _tg_send_message(chat_id, "Usage: /admin_make_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=True)
    _tg_send_message(chat_id, f"Admin role granted: {source}:{external_id}")


def _sync_admin_remove_admin_handler(chat_id: int, text: str) -> None:
    # Снятие admin роли (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    parsed = _parse_source_external(text)
    if not parsed:
        _tg_send_message(chat_id, "Usage: /admin_remove_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=False)
    _tg_send_message(chat_id, f"Admin role removed: {source}:{external_id}")


def _sync_admin_add_tokens_handler(chat_id: int, text: str) -> None:
    # Добавление бонусных токенов (к месячному лимиту).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    parsed = _parse_source_external_amount(text)
    if not parsed:
        _tg_send_message(chat_id, "Usage: /admin_add_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=amount)
    if new_balance is None:
        _tg_send_message(chat_id, "DB unavailable.")
        return
    effective_limit = settings.user_monthly_request_limit + new_balance
    _tg_send_message(
        chat_id,
        f"Bonus tokens updated: {source}:{external_id} => {new_balance}. "
        f"Effective monthly limit: {effective_limit}.",
    )


def _sync_admin_take_tokens_handler(chat_id: int, text: str) -> None:
    # Снятие бонусных токенов (из месячного лимита, не ниже нуля).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Access denied.")
        return

    parsed = _parse_source_external_amount(text)
    if not parsed:
        _tg_send_message(chat_id, "Usage: /admin_take_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=-amount)
    if new_balance is None:
        _tg_send_message(chat_id, "DB unavailable.")
        return
    effective_limit = settings.user_monthly_request_limit + new_balance
    _tg_send_message(
        chat_id,
        f"Bonus tokens updated: {source}:{external_id} => {new_balance}. "
        f"Effective monthly limit: {effective_limit}.",
    )


def _sync_question_handler(chat_id: int, question: str) -> None:
    mode = get_chat_mode(chat_id)
    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")
    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)

    quota = db.check_monthly_quota(
        external_id=str(chat_id),
        source="telegram",
        monthly_limit=settings.user_monthly_request_limit,
    )
    if not quota.get("allowed", True):
        _tg_send_message(
            chat_id,
            f"Monthly limit reached: {quota.get('used', 0)} / {quota.get('limit', settings.user_monthly_request_limit)}.",
        )
        return

    _tg_send_message(chat_id, f"Processing your question in {mode} mode...")

    try:
        started = time.perf_counter()
        result = get_pipeline().answer(question, mode=mode)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = build_bot_answer(result)

        db.log_query(
            external_id=str(chat_id),
            source="telegram",
            mode=mode,
            query=question,
            generated_answer=answer,
            provider=result.get("provider", ""),
            model=result.get("model", ""),
            latency_ms=latency_ms,
        )
    except Exception as exc:
        _tg_send_message(chat_id, f"Processing error: {exc}")
        return

    mode_title = "ChunkBased" if mode == "chunk" else "EntityBased"
    response_text = f"Режим: {mode_title}\n\n{answer}{format_primary_hit(result.get('hits', []))}"
    _tg_send_message(chat_id, response_text)


def _sync_faq_handler(chat_id: int) -> None:
    mode = get_chat_mode(chat_id)
    _tg_send_message(chat_id, build_faq_text(mode=mode), reply_markup=_reply_keyboard_payload())


def _handle_text_update_sync(chat_id: int, text: str) -> None:
    raw = (text or "").strip()
    if not raw:
        return

    command = raw.split(maxsplit=1)[0].lower()
    if command.startswith("/") and "@" in command:
        command = command.split("@", 1)[0]

    if raw in MODE_BUTTONS:
        mode = MODE_BUTTONS[raw]
        set_chat_mode(chat_id, mode)
        _tg_send_message(chat_id, f"Mode switched to: {raw}")
        _tg_send_message(chat_id, "вы можете задать свой вопрос")
        return
    if raw == FAQ_BUTTON:
        _sync_faq_handler(chat_id)
        return

    if command in {"/chunk", "/chunkbased"}:
        set_chat_mode(chat_id, "chunk")
        _tg_send_message(chat_id, "Mode switched to: ChunkBased")
        _tg_send_message(chat_id, "вы можете задать свой вопрос")
        return
    if command in {"/entity", "/entitybased"}:
        set_chat_mode(chat_id, "entity")
        _tg_send_message(chat_id, "Mode switched to: EntityBased")
        _tg_send_message(chat_id, "вы можете задать свой вопрос")
        return

    if command == "/start":
        _sync_start_handler(chat_id)
        return
    if command == "/token":
        _sync_token_handler(chat_id)
        return
    if command == "/verify":
        _sync_verify_handler(chat_id, raw)
        return
    if command == "/admin_stats":
        _sync_admin_stats_handler(chat_id, raw)
        return
    if command == "/admin_new_users":
        _sync_admin_new_users_handler(chat_id, raw)
        return
    if command == "/admin_tokens":
        _sync_admin_tokens_handler(chat_id, raw)
        return
    if command == "/admin_activate":
        _sync_admin_activate_handler(chat_id, raw)
        return
    if command == "/admin_deactivate":
        _sync_admin_deactivate_handler(chat_id, raw)
        return
    if command == "/admin_make_admin":
        _sync_admin_make_admin_handler(chat_id, raw)
        return
    if command == "/admin_remove_admin":
        _sync_admin_remove_admin_handler(chat_id, raw)
        return
    if command == "/admin_add_tokens":
        _sync_admin_add_tokens_handler(chat_id, raw)
        return
    if command == "/admin_take_tokens":
        _sync_admin_take_tokens_handler(chat_id, raw)
        return
    if command == "/admin":
        _sync_admin_handler(chat_id)
        return
    if command == "/faq":
        _sync_faq_handler(chat_id)
        return
    if command.startswith("/"):
        _tg_send_message(chat_id, "Unknown command. Use /start.")
        return

    _sync_question_handler(chat_id, raw)


def _run_requests_polling_loop() -> None:
    # Резервный polling backend через requests, когда aiohttp/aiogram недоступен в сети.
    offset: int | None = None
    print("Telegram bot requests polling started.")
    while True:
        try:
            params: dict[str, int] = {"timeout": REQUESTS_POLL_TIMEOUT_SEC}
            if offset is not None:
                params["offset"] = int(offset)

            response = requests.get(
                _tg_api_url("getUpdates"),
                params=params,
                timeout=REQUESTS_POLL_TIMEOUT_SEC + 15,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok", False):
                print(f"Telegram getUpdates returned not ok: {payload}")
                time.sleep(BOT_RETRY_DELAY_SEC)
                continue

            for item in payload.get("result", []):
                if not isinstance(item, dict):
                    continue
                update_id = item.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1

                message = item.get("message") or item.get("edited_message")
                if not isinstance(message, dict):
                    continue
                chat = message.get("chat") or {}
                text = (message.get("text") or "").strip()
                chat_id = int(chat.get("id", 0)) if chat.get("id") else 0
                if chat_id <= 0 or not text:
                    continue
                _handle_text_update_sync(chat_id, text)

        except Exception as exc:
            print(f"Telegram requests polling error: {exc}. Retry in {BOT_RETRY_DELAY_SEC}s.")
            time.sleep(BOT_RETRY_DELAY_SEC)


async def _probe_aiogram_transport() -> bool:
    # Быстрая проверка, может ли aiogram достучаться до Telegram API.
    bot = Bot(settings.telegram_bot_token)
    try:
        await bot.get_me(request_timeout=8)
        return True
    except Exception as exc:
        print(f"Aiogram transport probe failed: {exc}")
        return False
    finally:
        await bot.session.close()


async def start_handler(message: Message) -> None:
    # Сбрасываем режим на стандартный и показываем выбор режима.
    chat_id = message.chat.id
    set_chat_mode(chat_id, "chunk")

    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")
    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)
        limit_line = "Monthly limit: unlimited (admin)."
    else:
        limit_line = f"Monthly limit: {settings.user_monthly_request_limit} requests."

    await message.answer(
        "Select answer mode, then send your question.\n"
        "Current mode: ChunkBased.\n\n"
        "Useful commands:\n"
        "/admin - admin panel link\n"
        "/faq - show FAQ with answers\n"
        f"{limit_line}",
        reply_markup=build_mode_keyboard(),
    )


async def mode_handler(message: Message) -> None:
    # Обработка нажатий кнопок режима.
    mode = MODE_BUTTONS.get((message.text or "").strip())
    if not mode:
        return
    set_chat_mode(message.chat.id, mode)
    await message.answer(f"Mode switched to: {message.text}")
    await message.answer("вы можете задать свой вопрос")


async def faq_handler(message: Message) -> None:
    # Показ FAQ с кратким ответом по каждому вопросу.
    mode = get_chat_mode(message.chat.id)
    text = await asyncio.to_thread(build_faq_text, 100, 10, mode)
    await message.answer(text[:4000])


async def token_handler(message: Message) -> None:
    await message.answer(VERIFICATION_DISABLED_TEXT)


async def verify_handler(message: Message) -> None:
    await message.answer(VERIFICATION_DISABLED_TEXT)


async def admin_mini_handler(message: Message) -> None:
    # /admin: просто ссылка на веб-админку.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    admin_text = build_admin_help_text()
    await message.answer(admin_text)


async def admin_stats_handler(message: Message) -> None:
    # График активности (в текстовом виде) за период.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    raw = _command_arg(message.text or "")
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().activity_series(days=days)
    if not rows:
        await message.answer("No activity data (DB unavailable or empty).")
        return

    lines = [f"Activity for last {days} days:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    await message.answer("\n".join(lines)[:4000])


async def admin_new_users_handler(message: Message) -> None:
    # График новых пользователей (в текстовом виде) за период.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    raw = _command_arg(message.text or "")
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().new_users_series(days=days)
    if not rows:
        await message.answer("No user data (DB unavailable or empty).")
        return

    lines = [f"New users for last {days} days:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    await message.answer("\n".join(lines)[:4000])


async def admin_tokens_handler(message: Message) -> None:
    # Просмотр последних токенов.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    raw = _command_arg(message.text or "")
    try:
        limit = max(1, min(50, int(raw))) if raw else 10
    except ValueError:
        limit = 10

    rows = get_db().list_tokens(limit=limit)
    if not rows:
        await message.answer("No token data (DB unavailable or empty).")
        return

    lines = [f"Latest {limit} tokens:"]
    for row in rows:
        status = "active" if row.get("active") else "disabled"
        lines.append(f"{row.get('source')}:{row.get('external_id')} | {status} | {row.get('token')}")

    await message.answer("\n".join(lines)[:4000])


async def admin_deactivate_handler(message: Message) -> None:
    # Деактивация токена админом: /admin_deactivate <token>.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    token = _command_arg(message.text or "")
    if not token:
        await message.answer("Usage: /admin_deactivate <token>")
        return

    ok = get_db().set_token_active(token=token, active=False)
    await message.answer("Token deactivated." if ok else "Token not found / DB unavailable.")


async def admin_activate_handler(message: Message) -> None:
    # Активация токена админом: /admin_activate <token>.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    token = _command_arg(message.text or "")
    if not token:
        await message.answer("Usage: /admin_activate <token>")
        return

    ok = get_db().set_token_active(token=token, active=True)
    await message.answer("Token activated." if ok else "Token not found / DB unavailable.")


async def admin_make_admin_handler(message: Message) -> None:
    # Назначение admin роли пользователю: /admin_make_admin <source> <external_id>.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    parsed = _parse_source_external(message.text or "")
    if not parsed:
        await message.answer("Usage: /admin_make_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=True)
    await message.answer(f"Admin role granted: {source}:{external_id}")


async def admin_remove_admin_handler(message: Message) -> None:
    # Снятие admin роли: /admin_remove_admin <source> <external_id>.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    parsed = _parse_source_external(message.text or "")
    if not parsed:
        await message.answer("Usage: /admin_remove_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=False)
    await message.answer(f"Admin role removed: {source}:{external_id}")


async def admin_add_tokens_handler(message: Message) -> None:
    # Добавление бонусных токенов к месячному лимиту.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    parsed = _parse_source_external_amount(message.text or "")
    if not parsed:
        await message.answer("Usage: /admin_add_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=amount)
    if new_balance is None:
        await message.answer("DB unavailable.")
        return

    effective_limit = settings.user_monthly_request_limit + new_balance
    await message.answer(
        f"Bonus tokens updated: {source}:{external_id} => {new_balance}. "
        f"Effective monthly limit: {effective_limit}."
    )


async def admin_take_tokens_handler(message: Message) -> None:
    # Снятие бонусных токенов из месячного лимита.
    if not is_admin(message.chat.id):
        await message.answer("Access denied.")
        return

    parsed = _parse_source_external_amount(message.text or "")
    if not parsed:
        await message.answer("Usage: /admin_take_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=-amount)
    if new_balance is None:
        await message.answer("DB unavailable.")
        return

    effective_limit = settings.user_monthly_request_limit + new_balance
    await message.answer(
        f"Bonus tokens updated: {source}:{external_id} => {new_balance}. "
        f"Effective monthly limit: {effective_limit}."
    )


async def question_handler(message: Message) -> None:
    # Обрабатываем обычный текст как вопрос в текущем режиме чата.
    chat_id = message.chat.id
    question = (message.text or "").strip()
    if not question:
        await message.answer("Please send a non-empty question.")
        return

    mode = get_chat_mode(chat_id)
    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")
    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)

    quota = db.check_monthly_quota(
        external_id=str(chat_id),
        source="telegram",
        monthly_limit=settings.user_monthly_request_limit,
    )
    if not quota.get("allowed", True):
        await message.answer(
            f"Monthly limit reached: {quota.get('used', 0)} / {quota.get('limit', settings.user_monthly_request_limit)}."
        )
        return

    # Первый запрос может быть медленнее, потому что пайплайн инициализируется лениво.
    await message.answer(f"Processing your question in {mode} mode...")

    try:
        started = time.perf_counter()
        result = await asyncio.to_thread(lambda: get_pipeline().answer(question, mode=mode))
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = build_bot_answer(result)

        db.log_query(
            external_id=str(chat_id),
            source="telegram",
            mode=mode,
            query=question,
            generated_answer=answer,
            provider=result.get("provider", ""),
            model=result.get("model", ""),
            latency_ms=latency_ms,
        )
    except Exception as exc:
        await message.answer(f"Processing error: {exc}")
        return

    mode_title = "ChunkBased" if mode == "chunk" else "EntityBased"
    response_text = f"Режим: {mode_title}\n\n{answer}{format_primary_hit(result.get('hits', []))}"
    # Жесткий лимит сообщения Telegram — 4096 символов.
    await message.answer(response_text[:4000])


async def main() -> None:
    # Точка входа бота: проверка токена, регистрация хендлеров, запуск polling.
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

    backend = (os.getenv("TELEGRAM_POLLING_BACKEND", "auto") or "auto").strip().lower()
    if backend in {"requests", "sync"}:
        await asyncio.to_thread(_run_requests_polling_loop)
        return

    if backend == "auto":
        aiogram_ok = await _probe_aiogram_transport()
        if not aiogram_ok:
            print("Aiogram transport unavailable, switching to requests polling backend.")
            await asyncio.to_thread(_run_requests_polling_loop)
            return

    dp = Dispatcher()

    dp.message.register(start_handler, CommandStart())
    dp.message.register(faq_handler, Command("faq"))
    dp.message.register(token_handler, Command("token"))
    dp.message.register(verify_handler, Command("verify"))
    dp.message.register(admin_mini_handler, Command("admin"))

    dp.message.register(admin_stats_handler, Command("admin_stats"))
    dp.message.register(admin_new_users_handler, Command("admin_new_users"))
    dp.message.register(admin_tokens_handler, Command("admin_tokens"))
    dp.message.register(admin_deactivate_handler, Command("admin_deactivate"))
    dp.message.register(admin_activate_handler, Command("admin_activate"))
    dp.message.register(admin_make_admin_handler, Command("admin_make_admin"))
    dp.message.register(admin_remove_admin_handler, Command("admin_remove_admin"))
    dp.message.register(admin_add_tokens_handler, Command("admin_add_tokens"))
    dp.message.register(admin_take_tokens_handler, Command("admin_take_tokens"))

    dp.message.register(faq_handler, F.text == FAQ_BUTTON)
    dp.message.register(mode_handler, F.text.in_(tuple(MODE_BUTTONS.keys())))
    dp.message.register(
        question_handler,
        F.text & ~F.text.startswith("/") & ~F.text.in_(tuple(MODE_BUTTONS.keys()) + (FAQ_BUTTON,)),
    )

    while True:
        bot = Bot(settings.telegram_bot_token)
        try:
            print("Telegram bot polling started.")
            await dp.start_polling(bot)
            break
        except TelegramNetworkError as exc:
            print(f"Telegram network error: {exc}. Retry in {BOT_RETRY_DELAY_SEC}s.")
            await asyncio.sleep(BOT_RETRY_DELAY_SEC)
        finally:
            await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())



