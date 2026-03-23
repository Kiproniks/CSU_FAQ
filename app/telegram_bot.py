from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
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
from app.faq_catalog import build_faq_rows
from app.rag_pipeline import RAGPipeline
from app.source_attribution import SourceAttributionFormatter

# Единый экземпляр пайплайна и хранение режима для каждого чата.
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()
_pipeline_ready = threading.Event()
_chat_modes: dict[int, str] = {}

# Кнопки для пользователя и внутреннее сопоставление режимов.
CHUNK_MODE_BUTTON = "Режим поиска по фрагментам текста"
ENTITY_MODE_BUTTON = "Режим поиска по сущностям"
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
ANSWER_TIMEOUT_SEC = settings.answer_timeout_sec


def get_pipeline() -> RAGPipeline:
    # Ленивая инициализация для быстрого старта бота.
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
                _pipeline_ready.set()
    return _pipeline


def is_pipeline_ready() -> bool:
    return _pipeline_ready.is_set()


def _answer_sync_with_timeout(question: str, mode: str) -> dict:
    # Защита requests-backend от подвисания пайплайна.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(get_pipeline().answer, query=question, top_k=1, mode=mode)
    try:
        return future.result(timeout=ANSWER_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def _answer_async_with_timeout(question: str, mode: str) -> dict:
    # Защита aiogram-backend от долгого ответа пайплайна.
    return await asyncio.wait_for(
        asyncio.to_thread(lambda: get_pipeline().answer(question, top_k=1, mode=mode)),
        timeout=ANSWER_TIMEOUT_SEC,
    )


def get_db():
    return get_database()


def is_admin(chat_id: int) -> bool:
    chat_id_int = int(chat_id)
    return chat_id_int in set(settings.admin_telegram_ids or [])


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


def get_chat_mode(chat_id: int) -> str | None:
    # Режим считается выбранным только после явного нажатия кнопки.
    return _chat_modes.get(chat_id)


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
    # FAQ в 2 слоя: 10 фиксированных + 10 динамических вопросов из query_logs.
    try:
        dynamic_rows = get_db().top_faq_questions(last_n=last_n, top_n=50)
    except Exception as exc:
        return f"FAQ временно недоступен: {exc}"
    rows = build_faq_rows(dynamic_rows=dynamic_rows, dynamic_limit=10)
    if not rows:
        return "FAQ пока пуст. Задай несколько вопросов."

    lines = ["FAQ: 10 базовых + 10 актуальных"]
    for idx, row in enumerate(rows, start=1):
        question = " ".join(str(row.get("question", "")).split())
        if not question:
            continue

        saved_answer = " ".join(str(row.get("answer", "")).split())
        answer = _snippet(saved_answer, max_len=FAQ_ANSWER_MAX_LEN)

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
    if source not in {"telegram", "web"} or not external_id:
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
    if source not in {"telegram", "web"} or not external_id or amount <= 0:
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
    _chat_modes.pop(chat_id, None)
    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")

    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)

    _tg_send_message(
        chat_id,
        "Выберете режим,а потом спросите вопрос\n\n"
        "Полезные команды:\n"
        "/faq",
        reply_markup=_reply_keyboard_payload(),
    )


def _sync_token_handler(chat_id: int) -> None:
    _tg_send_message(chat_id, VERIFICATION_DISABLED_TEXT)


def _sync_verify_handler(chat_id: int, text: str) -> None:
    _tg_send_message(chat_id, VERIFICATION_DISABLED_TEXT)


def _sync_admin_handler(chat_id: int) -> None:
    # /admin для requests-backend: просто ссылка на веб-админку.
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    admin_text = build_admin_help_text()
    _tg_send_message(chat_id, admin_text)


def _sync_admin_stats_handler(chat_id: int, text: str) -> None:
    # Статистика активности за N дней (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    raw = _command_arg(text)
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().activity_series(days=days)
    if not rows:
        _tg_send_message(chat_id, "Нет данных активности (БД недоступна или пуста).")
        return

    lines = [f"Активность за последние {days} дн.:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_new_users_handler(chat_id: int, text: str) -> None:
    # Статистика новых пользователей за N дней (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    raw = _command_arg(text)
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().new_users_series(days=days)
    if not rows:
        _tg_send_message(chat_id, "Нет данных по пользователям (БД недоступна или пуста).")
        return

    lines = [f"Новые пользователи за последние {days} дн.:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_tokens_handler(chat_id: int, text: str) -> None:
    # Список последних токенов (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    raw = _command_arg(text)
    try:
        limit = max(1, min(50, int(raw))) if raw else 10
    except ValueError:
        limit = 10

    rows = get_db().list_tokens(limit=limit)
    if not rows:
        _tg_send_message(chat_id, "Нет данных по токенам (БД недоступна или пуста).")
        return

    lines = [f"Последние {limit} токенов:"]
    for row in rows:
        status = "active" if row.get("active") else "disabled"
        lines.append(f"{row.get('source')}:{row.get('external_id')} | {status} | {row.get('token')}")

    _tg_send_message(chat_id, "\n".join(lines))


def _sync_admin_activate_handler(chat_id: int, text: str) -> None:
    # Активация токена (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    token = _command_arg(text)
    if not token:
        _tg_send_message(chat_id, "Использование: /admin_activate <token>")
        return

    ok = get_db().set_token_active(token=token, active=True)
    _tg_send_message(chat_id, "Токен активирован." if ok else "Токен не найден / БД недоступна.")


def _sync_admin_deactivate_handler(chat_id: int, text: str) -> None:
    # Деактивация токена (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    token = _command_arg(text)
    if not token:
        _tg_send_message(chat_id, "Использование: /admin_deactivate <token>")
        return

    ok = get_db().set_token_active(token=token, active=False)
    _tg_send_message(chat_id, "Токен деактивирован." if ok else "Токен не найден / БД недоступна.")


def _sync_admin_make_admin_handler(chat_id: int, text: str) -> None:
    # Назначение admin роли пользователю (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    parsed = _parse_source_external(text)
    if not parsed:
        _tg_send_message(chat_id, "Использование: /admin_make_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=True)
    _tg_send_message(chat_id, f"Роль admin выдана: {source}:{external_id}")


def _sync_admin_remove_admin_handler(chat_id: int, text: str) -> None:
    # Снятие admin роли (requests backend).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    parsed = _parse_source_external(text)
    if not parsed:
        _tg_send_message(chat_id, "Использование: /admin_remove_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=False)
    _tg_send_message(chat_id, f"Роль admin снята: {source}:{external_id}")


def _sync_admin_add_tokens_handler(chat_id: int, text: str) -> None:
    # Добавление бонусных токенов (к месячному лимиту).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    parsed = _parse_source_external_amount(text)
    if not parsed:
        _tg_send_message(chat_id, "Использование: /admin_add_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=amount)
    if new_balance is None:
        _tg_send_message(chat_id, "БД недоступна.")
        return
    effective_limit = settings.user_monthly_request_limit + new_balance
    _tg_send_message(
        chat_id,
        f"Бонусные токены обновлены: {source}:{external_id} => {new_balance}. "
        f"Эффективный месячный лимит: {effective_limit}.",
    )


def _sync_admin_take_tokens_handler(chat_id: int, text: str) -> None:
    # Снятие бонусных токенов (из месячного лимита, не ниже нуля).
    if not is_admin(chat_id):
        _tg_send_message(chat_id, "Доступ запрещен.")
        return

    parsed = _parse_source_external_amount(text)
    if not parsed:
        _tg_send_message(chat_id, "Использование: /admin_take_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=-amount)
    if new_balance is None:
        _tg_send_message(chat_id, "БД недоступна.")
        return
    effective_limit = settings.user_monthly_request_limit + new_balance
    _tg_send_message(
        chat_id,
        f"Бонусные токены обновлены: {source}:{external_id} => {new_balance}. "
        f"Эффективный месячный лимит: {effective_limit}.",
    )


def _sync_question_handler(chat_id: int, question: str) -> None:
    mode = get_chat_mode(chat_id)
    if mode not in {"chunk", "entity"}:
        _tg_send_message(chat_id, "Выберете режим,а потом спросите вопрос")
        return
    if not is_pipeline_ready():
        _tg_send_message(chat_id, "Сервис прогревается. Повторите вопрос через 10-20 секунд.")
        return

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
            f"Лимит исчерпан: {quota.get('used', 0)} / {quota.get('limit', settings.user_monthly_request_limit)}.",
        )
        return

    _tg_send_message(chat_id, "Обрабатываю ваш вопрос...")

    try:
        started = time.perf_counter()
        result = _answer_sync_with_timeout(question, mode)
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
    except concurrent.futures.TimeoutError:
        _tg_send_message(chat_id, f"Превышено время ожидания ответа ({ANSWER_TIMEOUT_SEC} сек). Попробуйте еще раз.")
        return
    except Exception as exc:
        _tg_send_message(chat_id, f"Ошибка обработки: {exc}")
        return

    mode_title = "Поиск по фрагментам текста" if mode == "chunk" else "Поиск по сущностям"
    response_text = f"Режим: {mode_title}\n\n{answer}{format_primary_hit(result.get('hits', []))}"
    _tg_send_message(chat_id, response_text)


def _sync_faq_handler(chat_id: int) -> None:
    mode = get_chat_mode(chat_id) or "chunk"
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
        _tg_send_message(chat_id, f"Теперь вы используете: {raw}")
        _tg_send_message(chat_id, "Вы можете задать свой вопрос")
        return
    if raw == FAQ_BUTTON:
        _sync_faq_handler(chat_id)
        return

    if command in {"/chunk", "/chunkbased"}:
        set_chat_mode(chat_id, "chunk")
        _tg_send_message(chat_id, "Теперь вы используете: Режим поиска по фрагментам текста")
        _tg_send_message(chat_id, "Вы можете задать свой вопрос")
        return
    if command in {"/entity", "/entitybased"}:
        set_chat_mode(chat_id, "entity")
        _tg_send_message(chat_id, "Теперь вы используете: Режим поиска по сущностям")
        _tg_send_message(chat_id, "Вы можете задать свой вопрос")
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
        _tg_send_message(chat_id, "Выберете режим,а потом спросите вопрос")
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
    _chat_modes.pop(chat_id, None)

    db = get_db()
    db.register_user(external_id=str(chat_id), source="telegram")
    if is_admin(chat_id):
        db.set_user_admin(external_id=str(chat_id), source="telegram", value=True)

    await message.answer(
        "Выберете режим,а потом спросите вопрос\n\n"
        "Полежные команды:\n"
        "/faq",
        reply_markup=build_mode_keyboard(),
    )


async def mode_handler(message: Message) -> None:
    # Обработка нажатий кнопок режима.
    mode = MODE_BUTTONS.get((message.text or "").strip())
    if not mode:
        return
    set_chat_mode(message.chat.id, mode)
    await message.answer(f"Теперь вы используете: {message.text}")
    await message.answer("Вы можете задать свой вопрос")


async def faq_handler(message: Message) -> None:
    # Показ FAQ с кратким ответом по каждому вопросу.
    mode = get_chat_mode(message.chat.id) or "chunk"
    text = await asyncio.to_thread(build_faq_text, 100, 10, mode)
    await message.answer(text[:4000])


async def token_handler(message: Message) -> None:
    await message.answer(VERIFICATION_DISABLED_TEXT)


async def verify_handler(message: Message) -> None:
    await message.answer(VERIFICATION_DISABLED_TEXT)


async def unknown_command_handler(message: Message) -> None:
    # Для любых неизвестных команд не оставляем пользователя без понятного шага.
    await message.answer("Выберете режим,а потом спросите вопрос")


async def admin_mini_handler(message: Message) -> None:
    # /admin: просто ссылка на веб-админку.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    admin_text = build_admin_help_text()
    await message.answer(admin_text)


async def admin_stats_handler(message: Message) -> None:
    # График активности (в текстовом виде) за период.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    raw = _command_arg(message.text or "")
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().activity_series(days=days)
    if not rows:
        await message.answer("Нет данных активности (БД недоступна или пуста).")
        return

    lines = [f"Активность за последние {days} дн.:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    await message.answer("\n".join(lines)[:4000])


async def admin_new_users_handler(message: Message) -> None:
    # График новых пользователей (в текстовом виде) за период.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    raw = _command_arg(message.text or "")
    try:
        days = max(1, int(raw)) if raw else settings.admin_default_days
    except ValueError:
        days = settings.admin_default_days

    rows = get_db().new_users_series(days=days)
    if not rows:
        await message.answer("Нет данных по пользователям (БД недоступна или пуста).")
        return

    lines = [f"Новые пользователи за последние {days} дн.:"]
    for row in rows:
        lines.append(f"{row['day']}: {row['count']}")

    await message.answer("\n".join(lines)[:4000])


async def admin_tokens_handler(message: Message) -> None:
    # Просмотр последних токенов.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    raw = _command_arg(message.text or "")
    try:
        limit = max(1, min(50, int(raw))) if raw else 10
    except ValueError:
        limit = 10

    rows = get_db().list_tokens(limit=limit)
    if not rows:
        await message.answer("Нет данных по токенам (БД недоступна или пуста).")
        return

    lines = [f"Последние {limit} токенов:"]
    for row in rows:
        status = "active" if row.get("active") else "disabled"
        lines.append(f"{row.get('source')}:{row.get('external_id')} | {status} | {row.get('token')}")

    await message.answer("\n".join(lines)[:4000])


async def admin_deactivate_handler(message: Message) -> None:
    # Деактивация токена админом: /admin_deactivate <token>.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    token = _command_arg(message.text or "")
    if not token:
        await message.answer("Использование: /admin_deactivate <token>")
        return

    ok = get_db().set_token_active(token=token, active=False)
    await message.answer("Токен деактивирован." if ok else "Токен не найден / БД недоступна.")


async def admin_activate_handler(message: Message) -> None:
    # Активация токена админом: /admin_activate <token>.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    token = _command_arg(message.text or "")
    if not token:
        await message.answer("Использование: /admin_activate <token>")
        return

    ok = get_db().set_token_active(token=token, active=True)
    await message.answer("Токен активирован." if ok else "Токен не найден / БД недоступна.")


async def admin_make_admin_handler(message: Message) -> None:
    # Назначение admin роли пользователю: /admin_make_admin <source> <external_id>.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    parsed = _parse_source_external(message.text or "")
    if not parsed:
        await message.answer("Использование: /admin_make_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=True)
    await message.answer(f"Роль admin выдана: {source}:{external_id}")


async def admin_remove_admin_handler(message: Message) -> None:
    # Снятие admin роли: /admin_remove_admin <source> <external_id>.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    parsed = _parse_source_external(message.text or "")
    if not parsed:
        await message.answer("Использование: /admin_remove_admin <source> <external_id>")
        return

    source, external_id = parsed
    get_db().set_user_admin(external_id=external_id, source=source, value=False)
    await message.answer(f"Роль admin снята: {source}:{external_id}")


async def admin_add_tokens_handler(message: Message) -> None:
    # Добавление бонусных токенов к месячному лимиту.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    parsed = _parse_source_external_amount(message.text or "")
    if not parsed:
        await message.answer("Использование: /admin_add_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=amount)
    if new_balance is None:
        await message.answer("БД недоступна.")
        return

    effective_limit = settings.user_monthly_request_limit + new_balance
    await message.answer(
        f"Бонусные токены обновлены: {source}:{external_id} => {new_balance}. "
        f"Эффективный месячный лимит: {effective_limit}."
    )


async def admin_take_tokens_handler(message: Message) -> None:
    # Снятие бонусных токенов из месячного лимита.
    if not is_admin(message.chat.id):
        await message.answer("Доступ запрещен.")
        return

    parsed = _parse_source_external_amount(message.text or "")
    if not parsed:
        await message.answer("Использование: /admin_take_tokens <source> <external_id> <amount>")
        return

    source, external_id, amount = parsed
    new_balance = get_db().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=-amount)
    if new_balance is None:
        await message.answer("БД недоступна.")
        return

    effective_limit = settings.user_monthly_request_limit + new_balance
    await message.answer(
        f"Бонусные токены обновлены: {source}:{external_id} => {new_balance}. "
        f"Эффективный месячный лимит: {effective_limit}."
    )


async def question_handler(message: Message) -> None:
    # Обрабатываем обычный текст как вопрос в текущем режиме чата.
    chat_id = message.chat.id
    question = (message.text or "").strip()
    if not question:
        await message.answer("Отправьте непустой вопрос.")
        return

    mode = get_chat_mode(chat_id)
    if mode not in {"chunk", "entity"}:
        await message.answer("Выберете режим,а потом спросите вопрос")
        return
    if not is_pipeline_ready():
        await message.answer("Сервис прогревается. Повторите вопрос через 10-20 секунд.")
        return

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
            f"Лимит исчерпан: {quota.get('used', 0)} / {quota.get('limit', settings.user_monthly_request_limit)}."
        )
        return

    # Первый запрос может быть медленнее, потому что пайплайн инициализируется лениво.
    await message.answer("Обрабатываю ваш вопрос...")

    try:
        started = time.perf_counter()
        result = await _answer_async_with_timeout(question, mode)
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
    except asyncio.TimeoutError:
        await message.answer(f"Превышено время ожидания ответа ({ANSWER_TIMEOUT_SEC} сек). Попробуйте еще раз.")
        return
    except Exception as exc:
        await message.answer(f"Ошибка обработки: {exc}")
        return

    mode_title = "Поиск по фрагментам текста" if mode == "chunk" else "Поиск по сущностям"
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
    dp.message.register(unknown_command_handler, F.text.startswith("/"))

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




