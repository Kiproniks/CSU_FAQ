# CSU FAQ

Учебный RAG-проект по интеллектуальному поиску по PDF-документам.

## Что реализовано
- `ChunkBased` поиск по embedding (Chroma) с улучшенным смысловым разбиением текста.
- `EntityBased` поиск (TF-IDF + entity overlap), честно: это keyword/entity-level подход, не graph.
- `RAGPipeline` с режимами `chunk`, `entity`, `hybrid`.
- Локальная LLM через Ollama + fallback `echo`.
- Web UI для вопросов и верификации токенов.
- Telegram-бот с переключением режимов.
- PostgreSQL для токенов пользователей и аналитики (запросы/новые пользователи).
- Админ-панель с графиками и управлением токенами.

## Структура
- `ChunkBased/ChunkBased.py` — chunk retrieval.
- `EntityBased/EntityBased.py` — entity/keyword retrieval.
- `app/rag_pipeline.py` — общий пайплайн.
- `app/llm_service.py` — провайдеры LLM (`ollama`, `openai`, `echo`).
- `app/web_app.py` — web-приложение + админка.
- `app/telegram_bot.py` — Telegram-бот.
- `app/db.py` — PostgreSQL-слой (users, tokens, logs).
- `scripts/reindex_harry_potter.py` — переиндексация PDF.
- `scripts/chunk_size_lab.py` — удобный эксперимент по размерам чанков.

## Быстрый запуск (Windows, PowerShell)
```powershell
cd C:\path\to\CSU_FAQ
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Настройка `.env`
```powershell
Copy-Item .env.example .env
```

Минимум заполнить:
- `TELEGRAM_BOT_TOKEN` — токен бота.
- `LLM_PROVIDER=ollama`
- `LLM_MODEL=llama3.2:2b`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/csu_faq`
- `WEB_ADMIN_USERNAME=kiproniks`
- `ADMIN_WEB_PASSWORD=<ваш_пароль>`
- `ADMIN_TELEGRAM_IDS=<chat_id_админа_через_запятую>`
- `PUBLIC_BASE_URL=<публичный URL web, нужен для Telegram mini app /admin>`
- `WEBAPP_URL=<полный URL mini app, например https://.../admin/mini>`
- `USER_MONTHLY_REQUEST_LIMIT=100`
- `ENTITY_CHUNK_SIZE`, `ENTITY_CHUNK_OVERLAP`, `ENTITY_MIN_LENGTH`, `ENTITY_MAX_ENTITIES_PER_CHUNK`,
  `ENTITY_TFIDF_WEIGHT`, `ENTITY_OVERLAP_WEIGHT`, `ENTITY_MIN_SCORE` (тонкая настройка EntityBased)

Для оценки benchmark через GigaChat:
- `GIGACHAT_API_KEY` (если у тебя уже есть готовый bearer token)
- `GIGACHAT_AUTH_KEY` (если есть только Authorization Key; скрипт сам получит token)
- `GIGACHAT_SCOPE` (обычно `GIGACHAT_API_PERS`)
- `GIGACHAT_AUTH_URL` (обычно `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`)
- `GIGACHAT_BASE_URL`
- `GIGACHAT_MODEL` (например `GigaChat-2-Max`)
- `GIGACHAT_VERIFY_SSL` (`1`/`0`)
- `GIGACHAT_TRUST_ENV` (`0` отключает прокси из env для judge-запросов)

## Ollama
```powershell
ollama pull llama3.2
ollama cp llama3.2:latest llama3.2:2b
ollama run llama3.2:2b "Привет"
```

## PostgreSQL
1. Подними локальный PostgreSQL.
2. Создай БД `csu_faq`.
3. Проверь, что `DATABASE_URL` в `.env` указывает на эту БД.

Схема создается автоматически при первом обращении приложения.

## Переиндексация PDF
Скопируй книги в `harry_potter/`, затем:
```powershell
python scripts/reindex_harry_potter.py --clear-chunk-index --clear-entity-index
```

## Web
```powershell
python run_web.py
```
- Основная страница: `http://127.0.0.1:8000/`
- Верификация: `http://127.0.0.1:8000/verify`
- Админка: `http://127.0.0.1:8000/admin`
- Логин: кнопка **«Авторизоваться»** справа сверху на главной.
- Для админов лимит запросов отключен (unlimited), для обычных пользователей действует `USER_MONTHLY_REQUEST_LIMIT`.

## Telegram bot
```powershell
python run_bot.py
```

## Стабильный запуск (рекомендуется)
```powershell
run_all.bat
```
- поднимает `ollama serve` (если не запущен),
- запускает web + tg бота,
- перезапускает их при падении,
- пишет логи в папку `logs`.

Команды:
- `/start` — запуск и выбор режима.
- `/token` — выдать токен пользователя.
- `/verify <token>` — верификация пользователя.
- `/admin` — открыть Telegram mini app админ-панели (только админ, нужен `WEBAPP_URL` или `PUBLIC_BASE_URL`).
- `/admin_make_admin <source> <external_id>` — назначить админа.
- `/admin_remove_admin <source> <external_id>` — снять админа.
- `/admin_stats [days]` — активность запросов (только админ).
- `/admin_new_users [days]` — новые пользователи (только админ).
- `/admin_tokens [limit]` — список токенов (только админ).
- `/admin_deactivate <token>` / `/admin_activate <token>` — управление токенами (только админ).

## Benchmark chunk/entity (с оценкой GigaChat и логом в PostgreSQL)
1. Подготовь файл с 50 вопросами и эталонными ответами (JSON):
   - пример формата: `data/benchmark_questions.example.json`
2. Запусти benchmark:
```powershell
python scripts/chunk_size_lab.py --mode both --questions-file data\benchmark_questions.json
```

По умолчанию:
- `chunk` режим: 100 вариаций настроек;
- `entity` режим: 50 вариаций настроек;
- на каждый вопрос сохраняются:
  - все параметры варианта;
  - вопрос;
  - ожидаемый ответ;
  - полученный ответ;
  - score и комментарий judge (GigaChat/OpenAI/heuristic fallback);
  - контекст retrieval;
  - latency.

Результаты сохраняются:
- в PostgreSQL (`benchmark_runs`, `benchmark_variants`, `benchmark_results`);
- в файлы:
  - `data/chunk_size_lab/benchmark_details_<timestamp>.csv`
  - `data/chunk_size_lab/benchmark_summary_<timestamp>.csv`
  - `data/chunk_size_lab/benchmark_summary_<timestamp>.md`

Чтобы автоматически применить лучшие настройки в `.env`:
```powershell
python scripts/chunk_size_lab.py --mode both --questions-file data\benchmark_questions.json --apply-best-to-env
```

Построение графиков по итогам benchmark:
```powershell
python scripts/visualize_benchmark.py
```
PNG графики появятся в `data/chunk_size_lab/plots`.

## Диагностика LLM
```powershell
python -c "from app.config import settings; print(settings.llm_provider, settings.llm_model, settings.ollama_base_url)"
python -c "import requests; print(requests.get('http://127.0.0.1:11434/api/tags', timeout=5).status_code)"
```
Если не `200`, проверь запуск Ollama и прокси.
