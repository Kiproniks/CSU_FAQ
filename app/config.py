from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные окружения из .env один раз при импорте.
load_dotenv()


def _as_bool(value: str, default: bool = False) -> bool:
    # Преобразуем строковые флаги в bool с безопасным значением по умолчанию.
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int_list(value: str) -> list[int]:
    # Парсим CSV строку вида "123,456" в список int.
    if not value:
        return []
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            continue
    return result


@dataclass
class Settings:
    # Настройки рантайма и API.
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "echo")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:2b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    llm_timeout_sec: int = int(os.getenv("LLM_TIMEOUT_SEC", "30"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Retrieval настройки.
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    chunk_collection: str = os.getenv("CHUNK_COLLECTION", "harry_potter_collection")
    chunk_embedding_model: str = os.getenv(
        "CHUNK_EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunk_splitter_mode: str = os.getenv("CHUNK_SPLITTER_MODE", "smart")
    chunk_mmr_lambda: float = float(os.getenv("CHUNK_MMR_LAMBDA", "0.78"))
    top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", "3"))
    top_k_entities: int = int(os.getenv("TOP_K_ENTITIES", "3"))
    entity_chunk_size: int = int(os.getenv("ENTITY_CHUNK_SIZE", os.getenv("CHUNK_SIZE", "1200")))
    entity_chunk_overlap: int = int(os.getenv("ENTITY_CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP", "200")))
    entity_min_length: int = int(os.getenv("ENTITY_MIN_LENGTH", "3"))
    entity_max_entities_per_chunk: int = int(os.getenv("ENTITY_MAX_ENTITIES_PER_CHUNK", "16"))
    entity_tfidf_weight: float = float(os.getenv("ENTITY_TFIDF_WEIGHT", "0.8"))
    entity_overlap_weight: float = float(os.getenv("ENTITY_OVERLAP_WEIGHT", "0.2"))
    entity_min_score: float = float(os.getenv("ENTITY_MIN_SCORE", "0.03"))
    entity_mmr_lambda: float = float(os.getenv("ENTITY_MMR_LAMBDA", "0.74"))
    bootstrap_entity_from_chroma: bool = _as_bool(
        os.getenv("BOOTSTRAP_ENTITY_FROM_CHROMA", "1"),
        default=True,
    )

    # Web / Admin.
    web_secret_key: str = os.getenv("WEB_SECRET_KEY", "change-me-dev-secret")
    web_admin_username: str = os.getenv("WEB_ADMIN_USERNAME", "kiproniks")
    admin_web_password: str = os.getenv("ADMIN_WEB_PASSWORD", "")
    admin_default_days: int = int(os.getenv("ADMIN_DEFAULT_DAYS", "14"))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "").strip()
    mini_admin_ttl_sec: int = int(os.getenv("MINI_ADMIN_TTL_SEC", "900"))
    user_monthly_request_limit: int = int(os.getenv("USER_MONTHLY_REQUEST_LIMIT", "100"))

    # PostgreSQL аналитика и токены пользователей.
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/csu_faq",
    )

    # Telegram admin ids: "123,456".
    admin_telegram_ids: list[int] = None

    def __post_init__(self) -> None:
        self.llm_provider = (self.llm_provider or "echo").strip().lower()
        self.llm_model = (self.llm_model or "").strip()
        self.ollama_base_url = (self.ollama_base_url or "").strip()
        self.chunk_splitter_mode = (self.chunk_splitter_mode or "smart").strip().lower() or "smart"
        self.chunk_mmr_lambda = max(0.0, min(1.0, float(self.chunk_mmr_lambda)))
        self.entity_mmr_lambda = max(0.0, min(1.0, float(self.entity_mmr_lambda)))
        self.admin_telegram_ids = _as_int_list(os.getenv("ADMIN_TELEGRAM_IDS", ""))


# Общий экземпляр настроек приложения, используемый во всех модулях.
settings = Settings()

