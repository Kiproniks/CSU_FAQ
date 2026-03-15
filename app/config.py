from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "echo")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_timeout_sec: int = int(os.getenv("LLM_TIMEOUT_SEC", "30"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    chunk_collection: str = os.getenv("CHUNK_COLLECTION", "harry_potter_collection")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", "3"))
    top_k_entities: int = int(os.getenv("TOP_K_ENTITIES", "3"))
    bootstrap_entity_from_chroma: bool = _as_bool(
        os.getenv("BOOTSTRAP_ENTITY_FROM_CHROMA", "1"),
        default=True,
    )


settings = Settings()
