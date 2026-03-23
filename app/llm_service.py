from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import requests

from app.config import settings

if TYPE_CHECKING:
    # Тип клиента нужен только для статической проверки.
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any

try:
    # Библиотека OpenAI опциональна: при отсутствии пакета используем безопасный fallback.
    from openai import OpenAI as OpenAIClass
except Exception:  # pragma: no cover
    OpenAIClass = None


@dataclass
class LLMResponse:
    # Единый формат результата для всех LLM-бэкендов.
    answer: str
    provider: str
    model: str


class LLMService:
    _ollama_bootstrap_attempted: bool = False

    def __init__(self) -> None:
        # Выбор бэкенда читаем один раз; маршрутизация делается в generate_answer().
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.client: Optional[OpenAIClient] = None
        # Короткий circuit-breaker: не штурмовать Ollama, если она недавно упала.
        self._ollama_unavailable_until: float = 0.0
        if self.provider == "ollama":
            self._ensure_ollama_server()
        if self.provider == "openai" and settings.openai_api_key and OpenAIClass is not None:
            self.client = OpenAIClass(api_key=settings.openai_api_key)

    @staticmethod
    def _contains_latin(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text or ""))

    def _rewrite_cyrillic_ollama(self, text: str) -> str:
        # Второй проход: просим модель переписать текст без латиницы.
        try:
            base_url = settings.ollama_base_url.rstrip("/")
            hostname = (urlparse(base_url).hostname or "").lower()
            prompt = (
                "Перепиши текст строго на русском языке.\n"
                "Используй только кириллицу, не используй латиницу вообще.\n"
                "Сохрани смысл исходного текста.\n\n"
                f"ТЕКСТ:\n{text}"
            )
            with requests.Session() as session:
                if hostname in {"localhost", "127.0.0.1", "::1"}:
                    session.trust_env = False
                response = session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "top_p": 0.9,
                        },
                    },
                    timeout=(3, min(20, max(8, settings.llm_timeout_sec))),
                )
                response.raise_for_status()
                payload = response.json()
                return str(payload.get("response") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _ollama_tags_ready(timeout_sec: float = 1.5) -> bool:
        base_url = settings.ollama_base_url.rstrip("/")
        hostname = (urlparse(base_url).hostname or "").lower()
        try:
            with requests.Session() as session:
                if hostname in {"localhost", "127.0.0.1", "::1"}:
                    session.trust_env = False
                response = session.get(f"{base_url}/api/tags", timeout=timeout_sec)
                response.raise_for_status()
            return True
        except Exception:
            return False

    @classmethod
    def _ensure_ollama_server(cls) -> None:
        # Авто-подъем локального Ollama, если он не запущен.
        if cls._ollama_tags_ready(timeout_sec=1.5):
            return
        if cls._ollama_bootstrap_attempted:
            return

        cls._ollama_bootstrap_attempted = True
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as exc:
            print(f"[LLMService] failed to start ollama serve: {exc}")
            return

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if cls._ollama_tags_ready(timeout_sec=1.5):
                print("[LLMService] ollama server is ready.")
                return
            time.sleep(1.0)
        print("[LLMService] ollama server is still unavailable after bootstrap.")

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            "Ты ассистент университетского проекта вопросов и ответов.\n"
            "Правила ответа:\n"
            "1) Отвечай только на русском языке, используя только кириллицу.\n"
            "2) Не используй латиницу в ответе (исключений нет).\n"
            "3) Используй только факты из блока CONTEXT.\n"
            "4) Не упоминай технические детали поиска, если об этом не спросили.\n"
            "5) Не выдумывай факты. Если данных не хватает, напиши: "
            "\"Недостаточно данных в найденных фрагментах.\"\n"
            "6) Дай связный ответ на 3-6 предложений.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}"
        )

    def _generate_openai(self, query: str, context: str) -> Optional[LLMResponse]:
        # Ветка OpenAI: используем Responses API и приводим ответ к LLMResponse.
        if self.client is None:
            return None

        prompt = self._build_prompt(query, context)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return LLMResponse(
            answer=response.output_text.strip(),
            provider="openai",
            model=self.model,
        )

    def _generate_ollama(self, query: str, context: str) -> Optional[LLMResponse]:
        # Ветка Ollama: локальный HTTP-вызов с низкой temperature для стабильных ответов.
        now = time.monotonic()
        if now < self._ollama_unavailable_until:
            return None

        prompt = self._build_prompt(query, context)
        try:
            base_url = settings.ollama_base_url.rstrip("/")
            hostname = (urlparse(base_url).hostname or "").lower()
            # Для localhost отключаем прокси из окружения: они часто дают 502 на локальном Ollama.
            with requests.Session() as session:
                if hostname in {"localhost", "127.0.0.1", "::1"}:
                    session.trust_env = False

                response = session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_predict": 120,
                        },
                    },
                    timeout=(3, settings.llm_timeout_sec),
                )
                response.raise_for_status()
                payload = response.json()
                text = (payload.get("response") or "").strip()
                if not text:
                    self._ollama_unavailable_until = time.monotonic() + 20
                    return None

                self._ollama_unavailable_until = 0.0
                return LLMResponse(
                    answer=text,
                    provider="ollama",
                    model=self.model,
                )
        except Exception:
            # Любая ошибка сети/модели уходит во fallback без падения приложения.
            self._ollama_unavailable_until = time.monotonic() + 20
            return None

    @staticmethod
    def _echo_fallback(query: str, context: str, reason: str = "") -> LLMResponse:
        # Безопасный fallback без публикации длинного контекста в ответ пользователю.
        message = "LLM fallback mode (echo)."
        if reason:
            message = f"{message} Reason: {reason}"
        message = f"{message}\n\nLLM временно недоступна. Повторите запрос позже."
        return LLMResponse(
            answer=message,
            provider="echo",
            model="debug",
        )

    def generate_answer(self, query: str, context: str) -> LLMResponse:
        # Маршрутизатор провайдеров с устойчивой цепочкой fallback.
        if self.provider == "openai":
            result = self._generate_openai(query, context)
            if result is not None:
                return result
            return self._echo_fallback(query, context, reason="OpenAI is not configured")

        if self.provider == "ollama":
            result = self._generate_ollama(query, context)
            if result is not None:
                if self._contains_latin(result.answer):
                    rewritten = self._rewrite_cyrillic_ollama(result.answer)
                    if rewritten and not self._contains_latin(rewritten):
                        result.answer = rewritten
                return result
            return self._echo_fallback(query, context, reason="Ollama is not available")

        return self._echo_fallback(query, context)
