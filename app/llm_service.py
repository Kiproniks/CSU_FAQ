from __future__ import annotations

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
    def __init__(self) -> None:
        # Выбор бэкенда читаем один раз; маршрутизация делается в generate_answer().
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.client: Optional[OpenAIClient] = None
        if self.provider == "openai" and settings.openai_api_key and OpenAIClass is not None:
            self.client = OpenAIClass(api_key=settings.openai_api_key)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            "Ты ассистент университетского RAG-проекта.\n"
            "Правила ответа:\n"
            "1) Отвечай только на русском языке.\n"
            "2) Используй только факты из блока CONTEXT.\n"
            "3) Не упоминай retrieval, TF-IDF, chunk/entity, если об этом не спросили.\n"
            "4) Не выдумывай факты. Если данных не хватает, напиши: "
            "\"Недостаточно данных в найденных фрагментах.\"\n"
            "5) Дай связный ответ на 3-6 предложений.\n\n"
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
        prompt = self._build_prompt(query, context)
        try:
            base_url = settings.ollama_base_url.rstrip("/")
            hostname = (urlparse(base_url).hostname or "").lower()
            # Для localhost отключаем прокси из окружения: они часто дают 502 на локальном Ollama.
            session = requests.Session()
            if hostname in {"localhost", "127.0.0.1", "::1"}:
                session.trust_env = False

            response = session.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                    },
                },
                timeout=settings.llm_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            text = (payload.get("response") or "").strip()
            if not text:
                return None

            return LLMResponse(
                answer=text,
                provider="ollama",
                model=self.model,
            )
        except Exception:
            # Любая ошибка сети/модели уходит во fallback без падения приложения.
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
                return result
            return self._echo_fallback(query, context, reason="Ollama is not available")

        return self._echo_fallback(query, context)
