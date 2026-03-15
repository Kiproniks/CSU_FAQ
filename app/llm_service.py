from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from app.config import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class LLMResponse:
    answer: str
    provider: str
    model: str


class LLMService:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.client: Optional[OpenAI] = None
        if self.provider == "openai" and settings.openai_api_key and OpenAI is not None:
            self.client = OpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            "You are an assistant for a university RAG project. "
            "Answer only from the provided context. "
            "If context is insufficient, say it explicitly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

    def _generate_openai(self, query: str, context: str) -> Optional[LLMResponse]:
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
        prompt = self._build_prompt(query, context)
        try:
            response = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
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
            return None

    @staticmethod
    def _echo_fallback(query: str, context: str, reason: str = "") -> LLMResponse:
        prefix = "LLM fallback mode (echo)."
        if reason:
            prefix = f"{prefix} Reason: {reason}"

        return LLMResponse(
            answer=(
                f"{prefix}\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{context}"
            ),
            provider="echo",
            model="debug",
        )

    def generate_answer(self, query: str, context: str) -> LLMResponse:
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
