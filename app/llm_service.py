import os
from dataclasses import dataclass
import requests
from app.config import settings

@dataclass
class LLMResponse:
    answer: str
    provider: str
    model: str

class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.ollama_url = settings.ollama_base_url.rstrip('/') + "/api/generate"
        self.timeout = settings.llm_timeout_sec
        
        print(f"[DEBUG] Ollama URL: {self.ollama_url}")
        print(f"[DEBUG] Model: {self.model} | Timeout: {self.timeout}s")

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Ты — эксперт по книгам Гарри Поттера.
            Отвечай **строго на том же языке**, на котором задан вопрос:
            - Если вопрос на русском — ответ полностью на русском.
            - Если вопрос на английском — ответ полностью на английском.
            Никакого микса языков, никакого транслита, никаких английских слов в русском ответе.

            Будь вежливым, естественным и подробным.
            Используй только контекст ниже.
            Если информации недостаточно — честно скажи.

            CONTEXT:
            {context}

            ВОПРОС: {query}

            ОТВЕТ:"""

    def generate_answer(self, query: str, context: str) -> LLMResponse:
        if self.provider == "ollama":
            return self._generate_ollama(query, context)
        return self._echo_fallback("Неизвестный провайдер")

    def _generate_ollama(self, query: str, context: str) -> LLMResponse:
        prompt = self._build_prompt(query, context)
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4, "top_p": 0.95}
                },
                timeout=self.timeout,
                proxies={"http": None, "https": None}   # ← БЛОКИРУЕМ ПРОКСИ
            )
            response.raise_for_status()
            data = response.json()
            answer = (data.get("response") or "").strip()

            return LLMResponse(
                answer=f"✅ **Ответ от локальной LLM** (`{self.model}`)\n\n{answer}",
                provider="ollama",
                model=self.model
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[Ollama Error] {error_msg}")
            return self._echo_fallback(error_msg)

    @staticmethod
    def _echo_fallback(reason: str = "") -> LLMResponse:
        return LLMResponse(
            answer=f"❌ LLM недоступна.\nПричина: {reason}\n\nПоказываю лучшие фрагменты...",
            provider="echo",
            model="fallback"
        )