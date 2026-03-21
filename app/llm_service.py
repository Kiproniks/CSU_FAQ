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
        """Универсальный промпт — без привязки к Гарри Поттеру"""
        return f"""Ты — точный и честный помощник по базе знаний.

ПРАВИЛА (обязательно соблюдай):
- Отвечай **строго на том же языке**, на котором задан вопрос.
- Если вопрос на русском — ответ полностью на русском.
- Если вопрос на английском — ответ полностью на английском.
- Никакого микса языков, никакого транслита, никаких английских слов в русском ответе.
- Используй только информацию из контекста ниже.
- Если в контексте нет ответа — честно скажи: "В базе нет информации по этому вопросу".
- Будь кратким, естественным и по делу.

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
                    "options": {
                        "temperature": 0.3,      # ниже = меньше фантазий
                        "top_p": 0.95,
                        "num_ctx": 4096
                    }
                },
                timeout=self.timeout,
                proxies={"http": None, "https": None}
            )
            response.raise_for_status()
            data = response.json()
            raw_answer = (data.get("response") or "").strip()

            return LLMResponse(
                answer=raw_answer,                    # ← УБРАЛИ дублирующуюся шапку
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
            answer=f"❌ LLM недоступна.\nПричина: {reason}",
            provider="echo",
            model="fallback"
        )