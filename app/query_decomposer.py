from typing import List
from app.llm_service import LLMService

class QueryDecomposer:
    def __init__(self):
        self.llm = LLMService()

    def decompose(self, query: str) -> List[str]:
        """МАКСИМАЛЬНО СТРОГИЙ decomposer — почти никогда не разбивает простой вопрос."""
        prompt = f"""Ты — очень строгий помощник. 
Твоя задача: разбить вопрос ТОЛЬКО если в нём явно несколько независимых тем.

Правила:
- Если вопрос про одного персонажа/предмет — верни ровно ОДИН под-вопрос (оригинальный текст).
- Разбивай только если есть слова "и", "почему", "а также", "?" больше одного раза или явно две разные темы.
- Отвечай ТОЛЬКО под-вопросами, каждый на новой строке, без номеров, тире и лишнего текста.

Вопрос: {query}

Под-вопросы:"""

        response = self.llm.generate_answer(query=prompt, context="")

        lines = [line.strip() for line in response.answer.split('\n') 
                 if line.strip() and len(line.strip()) > 8]

        # Если decomposer вернул ерунду или больше 1 — берём только первый
        if not lines or len(lines) > 3:
            return [query]

        return lines