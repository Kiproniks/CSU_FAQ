from __future__ import annotations

from typing import Any, Dict, List

# Фиксированная база FAQ: всегда показываем эти 10 пар вопрос/ответ.
FIXED_FAQ: List[Dict[str, str]] = [
    {
        "question": "Кто такой Гарри Поттер?",
        "answer": "Гарри Поттер — главный герой серии, мальчик-волшебник, который в младенчестве выжил после нападения Волан-де-Морта.",
    },
    {
        "question": "Почему Гарри выжил после атаки Волан-де-Морта?",
        "answer": "Потому что его мать Лили пожертвовала собой ради него, и эта жертва создала сильную магическую защиту.",
    },
    {
        "question": "Кто убил родителей Гарри?",
        "answer": "Родителей Гарри, Джеймса и Лили Поттер, убил Волан-де-Морт.",
    },
    {
        "question": "Почему Гарри жил у Дурслей?",
        "answer": "Он жил у Дурслей, потому что они были его единственными родственниками по линии матери, и у них он находился под защитой.",
    },
    {
        "question": "Кто такой Волан-де-Морт?",
        "answer": "Волан-де-Морт — главный злодей серии, тёмный волшебник, который хотел стать бессмертным и захватить власть.",
    },
    {
        "question": "Почему Снейп ненавидел Гарри?",
        "answer": "Снейп был похож на человека, который ненавидит Гарри, потому что Гарри напоминал ему его отца, Джеймса Поттера. Но на самом деле его чувства были намного сложнее.",
    },
    {
        "question": "На какой факультет попал Гарри?",
        "answer": "Гарри попал на факультет Гриффиндор.",
    },
    {
        "question": "Кто лучшие друзья Гарри?",
        "answer": "Лучшие друзья Гарри — Рон Уизли и Гермиона Грейнджер.",
    },
    {
        "question": "Что такое Хогвартс?",
        "answer": "Хогвартс — это школа чародейства и волшебства, где учатся юные волшебники.",
    },
    {
        "question": "Чем закончилась история Гарри Поттера?",
        "answer": "История закончилась победой Гарри над Волан-де-Мортом и тем, что после войны герои начали мирную жизнь.",
    },
]


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def build_faq_rows(dynamic_rows: List[Dict[str, Any]], dynamic_limit: int = 10) -> List[Dict[str, str]]:
    """
    Собирает итоговый FAQ:
    - 10 фиксированных вопросов всегда;
    - до 10 динамических из БД по последним пользовательским вопросам.
    """
    result: List[Dict[str, str]] = []

    for row in FIXED_FAQ:
        q = " ".join(str(row.get("question", "")).split())
        a = " ".join(str(row.get("answer", "")).split())
        if q and a:
            result.append({"question": q, "answer": a})

    used = {_normalize_question(x.get("question", "")) for x in result}
    appended = 0

    for row in dynamic_rows or []:
        if appended >= max(1, int(dynamic_limit)):
            break

        q = " ".join(str((row or {}).get("question", "")).split())
        if not q:
            continue

        normalized = _normalize_question(q)
        if not normalized or normalized in used:
            continue

        saved_answer = " ".join(str((row or {}).get("answer", "")).split())
        if not saved_answer:
            saved_answer = "Ответ появится после следующего запроса этого вопроса в QA."

        result.append({"question": q, "answer": saved_answer})
        used.add(normalized)
        appended += 1

    return result
