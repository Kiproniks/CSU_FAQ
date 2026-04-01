from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

CHAPTER_RE = re.compile(r"\bглава\s+([0-9]{1,3}|[ivxlcdm]{1,8})\b", re.IGNORECASE)
SOURCE_BOOK_RE = re.compile(r"^(?:book[_\-\s]*)?([0-9]{1,2})(?:[_\-\s]|$)", re.IGNORECASE)


class SourceAttributionFormatter:
    """Форматирует человекочитаемую подпись источника: книга/глава/страница."""

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else None
        except Exception:
            return None

    @staticmethod
    def _roman_to_int(value: str) -> int | None:
        roman = (value or "").strip().upper()
        if not roman:
            return None
        numbers = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        prev = 0
        for ch in reversed(roman):
            current = numbers.get(ch, 0)
            if current == 0:
                return None
            if current < prev:
                total -= current
            else:
                total += current
            prev = current
        return total if total > 0 else None

    @classmethod
    def _book_number(cls, metadata: Dict[str, Any], source: str) -> int | None:
        for key in ("book", "book_number"):
            value = cls._as_int(metadata.get(key))
            if value is not None:
                return value

        stem = Path(source or "").stem
        match = SOURCE_BOOK_RE.search(stem)
        if not match:
            return None
        return cls._as_int(match.group(1))

    @classmethod
    def _chapter_number(cls, metadata: Dict[str, Any], text: str) -> int | None:
        for key in ("chapter", "chapter_number"):
            value = metadata.get(key)
            as_int = cls._as_int(value)
            if as_int is not None:
                return as_int
            if isinstance(value, str):
                roman = cls._roman_to_int(value)
                if roman is not None:
                    return roman

        chunk_text = str(text or "")
        match = CHAPTER_RE.search(chunk_text)
        if not match:
            return None
        token = (match.group(1) or "").strip()
        as_int = cls._as_int(token)
        if as_int is not None:
            return as_int
        return cls._roman_to_int(token)

    @classmethod
    def _page_number(cls, metadata: Dict[str, Any]) -> int | None:
        for key in ("page", "page_number"):
            value = cls._as_int(metadata.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _parts(cls, hit: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(hit.get("metadata") or {})
        source = str(hit.get("source") or metadata.get("source") or "unknown")
        text = str(hit.get("text") or "")

        book = cls._book_number(metadata, source)
        chapter = cls._chapter_number(metadata, text)
        page = cls._page_number(metadata)

        return {
            "source": source,
            "book": book,
            "chapter": chapter,
            "page": page,
        }

    @classmethod
    def format_short(cls, hit: Dict[str, Any]) -> str:
        parts = cls._parts(hit)
        labels: list[str] = []
        if parts["book"] is not None:
            labels.append(f"Книга {parts['book']}")
        if parts["chapter"] is not None:
            labels.append(f"Глава {parts['chapter']}")
        if parts["page"] is not None:
            labels.append(f"Стр. {parts['page']}")

        if labels:
            return " • ".join(labels)
        return str(parts["source"])

    @classmethod
    def to_card(cls, hit: Dict[str, Any]) -> Dict[str, Any]:
        parts = cls._parts(hit)
        title = cls.format_short(hit)
        source = str(parts["source"])

        return {
            "title": title,
            "source": source,
            "book": parts["book"],
            "chapter": parts["chapter"],
            "page": parts["page"],
            "score": float(hit.get("score", 0.0)),
            "text": str(hit.get("text", "")),
            "entities": hit.get("entities", []) if isinstance(hit.get("entities"), list) else [],
        }
