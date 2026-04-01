from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    # Возвращаем список страниц с 1-based нумерацией.
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    pages: list[dict] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            pages.append(
                {
                    "page": index,
                    "text": page.get_text() or "",
                }
            )
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    # Объединяем текст всех страниц в один блок (совместимость со старым API).
    return "\n".join(item["text"] for item in extract_pages_from_pdf(pdf_path))


def read_pdf_text(pdf_path: str) -> str:
    # Обратносовместимый алиас для старых импортов.
    return extract_text_from_pdf(pdf_path)
