from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    # Нормализуем путь и сразу завершаем с ошибкой, если файл не найден.
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Р¤Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ: {pdf_path}")

    # Читаем все страницы и объединяем извлеченный текст.
    text_parts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def read_pdf_text(pdf_path: str) -> str:
    # Обратносовместимый алиас для старых импортов.
    return extract_text_from_pdf(pdf_path)

