from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    text_parts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def read_pdf_text(pdf_path: str) -> str:
    return extract_text_from_pdf(pdf_path)