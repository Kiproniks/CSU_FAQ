from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.pdf_utils import extract_pages_from_pdf
from app.rag_pipeline import RAGPipeline

BOOK_RE = re.compile(r"^(?:book[_\-\s]*)?([0-9]{1,2})(?:[_\-\s]|$)", re.IGNORECASE)
CHAPTER_RE = re.compile(r"\bглава\s+([0-9]{1,3}|[ivxlcdm]{1,8})\b", re.IGNORECASE)


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


def _book_number_from_filename(filename: str) -> int | None:
    stem = Path(filename or "").stem
    match = BOOK_RE.search(stem)
    if not match:
        return None
    try:
        parsed = int(match.group(1))
        return parsed if parsed > 0 else None
    except Exception:
        return None


def _chapter_from_text(text: str) -> int | None:
    match = CHAPTER_RE.search(text or "")
    if not match:
        return None
    token = (match.group(1) or "").strip()
    try:
        parsed = int(token)
        return parsed if parsed > 0 else None
    except Exception:
        return _roman_to_int(token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reindex Harry Potter PDFs (with page metadata)")
    parser.add_argument(
        "--books-dir",
        default=str(BASE_DIR / "harry_potter"),
        help="Directory with PDF books",
    )
    parser.add_argument(
        "--clear-chunk-index",
        action="store_true",
        help="Clear chunk index before reindexing",
    )
    parser.add_argument(
        "--clear-entity-index",
        action="store_true",
        help="Clear in-memory entity index before reindexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books_dir = Path(args.books_dir)

    pipeline = RAGPipeline()

    if args.clear_chunk_index:
        pipeline.chunk_engine.clear()
        print("Chunk index cleared.")

    if args.clear_entity_index:
        pipeline.entity_engine.clear()
        print("Entity index cleared.")

    pdf_files = sorted(books_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {books_dir}")
        return

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        pages = extract_pages_from_pdf(str(pdf_path))
        if not pages:
            print("  Skipped: no pages")
            continue

        book_number = _book_number_from_filename(pdf_path.name)
        current_chapter: int | None = None
        indexed_pages = 0

        for item in pages:
            page_number = int(item.get("page", 0) or 0)
            page_text = str(item.get("text", "") or "")
            if page_number <= 0 or not page_text.strip():
                continue

            chapter = _chapter_from_text(page_text)
            if chapter is not None:
                current_chapter = chapter

            metadata = {"source": pdf_path.name, "page": page_number}
            if book_number is not None:
                metadata["book"] = book_number
            if current_chapter is not None:
                metadata["chapter"] = current_chapter

            pipeline.index_document(
                text=page_text,
                doc_id=f"{pdf_path.stem}_page_{page_number}",
                metadata=metadata,
            )
            indexed_pages += 1

        print(f"  Indexed pages: {indexed_pages}")

    print("\nReindexing completed.")


if __name__ == "__main__":
    main()
