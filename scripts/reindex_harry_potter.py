from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Делаем корень проекта импортируемым при прямом запуске скрипта.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.pdf_utils import extract_text_from_pdf
from app.rag_pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    # Параметры CLI для одноразовой переиндексации.
    parser = argparse.ArgumentParser(description="Reindex Harry Potter PDFs")
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
    # Создаем пайплайн один раз и при необходимости очищаем выбранные индексы.
    args = parse_args()
    books_dir = Path(args.books_dir)

    pipeline = RAGPipeline()

    if args.clear_chunk_index:
        pipeline.chunk_engine.clear()
        print("Chunk index cleared.")

    if args.clear_entity_index:
        pipeline.entity_engine.clear()
        print("Entity index cleared.")

    # Проходим по всем PDF и индексируем каждый документ в оба ретривера.
    pdf_files = sorted(books_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {books_dir}")
        return

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))

        if not text.strip():
            print("  Skipped: empty text")
            continue

        pipeline.index_document(
            text=text,
            doc_id=pdf_path.stem,
            metadata={"source": pdf_path.name},
        )
        print("  Indexed")

    print("\nReindexing completed.")


if __name__ == "__main__":
    main()

