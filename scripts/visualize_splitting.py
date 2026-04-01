from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
# Делаем корень проекта импортируемым при прямом запуске скрипта.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.EntityBased import EntityBased
from app.pdf_utils import extract_text_from_pdf


def parse_args() -> argparse.Namespace:
    # Параметры CLI для артефактов диагностики разбиения.
    parser = argparse.ArgumentParser(
        description="Visualize how texts are split for ChunkBased and EntityBased"
    )
    parser.add_argument(
        "--books-dir",
        default=str(BASE_DIR / "harry_potter"),
        help="Directory with PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "split_visualization"),
        help="Output folder for split visualization artifacts",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap in characters",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="How many PDFs to process (0 = all)",
    )
    return parser.parse_args()


def write_chunks_text(path: Path, chunks: list[str], title: str) -> None:
    # Текстовый дамп чанков для быстрой ручной проверки.
    lines = [f"{title}\n", f"Total chunks: {len(chunks)}\n"]
    for idx, chunk in enumerate(chunks):
        lines.append(f"\n=== Chunk {idx} | len={len(chunk)} ===\n")
        lines.append(chunk)
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_chunks_json(path: Path, chunks: list[str], strategy: str, source: str) -> None:
    # Машиночитаемый дамп чанков для инструментов и отчетов.
    payload = {
        "source": source,
        "strategy": strategy,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "index": idx,
                "length": len(chunk),
                "text": chunk,
            }
            for idx, chunk in enumerate(chunks)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_lengths(path: Path, chunk_lengths: list[int], entity_lengths: list[int], title: str) -> None:
    # Сравниваем распределения длины чанков у двух стратегий.
    plt.figure(figsize=(12, 6))
    plt.plot(chunk_lengths, label="ChunkBased", marker="o", linewidth=1)
    plt.plot(entity_lengths, label="EntityBased", marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Chunk index")
    plt.ylabel("Chunk length (chars)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_summary_line(name: str, values: list[int]) -> str:
    # Компактная строка статистики для SUMMARY.md.
    if not values:
        return f"{name}: no chunks"
    return (
        f"{name}: count={len(values)}, min={min(values)}, max={max(values)}, "
        f"avg={statistics.mean(values):.1f}"
    )


def split_for_entity(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Для оригинального tro-part EntityBased отдельного _split_text нет.
    # Используем совместимый внешний сплиттер на базе ChunkBased.
    split_fn = getattr(EntityBased, "_split_text", None)
    if callable(split_fn):
        return split_fn(text, chunk_size, overlap)
    return ChunkBased._split_text(text, chunk_size, overlap)


def generate_split_visualization(
    books_dir: Path,
    output_dir: Path,
    chunk_size: int = 1200,
    overlap: int = 200,
    limit: int = 2,
) -> Path:
    # Генерируем полный пакет артефактов разбиения в output_dir.
    chunk_dir = output_dir / "chunk_based"
    entity_dir = output_dir / "entity_based"
    plots_dir = output_dir / "plots"

    chunk_dir.mkdir(parents=True, exist_ok=True)
    entity_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Выбираем входные PDF (все или ограниченный набор).
    pdf_files = sorted(books_dir.glob("*.pdf"))
    if limit > 0:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        print(f"No PDF files found in: {books_dir}")
        return

    summary_lines = [
        "# Split Visualization Summary\n",
        f"Chunk size: {chunk_size}, overlap: {overlap}\n",
    ]

    for pdf_path in pdf_files:
        # Разбиваем один документ обоими методами и сохраняем артефакты.
        print(f"Processing split visualization for: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            print(f"  Skipped empty text: {pdf_path.name}")
            continue

        chunk_splits = ChunkBased._split_text(text, chunk_size, overlap)
        entity_splits = split_for_entity(text, chunk_size, overlap)

        stem = pdf_path.stem

        write_chunks_text(
            chunk_dir / f"{stem}_chunks.txt",
            chunk_splits,
            title=f"ChunkBased split for {pdf_path.name}",
        )
        write_chunks_json(
            chunk_dir / f"{stem}_chunks.json",
            chunk_splits,
            strategy="chunk-based",
            source=pdf_path.name,
        )

        write_chunks_text(
            entity_dir / f"{stem}_chunks.txt",
            entity_splits,
            title=f"EntityBased split for {pdf_path.name}",
        )
        write_chunks_json(
            entity_dir / f"{stem}_chunks.json",
            entity_splits,
            strategy="entity-based",
            source=pdf_path.name,
        )

        chunk_lengths = [len(c) for c in chunk_splits]
        entity_lengths = [len(c) for c in entity_splits]

        plot_lengths(
            plots_dir / f"{stem}_lengths.png",
            chunk_lengths,
            entity_lengths,
            title=f"Chunk length comparison: {pdf_path.name}",
        )

        summary_lines.append(f"\n## {pdf_path.name}\n")
        summary_lines.append(build_summary_line("ChunkBased", chunk_lengths) + "\n")
        summary_lines.append(build_summary_line("EntityBased", entity_lengths) + "\n")

    (output_dir / "SUMMARY.md").write_text("".join(summary_lines), encoding="utf-8")
    print(f"Split visualization saved to: {output_dir}")
    return output_dir


def main() -> None:
    # Точка входа CLI.
    args = parse_args()
    generate_split_visualization(
        books_dir=Path(args.books_dir),
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

