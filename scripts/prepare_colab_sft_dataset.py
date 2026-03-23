from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "обучение ллм"
OUT_DIR = TRAIN_DIR / "colab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    TRAIN_DIR / "dataset_eval.json",
    TRAIN_DIR / "second_dataset.json",
    TRAIN_DIR / "third.json",
]

SYSTEM_PROMPT = (
    "Ты отвечаешь по первой книге Гарри Поттера. "
    "Отвечай точно, по фактам из контекста вопроса. "
    "Пиши на русском, без выдумок."
)


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _normalize_item(item: dict[str, Any], src: str) -> dict[str, Any] | None:
    q = str(item.get("question") or "").strip()
    a = str(item.get("expected_answer") or "").strip()
    if not q or not a:
        return None

    book = item.get("source_book", "")
    chapter = item.get("source_chapter", "")
    page = item.get("source_page", "")

    meta_parts = []
    if book != "":
        meta_parts.append(f"Книга {book}")
    if chapter != "":
        meta_parts.append(f"Глава {chapter}")
    if page != "":
        meta_parts.append(f"Страница {page}")

    source_meta = " • ".join(meta_parts) if meta_parts else "Источник не указан"

    instruction = (
        f"{SYSTEM_PROMPT}\n"
        f"Источник: {source_meta}.\n"
        f"Вопрос: {q}"
    )

    # Формат для SFTTrainer dataset_text_field="text"
    text = (
        "<s>[INST] "
        f"{instruction} "
        "[/INST] "
        f"{a}</s>"
    )

    return {
        "id": str(item.get("id") or f"{src}-{random.randint(100000, 999999)}"),
        "source_dataset": src,
        "question": q,
        "answer": a,
        "source_book": book,
        "source_chapter": chapter,
        "source_page": page,
        "text": text,
    }


def main() -> None:
    random.seed(42)
    all_rows: list[dict[str, Any]] = []

    for ds_path in DATASETS:
        if not ds_path.exists():
            continue
        src_name = ds_path.stem
        for item in _read_json(ds_path):
            row = _normalize_item(item, src_name)
            if row is not None:
                all_rows.append(row)

    # Удаляем дубли по вопросу
    uniq: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        key = row["question"].strip().lower()
        if key not in uniq:
            uniq[key] = row

    rows = list(uniq.values())
    random.shuffle(rows)

    split_idx = int(len(rows) * 0.9)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    combined_path = OUT_DIR / "combined_sft.jsonl"
    train_path = OUT_DIR / "train_sft.jsonl"
    val_path = OUT_DIR / "val_sft.jsonl"
    stats_path = OUT_DIR / "dataset_stats.json"

    def dump_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in payload:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump_jsonl(combined_path, rows)
    dump_jsonl(train_path, train_rows)
    dump_jsonl(val_path, val_rows)

    stats = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "datasets_found": [p.name for p in DATASETS if p.exists()],
        "output_files": {
            "combined": str(combined_path),
            "train": str(train_path),
            "val": str(val_path),
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
