from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

import site
import sys

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
venv_site = BASE_DIR / "venv" / "Lib" / "site-packages"
if venv_site.exists():
    site.addsitedir(str(venv_site))

from ChunkBased.ChunkBased import ChunkBased  # noqa: E402
from app.config import settings  # noqa: E402
from app.pdf_utils import extract_text_from_pdf  # noqa: E402
from scripts import chunk_settings_benchmark as bench  # noqa: E402


@dataclass
class Setting:
    chunk_size: int = 420
    overlap: int = 70
    top_k: int = 3
    splitter_mode: str = "smart"
    mmr_lambda: float = 0.72

    @property
    def name(self) -> str:
        return f"s{self.chunk_size}_o{self.overlap}_k{self.top_k}_msm_mmr{self.mmr_lambda:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Important tests: before(one book) vs after(all books)")
    parser.add_argument(
        "--questions-file",
        default=str(BASE_DIR / "обучение ллм" / "ВАЖНЫЕ ТЕСТЫ" / "questions_30_hp.json"),
    )
    parser.add_argument("--books-dir", default=str(BASE_DIR / "harry_potter"))
    parser.add_argument("--report-root", default=str(BASE_DIR / "обучение ллм" / "ВАЖНЫЕ ТЕСТЫ"))
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        return []
    out: list[dict[str, str]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        a = str(row.get("expected_answer") or "").strip()
        if q and a:
            out.append({"question": q, "expected_answer": a})
    return out


def load_books(books_dir: Path) -> list[tuple[str, str]]:
    books: list[tuple[str, str]] = []
    for pdf in sorted(books_dir.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf))
        if text.strip():
            books.append((pdf.stem, text))
    return books


def evaluate(setting: Setting, books: list[tuple[str, str]], questions: list[dict[str, str]], prefix: str, judge: bench.SemanticJudge):
    engine = ChunkBased(
        chunk_size=setting.chunk_size,
        overlap=setting.overlap,
        collection_name=f"{prefix}_{int(time.time()*1000)}",
        chroma_path=settings.chroma_path,
        splitter_mode=setting.splitter_mode,
        mmr_lambda=setting.mmr_lambda,
        min_chunk_floor=80,
    )
    details: list[list[str]] = []
    s10: list[float] = []
    s01: list[float] = []

    try:
        for doc_id, text in books:
            engine.add_document(text=text, doc_id=doc_id, metadata={"source": doc_id})

        for idx, item in enumerate(tqdm(questions, desc=prefix, leave=False), start=1):
            q = item["question"]
            expected = item["expected_answer"]
            hits = engine.search(q, top_k=setting.top_k)
            generated = bench.extractive_answer(question=q, hits=hits, semantic_model=judge.model, max_sentences=5)
            v01, v10, comment = judge.score(question=q, expected=expected, generated=generated)
            s10.append(v10)
            s01.append(v01)

            source = ""
            if hits:
                top_payload = hits[0][0] or {}
                source = str((top_payload.get("metadata") or {}).get("source", ""))

            details.append([setting.name, str(idx), q, expected, generated, f"{v10:.2f}", comment, source])
    finally:
        engine.clear()

    avg10 = sum(s10) / len(s10) if s10 else 0.0
    avg01 = sum(s01) / len(s01) if s01 else 0.0
    return avg10, avg01, details


def write_xlsx(path: Path, setting: Setting, avg10: float, avg01: float, details: list[list[str]]) -> None:
    summary = [[
        "rank",
        "setting",
        "chunk_size",
        "overlap",
        "top_k",
        "splitter",
        "mmr_lambda",
        "avg_score_10",
        "avg_score_01",
    ], [
        "1",
        setting.name,
        str(setting.chunk_size),
        str(setting.overlap),
        str(setting.top_k),
        setting.splitter_mode,
        f"{setting.mmr_lambda:.2f}",
        f"{avg10:.3f}",
        f"{avg01:.4f}",
    ]]
    header = [["setting", "question_index", "question", "expected_answer", "generated_answer", "score_10", "comment", "top_source"]]
    bench._write_xlsx(path, summary, header + details)


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions_file)
    books_dir = Path(args.books_dir)
    report_root = Path(args.report_root)
    before_dir = report_root / "до"
    after_dir = report_root / "после"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    books = load_books(books_dir)
    if len(books) < 1:
        raise RuntimeError(f"Нет PDF в {books_dir}")
    if not questions:
        raise RuntimeError(f"Нет валидных вопросов в {questions_path}")

    setting = Setting()
    judge = bench.SemanticJudge()

    ts = time.strftime("%Y%m%d_%H%M%S")

    before_books = books
    before10, before01, before_details = evaluate(setting, before_books, questions, "important_before", judge)
    before_xlsx = before_dir / f"important_before_30_{ts}.xlsx"
    write_xlsx(before_xlsx, setting, before10, before01, before_details)

    after10, after01, after_details = evaluate(setting, books, questions, "important_after", judge)
    after_xlsx = after_dir / f"important_after_30_{ts}.xlsx"
    write_xlsx(after_xlsx, setting, after10, after01, after_details)

    summary_path = report_root / f"summary_30_{ts}.txt"
    lines = [
        "ВАЖНЫЕ ТЕСТЫ: сравнение до/после",
        "",
        "Что сравнили:",
        f"- До: все книги ({len(books)} шт.)",
        f"- После: все книги ({len(books)} шт.)",
        f"- Вопросов: {len(questions)}",
        f"- Конфиг: {setting.name}",
        "",
        f"До avg_score_10:    {before10:.3f}",
        f"После avg_score_10: {after10:.3f}",
        f"Дельта:             {after10 - before10:+.3f}",
        "",
        f"Файл ДО: {before_xlsx}",
        f"Файл ПОСЛЕ: {after_xlsx}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
