from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class TuneSetting:
    chunk_size: int
    overlap: int
    top_k: int
    splitter_mode: str
    mmr_lambda: float

    @property
    def name(self) -> str:
        mode = "sm" if self.splitter_mode == "smart" else "dd"
        return f"s{self.chunk_size}_o{self.overlap}_k{self.top_k}_m{mode}_mmr{self.mmr_lambda:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG tuning up to 1 hour with progress and report.")
    parser.add_argument("--dataset", default=str(BASE_DIR / "обучение ллм" / "dataset_eval.json"))
    parser.add_argument("--benchmark50", default=str(BASE_DIR / "data" / "benchmark_questions.json"))
    parser.add_argument("--books-dir", default=str(BASE_DIR / "harry_potter"))
    parser.add_argument("--max-minutes", type=int, default=55)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--complex-limit", type=int, default=20)
    parser.add_argument("--report-dir", default=str(BASE_DIR / "обучение ллм" / "отчёты"))
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_tags(dataset_path: Path) -> dict[str, Any]:
    data = _load_json(dataset_path)
    if not isinstance(data, list):
        raise RuntimeError("dataset_eval.json must be an array")

    # Единый словарь тегов (12 штук).
    canonical = {
        "персонажи",
        "место",
        "время",
        "предметы",
        "магия",
        "сюжет",
        "причина",
        "последствие",
        "сравнение",
        "уточнение",
        "ловушка",
        "неоднозначность",
    }

    aliases = {
        "факты": "сюжет",
        "факт": "сюжет",
        "события": "сюжет",
        "наблюдения": "сюжет",
        "новости": "сюжет",
        "имена": "персонажи",
        "родственники": "персонажи",
        "дружба": "персонажи",
        "причинно-следственное": "причина",
        "уточнение факта": "уточнение",
    }

    backup_path = dataset_path.with_suffix(".backup.json")
    if not backup_path.exists():
        backup_path.write_text(dataset_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    replaced = 0
    unknown = 0
    for row in data:
        tags = row.get("tags", [])
        normalized: list[str] = []
        if not isinstance(tags, list):
            tags = []
        for raw in tags:
            tag = str(raw).strip().lower()
            if not tag:
                continue
            mapped = aliases.get(tag, tag)
            if mapped not in canonical:
                unknown += 1
                continue
            normalized.append(mapped)
            if mapped != tag:
                replaced += 1
        if not normalized:
            normalized = ["сюжет"]
        # уникализация с сохранением порядка
        seen = set()
        row["tags"] = [x for x in normalized if not (x in seen or seen.add(x))]

    dataset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "items": len(data),
        "backup": str(backup_path),
        "replaced_tags": replaced,
        "unknown_dropped": unknown,
    }


def load_training_questions(dataset_path: Path, limit: int) -> list[dict[str, str]]:
    data = _load_json(dataset_path)
    questions: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("expected_answer", "")).strip()
        if not q or not a:
            continue
        questions.append({"question": q, "expected_answer": a})
    if limit > 0:
        return questions[:limit]
    return questions


def select_complex_20(path50: Path, limit: int = 20) -> list[dict[str, str]]:
    items = bench.load_questions(path50)

    def complexity_score(x: dict[str, str]) -> float:
        q = str(x.get("question", ""))
        a = str(x.get("expected_answer", ""))
        markers = ["почему", "зачем", "как", "чем", "правда ли", "в чем", "сравни"]
        marker_hits = sum(1 for m in markers if m in q.lower())
        return marker_hits * 2.0 + len(q) / 35.0 + len(a) / 120.0

    ranked = sorted(items, key=complexity_score, reverse=True)
    return ranked[:limit]


def load_books(books_dir: Path) -> list[tuple[str, str]]:
    books: list[tuple[str, str]] = []
    for pdf in sorted(books_dir.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf))
        if text.strip():
            books.append((pdf.stem, text))
    if not books:
        raise RuntimeError(f"No PDF files found in {books_dir}")
    return books


def evaluate_setting(
    setting: TuneSetting,
    books: list[tuple[str, str]],
    questions: list[dict[str, str]],
    judge: bench.SemanticJudge,
    collection_prefix: str,
) -> tuple[float, float, list[list[str]]]:
    collection_name = f"{collection_prefix}_{setting.name}_{int(time.time() * 1000)}"
    engine = ChunkBased(
        chunk_size=setting.chunk_size,
        overlap=setting.overlap,
        collection_name=collection_name,
        chroma_path=settings.chroma_path,
        splitter_mode=setting.splitter_mode,
        mmr_lambda=setting.mmr_lambda,
        min_chunk_floor=80,
    )

    details: list[list[str]] = []
    scores10: list[float] = []
    scores01: list[float] = []

    try:
        for doc_id, text in books:
            engine.add_document(text=text, doc_id=doc_id, metadata={"source": doc_id})

        for idx, item in enumerate(tqdm(questions, desc=f"{setting.name}", leave=False), start=1):
            q = item["question"]
            expected = item.get("expected_answer", "")
            hits = engine.search(q, top_k=setting.top_k)
            generated = bench.extractive_answer(
                question=q,
                hits=hits,
                semantic_model=judge.model,
                max_sentences=5,
            )
            s01, s10, comment = judge.score(question=q, expected=expected, generated=generated)
            scores10.append(s10)
            scores01.append(s01)
            details.append([setting.name, str(idx), q, expected, generated, f"{s10:.2f}", comment])
    finally:
        engine.clear()

    avg10 = (sum(scores10) / len(scores10)) if scores10 else 0.0
    avg01 = (sum(scores01) / len(scores01)) if scores01 else 0.0
    return avg10, avg01, details


def update_env_with_best(best: TuneSetting, env_path: Path) -> None:
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    out: list[str] = []
    replace = {
        "CHUNK_SIZE": str(best.chunk_size),
        "CHUNK_OVERLAP": str(best.overlap),
        "TOP_K_CHUNKS": str(best.top_k),
        "CHUNK_SPLITTER_MODE": best.splitter_mode,
        "CHUNK_MMR_LAMBDA": f"{best.mmr_lambda:.2f}",
    }
    seen = set()
    for line in lines:
        replaced = False
        for key, value in replace.items():
            prefix = f"{key}="
            if line.startswith(prefix):
                out.append(f"{key}={value}")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)
    for key, value in replace.items():
        if key not in seen:
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()

    dataset_path = Path(args.dataset)
    benchmark50_path = Path(args.benchmark50)
    books_dir = Path(args.books_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Нормализация тегов...")
    norm_info = normalize_tags(dataset_path)

    print("[2/6] Загрузка данных...")
    train_questions = load_training_questions(dataset_path, args.train_limit)
    complex20 = select_complex_20(benchmark50_path, args.complex_limit)
    books = load_books(books_dir)

    judge = bench.SemanticJudge()

    baseline = TuneSetting(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        top_k=settings.top_k_chunks,
        splitter_mode=settings.chunk_splitter_mode,
        mmr_lambda=settings.chunk_mmr_lambda,
    )

    # Сетка тюнинга вокруг рабочего baseline.
    grid = [
        TuneSetting(360, 60, 3, "smart", 0.70),
        TuneSetting(420, 70, 3, "smart", 0.72),
        TuneSetting(480, 80, 3, "smart", 0.74),
        TuneSetting(520, 90, 3, "smart", 0.76),
        TuneSetting(420, 70, 4, "smart", 0.72),
        TuneSetting(420, 70, 5, "smart", 0.72),
        TuneSetting(560, 100, 3, "smart", 0.76),
        TuneSetting(620, 110, 4, "smart", 0.78),
    ]

    print("[3/6] Прогон ДО обучения на 20 сложных вопросах...")
    before_20_avg10, before_20_avg01, before_details = evaluate_setting(
        baseline,
        books,
        complex20,
        judge,
        collection_prefix="before20",
    )

    print("[4/6] Запуск тюнинга (до часа) с прогрессом...")
    max_seconds = max(60, int(args.max_minutes) * 60)
    tune_results: list[dict[str, Any]] = []
    tune_details: list[list[str]] = []

    for idx, setting in enumerate(tqdm(grid, desc="configs", leave=True), start=1):
        elapsed = time.time() - started_at
        if elapsed >= max_seconds:
            print(f"Time budget reached at config #{idx-1}. Stop tuning.")
            break

        avg10, avg01, details = evaluate_setting(
            setting,
            books,
            train_questions,
            judge,
            collection_prefix="train",
        )
        tune_results.append({"setting": setting, "avg10": avg10, "avg01": avg01})
        tune_details.extend(details)
        print(f"  {setting.name}: avg_score_10={avg10:.3f}")

    if not tune_results:
        raise RuntimeError("Tuning produced no results (time budget too small).")

    tune_results.sort(key=lambda x: float(x["avg10"]), reverse=True)
    best: TuneSetting = tune_results[0]["setting"]

    print("[5/6] Прогон ПОСЛЕ обучения на тех же 20 сложных вопросах...")
    after_20_avg10, after_20_avg01, after_details = evaluate_setting(
        best,
        books,
        complex20,
        judge,
        collection_prefix="after20",
    )

    update_env_with_best(best, BASE_DIR / ".env")

    print("[6/6] Сохранение отчётов...")
    summary_rows = [[
        "rank",
        "setting",
        "chunk_size",
        "overlap",
        "top_k",
        "splitter",
        "mmr_lambda",
        "avg_score_10",
        "avg_score_01",
    ]]
    for rank, item in enumerate(tune_results, start=1):
        s: TuneSetting = item["setting"]
        summary_rows.append([
            str(rank),
            s.name,
            str(s.chunk_size),
            str(s.overlap),
            str(s.top_k),
            s.splitter_mode,
            f"{s.mmr_lambda:.2f}",
            f"{float(item['avg10']):.3f}",
            f"{float(item['avg01']):.4f}",
        ])

    details_rows = [["setting", "question_index", "question", "expected_answer", "generated_answer", "score_10", "comment"]]
    details_rows.extend(tune_details)

    details_before = [["setting", "question_index", "question", "expected_answer", "generated_answer", "score_10", "comment"]]
    details_before.extend(before_details)
    details_after = [["setting", "question_index", "question", "expected_answer", "generated_answer", "score_10", "comment"]]
    details_after.extend(after_details)

    ts = time.strftime("%Y%m%d_%H%M%S")
    tune_xlsx = report_dir / f"training_tuning_{ts}.xlsx"
    before_xlsx = report_dir / f"before_20_complex_{ts}.xlsx"
    after_xlsx = report_dir / f"after_20_complex_{ts}.xlsx"
    report_txt = report_dir / f"training_report_{ts}.txt"

    bench._write_xlsx(tune_xlsx, summary_rows, details_rows)

    # Для before/after используем фиктивную summary из одной строки.
    bench._write_xlsx(
        before_xlsx,
        [summary_rows[0], ["1", baseline.name, str(baseline.chunk_size), str(baseline.overlap), str(baseline.top_k), baseline.splitter_mode, f"{baseline.mmr_lambda:.2f}", f"{before_20_avg10:.3f}", f"{before_20_avg01:.4f}"]],
        details_before,
    )
    bench._write_xlsx(
        after_xlsx,
        [summary_rows[0], ["1", best.name, str(best.chunk_size), str(best.overlap), str(best.top_k), best.splitter_mode, f"{best.mmr_lambda:.2f}", f"{after_20_avg10:.3f}", f"{after_20_avg01:.4f}"]],
        details_after,
    )

    delta = after_20_avg10 - before_20_avg10
    elapsed_total = time.time() - started_at

    report_lines = [
        "Отчёт по обучению (тюнингу RAG)",
        "",
        "Зачем сделано:",
        "- нормализовать теги датасета для чистой аналитики ошибок;",
        "- подобрать лучшие retrieval-настройки под твой набор 200 вопросов;",
        "- проверить реальный эффект на 20 сложных вопросах до/после.",
        "",
        f"Время выполнения: {elapsed_total/60:.1f} минут",
        f"Нормализация тегов: items={norm_info['items']}, replaced={norm_info['replaced_tags']}, dropped_unknown={norm_info['unknown_dropped']}",
        f"Backup датасета: {norm_info['backup']}",
        "",
        f"Baseline (до): {baseline.name}",
        f"Best (после): {best.name}",
        f"Сложные 20 до:   avg_score_10={before_20_avg10:.3f}",
        f"Сложные 20 после: avg_score_10={after_20_avg10:.3f}",
        f"Дельта: {delta:+.3f}",
        "",
        "Top tuning results:",
    ]
    for idx, item in enumerate(tune_results[:5], start=1):
        s = item["setting"]
        report_lines.append(f"{idx}. {s.name} -> {item['avg10']:.3f}")

    report_lines.extend([
        "",
        f"Файлы: {tune_xlsx}",
        f"       {before_xlsx}",
        f"       {after_xlsx}",
    ])
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print("Training completed.")
    print(f"Report: {report_txt}")


if __name__ == "__main__":
    main()
