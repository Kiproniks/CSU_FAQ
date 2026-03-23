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
class FixedSetting:
    chunk_size: int = 420
    overlap: int = 70
    top_k: int = 3
    splitter_mode: str = "smart"
    mmr_lambda: float = 0.72

    @property
    def name(self) -> str:
        return f"s{self.chunk_size}_o{self.overlap}_k{self.top_k}_msm_mmr{self.mmr_lambda:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed best config training/evaluation with progress and reports")
    parser.add_argument("--dataset", default=str(BASE_DIR / "обучение ллм" / "dataset_eval.json"))
    parser.add_argument("--benchmark50", default=str(BASE_DIR / "data" / "benchmark_questions.json"))
    parser.add_argument("--books-dir", default=str(BASE_DIR / "harry_potter"))
    parser.add_argument("--complex-limit", type=int, default=20)
    parser.add_argument("--report-dir", default=str(BASE_DIR / "обучение ллм" / "отчёты"))
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_tags(dataset_path: Path) -> dict:
    data = _load_json(dataset_path)
    canonical = {
        "персонажи", "место", "время", "предметы", "магия", "сюжет",
        "причина", "последствие", "сравнение", "уточнение", "ловушка", "неоднозначность",
    }
    aliases = {
        "факты": "сюжет", "факт": "сюжет", "события": "сюжет", "наблюдения": "сюжет",
        "новости": "сюжет", "имена": "персонажи", "родственники": "персонажи", "дружба": "персонажи",
    }

    backup = dataset_path.with_suffix(".backup.json")
    if not backup.exists():
        backup.write_text(dataset_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    replaced = 0
    dropped = 0
    for row in data:
        tags = row.get("tags", []) if isinstance(row.get("tags", []), list) else []
        out = []
        seen = set()
        for t in tags:
            v = aliases.get(str(t).strip().lower(), str(t).strip().lower())
            if v not in canonical:
                dropped += 1
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
            if v != str(t).strip().lower():
                replaced += 1
        if not out:
            out = ["сюжет"]
        row["tags"] = out

    dataset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"items": len(data), "replaced": replaced, "dropped": dropped, "backup": str(backup)}


def load_books(books_dir: Path):
    books = []
    for pdf in sorted(books_dir.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf))
        if text.strip():
            books.append((pdf.stem, text))
    if not books:
        raise RuntimeError(f"No PDF books in {books_dir}")
    return books


def load_eval_questions(dataset_path: Path):
    data = _load_json(dataset_path)
    out = []
    for x in data:
        q = str(x.get("question", "")).strip()
        a = str(x.get("expected_answer", "")).strip()
        if q and a:
            out.append({"question": q, "expected_answer": a})
    return out


def select_complex(path50: Path, limit: int):
    items = bench.load_questions(path50)

    def score(x: dict) -> float:
        q = str(x.get("question", "")).lower()
        a = str(x.get("expected_answer", ""))
        m = sum(1 for k in ["почему", "зачем", "как", "чем", "правда", "в чем", "объясни"] if k in q)
        return m * 2 + len(q) / 35 + len(a) / 120

    return sorted(items, key=score, reverse=True)[:limit]


def evaluate(setting: FixedSetting, books, questions, judge, prefix: str):
    engine = ChunkBased(
        chunk_size=setting.chunk_size,
        overlap=setting.overlap,
        collection_name=f"{prefix}_{int(time.time()*1000)}",
        chroma_path=settings.chroma_path,
        splitter_mode=setting.splitter_mode,
        mmr_lambda=setting.mmr_lambda,
        min_chunk_floor=80,
    )
    details = []
    s10 = []
    s01 = []
    try:
        for doc_id, text in books:
            engine.add_document(text=text, doc_id=doc_id, metadata={"source": doc_id})

        for idx, item in enumerate(tqdm(questions, desc=prefix, leave=False), start=1):
            q = item["question"]
            expected = item.get("expected_answer", "")
            hits = engine.search(q, top_k=setting.top_k)
            generated = bench.extractive_answer(question=q, hits=hits, semantic_model=judge.model, max_sentences=5)
            v01, v10, comment = judge.score(question=q, expected=expected, generated=generated)
            s10.append(v10)
            s01.append(v01)
            details.append([setting.name, str(idx), q, expected, generated, f"{v10:.2f}", comment])
    finally:
        engine.clear()

    return (sum(s10)/len(s10) if s10 else 0.0), (sum(s01)/len(s01) if s01 else 0.0), details


def write_env(setting: FixedSetting, env_path: Path):
    lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    replace = {
        "CHUNK_SIZE": str(setting.chunk_size),
        "CHUNK_OVERLAP": str(setting.overlap),
        "TOP_K_CHUNKS": str(setting.top_k),
        "CHUNK_SPLITTER_MODE": setting.splitter_mode,
        "CHUNK_MMR_LAMBDA": f"{setting.mmr_lambda:.2f}",
    }
    out = []
    seen = set()
    for line in lines:
        hit = False
        for k,v in replace.items():
            if line.startswith(k + "="):
                out.append(f"{k}={v}")
                seen.add(k)
                hit = True
                break
        if not hit:
            out.append(line)
    for k,v in replace.items():
        if k not in seen:
            out.append(f"{k}={v}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    started = time.time()
    dataset_path = Path(args.dataset)
    benchmark50 = Path(args.benchmark50)
    books_dir = Path(args.books_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Нормализация тегов...")
    norm = normalize_tags(dataset_path)

    print("[2/5] Загрузка данных...")
    train_q = load_eval_questions(dataset_path)
    complex20 = select_complex(benchmark50, args.complex_limit)
    books = load_books(books_dir)
    judge = bench.SemanticJudge()

    setting = FixedSetting()

    print("[3/5] Прогон ДО (20 сложных)...")
    before10, before01, before_details = evaluate(setting, books, complex20, judge, "before20")

    print("[4/5] Обучение/прогон на 200 вопросах (фиксированный лучший конфиг)...")
    train10, train01, train_details = evaluate(setting, books, train_q, judge, "train200")

    print("[5/5] Прогон ПОСЛЕ (те же 20 сложных)...")
    after10, after01, after_details = evaluate(setting, books, complex20, judge, "after20")

    write_env(setting, BASE_DIR / ".env")

    ts = time.strftime("%Y%m%d_%H%M%S")
    x_train = report_dir / f"fixed_train200_{ts}.xlsx"
    x_before = report_dir / f"fixed_before20_{ts}.xlsx"
    x_after = report_dir / f"fixed_after20_{ts}.xlsx"
    t_report = report_dir / f"fixed_training_report_{ts}.txt"

    header = ["rank", "setting", "chunk_size", "overlap", "top_k", "splitter", "mmr_lambda", "avg_score_10", "avg_score_01"]
    det_header = [["setting", "question_index", "question", "expected_answer", "generated_answer", "score_10", "comment"]]

    bench._write_xlsx(
        x_train,
        [header, ["1", setting.name, str(setting.chunk_size), str(setting.overlap), str(setting.top_k), setting.splitter_mode, f"{setting.mmr_lambda:.2f}", f"{train10:.3f}", f"{train01:.4f}"]],
        det_header + train_details,
    )
    bench._write_xlsx(
        x_before,
        [header, ["1", setting.name, str(setting.chunk_size), str(setting.overlap), str(setting.top_k), setting.splitter_mode, f"{setting.mmr_lambda:.2f}", f"{before10:.3f}", f"{before01:.4f}"]],
        det_header + before_details,
    )
    bench._write_xlsx(
        x_after,
        [header, ["1", setting.name, str(setting.chunk_size), str(setting.overlap), str(setting.top_k), setting.splitter_mode, f"{setting.mmr_lambda:.2f}", f"{after10:.3f}", f"{after01:.4f}"]],
        det_header + after_details,
    )

    elapsed = (time.time() - started) / 60
    report = [
        "Отчёт: обучение LLM (тюнинг RAG) на фиксированном лучшем конфиге",
        "",
        "Почему это делали:",
        "- привести теги к единому словарю для чистой аналитики;",
        "- прогнать 200 вопросов на стабильной лучшей конфигурации;",
        "- проверить качество на одинаковых 20 сложных вопросах до/после.",
        "",
        f"Конфиг: {setting.name}",
        f"Нормализация тегов: items={norm['items']} replaced={norm['replaced']} dropped={norm['dropped']}",
        f"Backup: {norm['backup']}",
        "",
        f"Train-200 avg_score_10: {train10:.3f}",
        f"Before-20 avg_score_10: {before10:.3f}",
        f"After-20 avg_score_10:  {after10:.3f}",
        f"Delta after-before:     {after10 - before10:+.3f}",
        f"Время: {elapsed:.1f} минут",
        "",
        f"Файлы:\n- {x_train}\n- {x_before}\n- {x_after}",
    ]
    t_report.write_text("\n".join(report), encoding="utf-8")

    print("Done.")
    print(t_report)


if __name__ == "__main__":
    main()
