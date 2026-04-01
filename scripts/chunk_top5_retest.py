from __future__ import annotations

import argparse
import importlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile

BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retest top-5 chunk settings on a new question set.")
    parser.add_argument(
        "--previous-xlsx",
        default=str(BASE_DIR / "для_отчёта" / "chunk_settings_benchmark_part2.xlsx"),
        help="Path to previous benchmark XLSX with summary sheet.",
    )
    parser.add_argument(
        "--questions-file",
        default=str(BASE_DIR / "data" / "benchmark_questions_new_chunks_20260322.json"),
        help="Path to new questions JSON.",
    )
    parser.add_argument(
        "--books-dir",
        default=str(BASE_DIR / "harry_potter"),
        help="Path to directory with source PDFs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many best settings to retest.",
    )
    parser.add_argument(
        "--output-xlsx",
        default=str(BASE_DIR / "для_отчёта" / "chunk_top5_new_questions.xlsx"),
        help="Output XLSX file.",
    )
    parser.add_argument(
        "--output-plot",
        default=str(BASE_DIR / "для_отчёта" / "chunk_top5_new_questions_plot.png"),
        help="Output chart PNG file.",
    )
    parser.add_argument(
        "--output-txt",
        default=str(BASE_DIR / "для_отчёта" / "chunk_top5_new_questions_summary.txt"),
        help="Output text report.",
    )
    return parser.parse_args()


def _cell_value(cell: ET.Element, ns: dict[str, str]) -> str:
    ctype = cell.attrib.get("t", "")
    if ctype == "inlineStr":
        node = cell.find("x:is/x:t", ns)
        return (node.text or "") if node is not None else ""
    node = cell.find("x:v", ns)
    return (node.text or "") if node is not None else ""


def read_top_settings(previous_xlsx: Path, top_n: int) -> list[dict[str, Any]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []

    with ZipFile(previous_xlsx) as zf:
        xml = zf.read("xl/worksheets/sheet1.xml")
        root = ET.fromstring(xml)
        for row in root.findall(".//x:sheetData/x:row", ns):
            values = [_cell_value(cell, ns) for cell in row.findall("x:c", ns)]
            if values:
                rows.append(values)

    if len(rows) < 2:
        raise RuntimeError(f"No summary rows found in {previous_xlsx}")

    header = [str(x).strip() for x in rows[0]]
    index = {name: idx for idx, name in enumerate(header)}
    required = ["setting", "chunk_size", "overlap", "top_k", "splitter", "mmr_lambda"]
    missing = [name for name in required if name not in index]
    if missing:
        raise RuntimeError(f"Summary sheet is missing columns: {missing}")

    top_rows = rows[1 : 1 + max(1, int(top_n))]
    result: list[dict[str, Any]] = []
    for row in top_rows:
        result.append(
            {
                "setting": row[index["setting"]],
                "chunk_size": int(float(row[index["chunk_size"]])),
                "overlap": int(float(row[index["overlap"]])),
                "top_k": int(float(row[index["top_k"]])),
                "splitter_mode": row[index["splitter"]],
                "mmr_lambda": float(row[index["mmr_lambda"]]),
            }
        )
    return result


def _load_chunk_benchmark_module():
    import sys

    scripts_dir = BASE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("chunk_settings_benchmark")


def _plot_summary(summary_rows: list[list[str]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available, skip plot: {exc}")
        return

    labels: list[str] = []
    scores: list[float] = []
    for row in summary_rows[1:]:
        labels.append(row[1])
        scores.append(float(row[7]))

    if not labels:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, scores, color="#0f766e")
    plt.title("Top-5 настроек ChunkBased на новом наборе вопросов")
    plt.ylabel("Средний score (0-10)")
    plt.ylim(0, 10)
    plt.xticks(rotation=20, ha="right")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(9.85, score + 0.08),
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_txt_report(
    out_path: Path,
    previous_xlsx: Path,
    questions_file: Path,
    summary_rows: list[list[str]],
    tested_questions: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Отчет: повторный тест top-5 настроек ChunkBased",
        "",
        f"Источник top-5: {previous_xlsx}",
        f"Новый набор вопросов: {questions_file}",
        f"Количество вопросов: {tested_questions}",
        "",
        "Итоги:",
    ]
    for row in summary_rows[1:]:
        lines.append(
            f"- #{row[0]} {row[1]} | chunk_size={row[2]} overlap={row[3]} top_k={row[4]} "
            f"splitter={row[5]} mmr={row[6]} | avg_score_10={row[7]}"
        )

    lines.append("")
    lines.append("Примечание: оценка совпадения считается через semantic-judge из scripts/chunk_settings_benchmark.py")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    previous_xlsx = Path(args.previous_xlsx)
    questions_file = Path(args.questions_file)
    books_dir = Path(args.books_dir)
    out_xlsx = Path(args.output_xlsx)
    out_plot = Path(args.output_plot)
    out_txt = Path(args.output_txt)

    bench = _load_chunk_benchmark_module()

    top_settings = read_top_settings(previous_xlsx, args.top_n)
    books = bench.load_books(books_dir)
    if not books:
        raise RuntimeError(f"No books found in {books_dir}")

    questions = bench.load_questions(questions_file)
    if not questions:
        raise RuntimeError(f"No questions loaded from {questions_file}")

    judge = bench.SemanticJudge()

    summary_items: list[dict[str, Any]] = []
    details_rows: list[list[str]] = [[
        "setting",
        "question_index",
        "question",
        "expected_answer",
        "generated_answer",
        "score_10",
        "comment",
    ]]

    for idx, row in enumerate(top_settings, start=1):
        setting = bench.ChunkSetting(
            chunk_size=row["chunk_size"],
            overlap=row["overlap"],
            top_k=row["top_k"],
            splitter_mode=row["splitter_mode"],
            mmr_lambda=row["mmr_lambda"],
        )
        print(f"[{idx}/{len(top_settings)}] {setting.name}")
        collection_name = f"chunk_top5_retest_{idx}_{setting.name}"
        engine = bench.ChunkBased(
            chunk_size=setting.chunk_size,
            overlap=setting.overlap,
            collection_name=collection_name,
            chroma_path=bench.settings.chroma_path,
            splitter_mode=setting.splitter_mode,
            mmr_lambda=setting.mmr_lambda,
            min_chunk_floor=80,
        )

        try:
            for doc_id, text in books:
                engine.add_document(text=text, doc_id=doc_id, metadata={"source": doc_id})

            scores_10: list[float] = []
            scores_01: list[float] = []

            for q_idx, item in enumerate(questions, start=1):
                question = str(item["question"])
                expected = str(item.get("expected_answer", ""))
                hits = engine.search(question, top_k=setting.top_k)
                generated = bench.extractive_answer(
                    question=question,
                    hits=hits,
                    semantic_model=judge.model,
                    max_sentences=5,
                )
                score01, score10, comment = judge.score(question=question, expected=expected, generated=generated)

                scores_10.append(score10)
                scores_01.append(score01)
                details_rows.append(
                    [
                        setting.name,
                        str(q_idx),
                        question,
                        expected,
                        generated,
                        f"{score10:.2f}",
                        comment,
                    ]
                )

            avg10 = (sum(scores_10) / len(scores_10)) if scores_10 else 0.0
            avg01 = (sum(scores_01) / len(scores_01)) if scores_01 else 0.0
            summary_items.append({"setting": setting, "avg10": avg10, "avg01": avg01})
            print(f"    avg_score_10={avg10:.3f}")
        finally:
            engine.clear()

    summary_items.sort(key=lambda x: float(x["avg10"]), reverse=True)

    summary_rows: list[list[str]] = [[
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

    for rank, item in enumerate(summary_items, start=1):
        s = item["setting"]
        summary_rows.append(
            [
                str(rank),
                s.name,
                str(s.chunk_size),
                str(s.overlap),
                str(s.top_k),
                s.splitter_mode,
                f"{s.mmr_lambda:.2f}",
                f"{float(item['avg10']):.3f}",
                f"{float(item['avg01']):.4f}",
            ]
        )

    bench._write_xlsx(out_xlsx, summary_rows, details_rows)
    _plot_summary(summary_rows, out_plot)
    _write_txt_report(out_txt, previous_xlsx, questions_file, summary_rows, tested_questions=len(questions))

    print(f"Saved XLSX: {out_xlsx}")
    print(f"Saved PLOT: {out_plot}")
    print(f"Saved TXT : {out_txt}")


if __name__ == "__main__":
    main()
