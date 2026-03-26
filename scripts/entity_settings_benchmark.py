from __future__ import annotations

import argparse
import json
import math
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from xml.sax.saxutils import escape
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self, judge_model: str = None, timeout_sec: int = 180, warmup: bool = True):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip('/')
        self.model = judge_model or os.getenv("LLM_MODEL", "qwen3:8b")
        self.temperature = 0.0
        self.timeout = timeout_sec
        if warmup:
            self._warmup()

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"Ошибка при вызове Ollama: {e}")
            return ""

    def _warmup(self):
        """Прогрев модели перед началом работы."""
        try:
            print(f"Прогрев модели {self.model}...")
            self._call_ollama("Привет")
            print("Модель готова.")
        except Exception as e:
            print(f"Прогрев не удался: {e}")

    def score(self, question: str, expected: str, generated: str) -> tuple[float, float, str]:
        prompt = f"""Ты — строгий эксперт по оценке качества ответов на вопросы по правилам дорожного движения.
Оцени сгенерированный ответ по трём критериям:
- Точность (соответствие фактам ПДД, отсутствие ошибок) – 40%
- Полнота (охватывает ли ответ ключевые моменты) – 30%
- Релевантность (отвечает ли непосредственно на вопрос) – 30%

Вопрос: {question}
Эталонный ответ: {expected}
Сгенерированный ответ: {generated}

Выставь итоговую оценку по шкале от 1 до 10 (целое число) и краткий комментарий.
Формат ответа:
Оценка: X
Комментарий: ... (кратко, почему такая оценка)

Примеры комментариев:
- "Ответ верный, но не упомянута табличка 8.13."
- "Ответ содержит грубую ошибку (названа не та статья)."
- "Точный и полный ответ."
- "Отлично, приведены все детали."
"""
        try:
            response = self._call_ollama(prompt)
            if not response:
                from scripts.chunk_settings_benchmark import SemanticJudge
                return SemanticJudge().score(question, expected, generated)

            # Поиск оценки
            match = re.search(r'Оценка:\s*(\d+(?:\.\d+)?)', response)
            if not match:
                match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            score = float(match.group(1)) if match else 5.0
            score = min(10.0, max(1.0, score))

            # Поиск комментария
            comment = ""
            if "Комментарий:" in response:
                comment = response.split("Комментарий:", 1)[1].strip()
            elif "Comment:" in response:
                comment = response.split("Comment:", 1)[1].strip()

            score10 = round(score, 2)
            score01 = round(score10 / 10.0, 4)
            return score01, score10, comment

        except Exception as e:
            print(f"LLMJudge error: {e}, falling back to SemanticJudge")
            from scripts.chunk_settings_benchmark import SemanticJudge
            return SemanticJudge().score(question, expected, generated)

BASE_DIR = Path(__file__).resolve().parent.parent


def _ensure_imports() -> None:
    import site
    import sys

    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    venv_site = BASE_DIR / "venv" / "Lib" / "site-packages"
    if venv_site.exists():
        site.addsitedir(str(venv_site))


_ensure_imports()

from sentence_transformers import SentenceTransformer  # noqa: E402

from EntityBased.EntityBased import EntityBased  # noqa: E402
from app.pdf_utils import extract_text_from_pdf  # noqa: E402
from app.text_splitter import split_sentences  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04FF0-9\-]{2,}")
STOPWORDS = {
    "и", "в", "во", "на", "по", "к", "ко", "с", "со", "для", "или", "а", "но", "что", "это", "как", "о",
    "об", "у", "из", "за", "не", "он", "она", "они", "его", "ее", "их", "the", "a", "an", "and", "or",
    "to", "in", "on", "of", "is", "are", "was", "were",
}


@dataclass
class EntitySetting:
    chunk_size: int
    overlap: int
    top_k: int
    min_entity_length: int
    max_entities_per_chunk: int
    tfidf_weight: float
    entity_overlap_weight: float
    min_score: float
    mmr_lambda: float

    @property
    def name(self) -> str:
        return (
            f"s{self.chunk_size}_o{self.overlap}_k{self.top_k}_"
            f"m{self.min_entity_length}_mx{self.max_entities_per_chunk}_"
            f"w{self.tfidf_weight:.2f}-{self.entity_overlap_weight:.2f}_"
            f"ms{self.min_score:.3f}_mmr{self.mmr_lambda:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 10 EntityBased settings and export XLSX.")
    parser.add_argument(
        "--books-dir",
        default=str(BASE_DIR / "harry_potter"),
        help="Path to directory with source PDFs",
    )
    parser.add_argument(
        "--questions-file",
        default=str(BASE_DIR / "data" / "benchmark_questions.json"),
        help="Path to benchmark questions JSON",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "для_отчёта" / "entity_settings_benchmark_15.xlsx"),
        help="Output XLSX path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Limit number of questions (default 15)",
    )
    return parser.parse_args()


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS]


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a) / float(b)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_from_embeddings(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    num = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class SemanticJudge:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def score(self, question: str, expected: str, generated: str) -> tuple[float, float, str]:
        question = str(question or "").strip()
        expected = str(expected or "").strip()
        generated = str(generated or "").strip()

        exp_tokens = set(_content_tokens(expected))
        gen_tokens = set(_content_tokens(generated))

        if not expected:
            informative = min(1.0, len(gen_tokens) / 80.0)
            score01 = 0.4 + 0.6 * informative
            return score01, round(score01 * 10.0, 2), "Нет эталона: оценка по информативности."

        if not generated:
            return 0.0, 0.0, "Ответ пустой."

        emb = self.model.encode([question, expected, generated], normalize_embeddings=True)
        sem_expected = _clamp((_cosine_from_embeddings(emb[1], emb[2]) + 1.0) / 2.0)
        sem_question = _clamp((_cosine_from_embeddings(emb[0], emb[2]) + 1.0) / 2.0)

        inter = len(exp_tokens & gen_tokens)
        precision = _safe_div(inter, len(gen_tokens) or 1)
        recall = _safe_div(inter, len(exp_tokens) or 1)
        exact = 1.0 if expected.lower() in generated.lower() else 0.0

        raw = _clamp(0.62 * sem_expected + 0.23 * sem_question + 0.10 * recall + 0.05 * exact)
        if len(generated) > max(1, len(expected)) * 8 and recall < 0.35:
            raw = max(0.0, raw - 0.05)

        score10 = round(min(10.0, max(0.0, 2.6 + 8.0 * raw)), 2)
        score01 = round(score10 / 10.0, 4)
        if score10 >= 8.0:
            comment = "Высокое совпадение."
        elif score10 >= 6.0:
            comment = "Среднее совпадение."
        else:
            comment = "Слабое совпадение."
        return score01, score10, comment


def load_books(books_dir: Path) -> List[tuple[str, str]]:
    books: List[tuple[str, str]] = []
    for pdf in sorted(books_dir.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf))
        if text.strip():
            books.append((pdf.stem, text))
    return books


def load_questions(path: Path) -> List[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    result: List[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("query") or "").strip()
        if not question:
            continue
        expected = str(
            item.get("expected_answer")
            or item.get("expected")
            or item.get("reference_answer")
            or item.get("gold_answer")
            or ""
        ).strip()
        result.append({"question": question, "expected_answer": expected})
    return result


def settings_grid() -> List[EntitySetting]:
    return [
        EntitySetting(420, 70, 3, 3, 14, 0.78, 0.22, 0.025, 0.72),
        EntitySetting(520, 90, 3, 3, 16, 0.75, 0.25, 0.022, 0.74),
        EntitySetting(620, 110, 3, 3, 18, 0.72, 0.28, 0.020, 0.74),
        EntitySetting(700, 120, 4, 3, 18, 0.70, 0.30, 0.020, 0.76),
        EntitySetting(820, 130, 4, 4, 20, 0.68, 0.32, 0.018, 0.78),
        EntitySetting(900, 140, 4, 4, 22, 0.66, 0.34, 0.018, 0.78),
        EntitySetting(1000, 180, 4, 4, 24, 0.64, 0.36, 0.017, 0.80),
        EntitySetting(1100, 200, 5, 5, 26, 0.62, 0.38, 0.016, 0.82),
        EntitySetting(760, 120, 4, 3, 20, 0.70, 0.30, 0.019, 0.76),
        EntitySetting(560, 100, 3, 2, 16, 0.80, 0.20, 0.023, 0.72),
    ]


def extractive_answer(
    question: str,
    hits: Sequence[tuple[dict[str, Any], float]],
    semantic_model: SentenceTransformer | None = None,
    max_sentences: int = 5,
) -> str:
    q_tokens = set(_content_tokens(question))
    candidates: List[tuple[float, str]] = []
    seen = set()

    for payload, hit_score in hits:
        text = str((payload or {}).get("text", "") or "")
        if not text.strip():
            continue
        for sentence in split_sentences(text):
            sent = sentence.strip()
            if len(sent) < 22:
                continue
            stokens = set(_content_tokens(sent))
            overlap = len(q_tokens & stokens)
            if overlap <= 0:
                continue
            density = overlap / max(1, len(stokens))
            entity_tokens = set((payload or {}).get("entities", []) or [])
            entity_bonus = 0.05 * len(q_tokens & entity_tokens)
            rank = 0.58 * overlap + 0.22 * density + 0.15 * float(hit_score) + entity_bonus
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((rank, sent))

    if not candidates:
        if hits:
            fallback = " ".join(str(hits[0][0].get("text", "") or "").split())[:1100]
            return fallback or "Недостаточно данных в найденных фрагментах."
        return "Недостаточно данных в найденных фрагментах."

    if semantic_model is not None:
        try:
            sents = [sent for _, sent in candidates]
            emb = semantic_model.encode([question] + sents, normalize_embeddings=True)
            qv = emb[0]
            rescored: List[tuple[float, str]] = []
            for (lex_rank, sent), sv in zip(candidates, emb[1:]):
                sem = _clamp((_cosine_from_embeddings(qv, sv) + 1.0) / 2.0)
                final_rank = 0.58 * float(lex_rank) + 0.42 * sem
                rescored.append((final_rank, sent))
            candidates = rescored
        except Exception:
            pass

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [sent for _, sent in candidates[: max(1, int(max_sentences))]]
    answer = " ".join(selected).strip()
    return answer[:1400] if len(answer) > 1400 else answer


def _xml_safe(value: Any) -> str:
    text = str(value if value is not None else "")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    return escape(text, {"'": "&apos;", '"': "&quot;"})


def _col_name(index_1_based: int) -> str:
    n = int(index_1_based)
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _iter_rows_xml(rows: Iterable[List[str]], style_idx: int = 1) -> str:
    xml_rows: List[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells: List[str] = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{_col_name(col_idx)}{row_idx}"
            cells.append(
                f'<c r="{ref}" s="{style_idx}" t="inlineStr">'
                f"<is><t xml:space=\"preserve\">{_xml_safe(value)}</t></is>"
                "</c>"
            )
        xml_rows.append(f"<row r=\"{row_idx}\">{''.join(cells)}</row>")
    return "".join(xml_rows)


def _sheet_xml(rows: List[List[str]], widths: List[float]) -> str:
    cols_parts: List[str] = []
    for i, width in enumerate(widths, start=1):
        cols_parts.append(f'<col min="{i}" max="{i}" width="{float(width):.2f}" customWidth="1"/>')
    cols_xml = "".join(cols_parts)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<sheetFormatPr defaultRowHeight=\"30\"/>"
        f"<cols>{cols_xml}</cols>"
        f"<sheetData>{_iter_rows_xml(rows)}</sheetData>"
        "</worksheet>"
    )


def _write_xlsx(path: Path, summary_rows: List[List[str]], details_rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_xml = _sheet_xml(
        summary_rows,
        widths=[8, 40, 10, 10, 8, 8, 12, 10, 10, 10, 14, 14],
    )
    details_xml = _sheet_xml(
        details_rows,
        widths=[40, 12, 48, 52, 56, 14, 24],
    )

    workbook_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<sheets>"
        "<sheet name=\"summary\" sheetId=\"1\" r:id=\"rId1\"/>"
        "<sheet name=\"details\" sheetId=\"2\" r:id=\"rId2\"/>"
        "</sheets>"
        "</workbook>"
    )

    workbook_rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
        "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet2.xml\"/>"
        "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
        "</Relationships>"
    )

    root_rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
        "</Relationships>"
    )

    styles_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        "<fills count=\"2\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill></fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"2\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyAlignment=\"1\"><alignment vertical=\"top\" wrapText=\"1\"/></xf>"
        "</cellXfs>"
        "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
        "</styleSheet>"
    )

    content_types_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
        "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        "<Override PartName=\"/xl/worksheets/sheet2.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>"
        "</Types>"
    )

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", summary_xml)
        zf.writestr("xl/worksheets/sheet2.xml", details_xml)
        zf.writestr("xl/styles.xml", styles_xml)


def _assessment(avg_score10: float) -> str:
    if avg_score10 >= 8.5:
        return "Отличные настройки"
    if avg_score10 >= 7.5:
        return "Хорошие настройки"
    if avg_score10 >= 6.0:
        return "Средние настройки"
    return "Слабые настройки, нужен дальнейший тюнинг"


def main() -> None:
    args = parse_args()
    books = load_books(Path(args.books_dir))
    if not books:
        raise RuntimeError(f"No PDF books found in {args.books_dir}")

    questions = load_questions(Path(args.questions_file))
    if not questions:
        raise RuntimeError("No questions loaded.")

    limit = int(args.limit or 0)
    if limit > 0:
        questions = questions[:limit]

    grid = settings_grid()
    judge = LLMJudge(judge_model="llama3.2:3b")

    summary_items: List[dict[str, Any]] = []
    details_rows: List[List[str]] = [[
        "setting",
        "question_index",
        "question",
        "expected_answer",
        "generated_answer",
        "score_10",
        "comment",
    ]]

    for idx_setting, setting in enumerate(grid, start=1):
        print(f"[{idx_setting}/{len(grid)}] setting: {setting.name}")
        engine = EntityBased(
            min_entity_length=setting.min_entity_length,
            max_entities_per_chunk=setting.max_entities_per_chunk,
            tfidf_weight=setting.tfidf_weight,
            entity_overlap_weight=setting.entity_overlap_weight,
            min_score=setting.min_score,
            mmr_lambda=setting.mmr_lambda,
        )
        try:
            for doc_id, text in books:
                engine.add_document(
                    text=text,
                    doc_id=doc_id,
                    metadata={"source": doc_id},
                    chunk_size=setting.chunk_size,
                    overlap=setting.overlap,
                )
            engine.build_index()

            scores_10: List[float] = []
            scores_01: List[float] = []
            for q_idx, item in enumerate(questions, start=1):
                question = str(item["question"])
                expected = str(item.get("expected_answer", ""))
                hits = engine.search(question, top_k=setting.top_k)
                generated = extractive_answer(
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

    summary_rows: List[List[str]] = [[
        "rank",
        "setting",
        "chunk_size",
        "overlap",
        "top_k",
        "min_len",
        "max_entities",
        "tfidf_w",
        "entity_w",
        "min_score",
        "avg_score_10",
        "avg_score_01",
    ]]
    for rank, item in enumerate(summary_items, start=1):
        s: EntitySetting = item["setting"]
        summary_rows.append(
            [
                str(rank),
                s.name,
                str(s.chunk_size),
                str(s.overlap),
                str(s.top_k),
                str(s.min_entity_length),
                str(s.max_entities_per_chunk),
                f"{s.tfidf_weight:.2f}",
                f"{s.entity_overlap_weight:.2f}",
                f"{s.min_score:.3f}",
                f"{float(item['avg10']):.3f}",
                f"{float(item['avg01']):.4f}",
            ]
        )

    out_path = Path(args.output)
    _write_xlsx(out_path, summary_rows, details_rows)
    print(f"Saved: {out_path}")

    if summary_items:
        best = summary_items[0]
        bs: EntitySetting = best["setting"]
        best_avg = float(best["avg10"])
        print("Best setting:")
        print(
            f"  {bs.name} | chunk_size={bs.chunk_size} overlap={bs.overlap} top_k={bs.top_k} "
            f"min_len={bs.min_entity_length} max_entities={bs.max_entities_per_chunk}"
        )
        print(
            f"  tfidf_w={bs.tfidf_weight:.2f} entity_w={bs.entity_overlap_weight:.2f} "
            f"min_score={bs.min_score:.3f} mmr={bs.mmr_lambda:.2f}"
        )
        print(f"  avg_score_10={best_avg:.3f}")
        print(f"  assessment={_assessment(best_avg)}")


if __name__ == "__main__":
    main()

