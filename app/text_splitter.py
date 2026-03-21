from __future__ import annotations

import re
from typing import Iterable, List


def normalize_text(text: str) -> str:
    """Нормализует пробелы и переводит текст в удобный для разбиения вид."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(paragraph: str) -> List[str]:
    """Делит абзац на предложения с учетом русского и английского текста."""
    if not paragraph:
        return []

    candidate = re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9\"«])", paragraph.strip())
    sentences = [s.strip() for s in candidate if s.strip()]
    if sentences:
        return sentences

    return [paragraph.strip()]


def _split_long_unit(unit: str, max_len: int) -> List[str]:
    """Режет слишком длинное предложение по словам, если оно не помещается в chunk."""
    compact = " ".join(unit.split())
    if len(compact) <= max_len:
        return [compact]

    words = compact.split(" ")
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        projected = current_len + (1 if current else 0) + len(word)
        if current and projected > max_len:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue

        current.append(word)
        current_len = projected

    if current:
        parts.append(" ".join(current))

    return [p for p in parts if p]


def _merge_small_chunks(chunks: List[str], min_chunk_size: int) -> List[str]:
    """Склеивает короткие хвосты, чтобы не плодить шумные mini-chunks."""
    if not chunks:
        return chunks

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk_size:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def _iter_text_units(text: str) -> Iterable[str]:
    """Возвращает поток смысловых единиц: предложения из абзацев."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    for paragraph in paragraphs:
        for sentence in split_sentences(paragraph):
            yield sentence


def smart_chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 200,
    min_chunk_floor: int = 250,
) -> List[str]:
    """
    Умное разбиение текста:
    - по предложениям (не ломаем мысль посередине),
    - с перекрытием по последним предложениям,
    - с защитой от слишком коротких чанков.
    """
    text = normalize_text(text)
    if not text:
        return []

    effective_chunk_size = max(int(min_chunk_floor), int(chunk_size))
    effective_overlap = max(0, min(int(overlap), effective_chunk_size - 80))
    effective_min_chunk = max(80, min(int(min_chunk_size), effective_chunk_size))

    units: List[str] = []
    for unit in _iter_text_units(text):
        units.extend(_split_long_unit(unit, effective_chunk_size))

    if not units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []

    def flush_current() -> None:
        if not current_units:
            return
        chunk = " ".join(current_units).strip()
        if chunk:
            chunks.append(chunk)

    for unit in units:
        projected = " ".join(current_units + [unit]).strip()
        if current_units and len(projected) > effective_chunk_size:
            flush_current()

            # Формируем новое окно с overlap по последним единицам предыдущего чанка.
            overlap_units: List[str] = []
            overlap_len = 0
            for prev in reversed(current_units):
                overlap_units.insert(0, prev)
                overlap_len += len(prev) + 1
                if overlap_len >= effective_overlap:
                    break

            current_units = overlap_units
            projected = " ".join(current_units + [unit]).strip()

            if current_units and len(projected) > effective_chunk_size:
                # Если overlap занял слишком много, начинаем свежий чанк.
                current_units = []

        current_units.append(unit)

    flush_current()

    return _merge_small_chunks(chunks, effective_min_chunk)


def _group_sentences_by_size(sentences: List[str], chunk_size: int) -> List[List[str]]:
    """
    Группирует предложения в базовые чанки без overlap, не разрывая предложение.
    """
    if not sentences:
        return []

    grouped: List[List[str]] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected = current_len + (1 if current else 0) + len(sentence)
        if current and projected > chunk_size:
            grouped.append(current)
            current = [sentence]
            current_len = len(sentence)
            continue
        current.append(sentence)
        current_len = projected

    if current:
        grouped.append(current)
    return grouped


def dot_dot_chunk_text(
    text: str,
    chunk_size: int,
    neighbor_overlap_sentences: int = 1,
    min_chunk_floor: int = 250,
) -> List[str]:
    """
    Разбиение "dot-dot":
    - режем строго по предложениям,
    - формируем базовые чанки без overlap,
    - добавляем контекст от соседних чанков:
      одно предложение слева и одно справа (по умолчанию).
    """
    text = normalize_text(text)
    if not text:
        return []

    effective_chunk_size = max(int(min_chunk_floor), int(chunk_size))
    overlap_n = max(0, int(neighbor_overlap_sentences))

    sentences: List[str] = []
    for sentence in _iter_text_units(text):
        sentences.extend(_split_long_unit(sentence, effective_chunk_size))
    if not sentences:
        return []

    base_chunks = _group_sentences_by_size(sentences, effective_chunk_size)
    if not base_chunks:
        return []

    chunks: List[str] = []
    for idx, base in enumerate(base_chunks):
        left_ctx: List[str] = []
        right_ctx: List[str] = []

        if overlap_n > 0 and idx > 0:
            prev_chunk = base_chunks[idx - 1]
            left_ctx = prev_chunk[-overlap_n:]

        if overlap_n > 0 and idx + 1 < len(base_chunks):
            next_chunk = base_chunks[idx + 1]
            right_ctx = next_chunk[:overlap_n]

        chunk_sentences = left_ctx + base + right_ctx
        compact = " ".join(s.strip() for s in chunk_sentences if s.strip()).strip()
        if compact:
            chunks.append(compact)

    return chunks
