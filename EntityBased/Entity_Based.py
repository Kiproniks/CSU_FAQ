import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EntityBased:
    """Entity-like TF-IDF retrieval over chunks (not a graph model)."""

    def __init__(self, min_entity_length: int = 2, max_entities_per_chunk: int = 10):
        self.min_entity_length = min_entity_length
        self.max_entities_per_chunk = max_entities_per_chunk
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_entities: List[List[str]] = []
        self.entity_to_chunks = defaultdict(list)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
            max_features=8000,
        )
        self.tfidf_matrix = None

    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"(?u)\b[^\W\d_]+\b", text)

    def _extract_entities(self, words: List[str]) -> List[str]:
        entities = [word for word in set(words) if len(word) >= self.min_entity_length]
        word_freq = Counter(words)
        sorted_entities = sorted(entities, key=lambda x: word_freq[x], reverse=True)
        return sorted_entities[: self.max_entities_per_chunk]

    @staticmethod
    def _split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []

        safe_chunk_size = max(200, int(chunk_size))
        safe_overlap = min(max(0, int(overlap)), safe_chunk_size // 2)
        step = max(1, safe_chunk_size - safe_overlap)

        normalized = re.sub(r"\r\n?", "\n", text)
        normalized = re.sub(r"[ \t]+", " ", normalized).strip()
        if not normalized:
            return []

        units = [
            unit.strip()
            for unit in re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
            if unit and unit.strip()
        ]
        if not units:
            units = [normalized]

        chunks: List[str] = []
        current = ""

        for unit in units:
            if len(unit) > safe_chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""

                start = 0
                while start < len(unit):
                    piece = unit[start:start + safe_chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                    if start + safe_chunk_size >= len(unit):
                        break
                    start += step
                continue

            candidate = f"{current} {unit}".strip() if current else unit
            if not current or len(candidate) <= safe_chunk_size:
                current = candidate
                continue

            chunks.append(current.strip())
            tail = current[-safe_overlap:].strip() if safe_overlap else ""
            current = f"{tail} {unit}".strip() if tail else unit

        if current:
            chunks.append(current.strip())

        if len(chunks) > 1 and len(chunks[-1]) < max(80, safe_chunk_size // 6):
            chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
            chunks.pop()

        return chunks

    def add_chunk(
        self,
        chunk: str,
        chunk_id: int = None,
        doc_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        if chunk_id is None:
            chunk_id = len(self.chunks)
        if metadata is None:
            metadata = {}

        self.chunks.append(
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk,
                "metadata": metadata,
            }
        )

        words = self._preprocess_text(chunk)
        entities = self._extract_entities(words)
        self.chunk_entities.append(entities)

        for entity in entities:
            self.entity_to_chunks[entity].append(chunk_id)

        self.tfidf_matrix = None

    def add_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:
        chunks = self._split_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            self.add_chunk(
                chunk=chunk,
                chunk_id=len(self.chunks),
                doc_id=doc_id,
                metadata=metadata or {},
            )

    def build_index(self) -> None:
        corpus = [chunk["text"] for chunk in self.chunks]
        if not corpus:
            self.tfidf_matrix = None
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        if self.tfidf_matrix is None:
            self.build_index()
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(cosine_sims)[-top_k:][::-1]
        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            if cosine_sims[idx] <= 0:
                continue
            results.append(
                {
                    "id": self.chunks[idx]["id"],
                    "doc_id": self.chunks[idx].get("doc_id"),
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx].get("metadata", {}),
                    "entities": self.chunk_entities[idx],
                    "score": float(cosine_sims[idx]),
                }
            )

        return results

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        self.chunks.clear()
        self.chunk_entities.clear()
        self.entity_to_chunks.clear()
        self.tfidf_matrix = None
