from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.text_splitter import smart_chunk_text

RUS_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "когда", "даже", "ну", "ли", "если", "или",
    "ни", "до", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "для", "мы", "их", "чем", "была", "сам", "без", "будто", "раз", "тоже", "под",
    "будет", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом",
    "один", "почти", "мой", "тем", "чтобы", "сейчас", "были", "куда", "зачем", "всех", "можно", "при",
}

EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "in", "on", "at", "of",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "with", "from", "by", "as", "about", "into", "over", "after", "before", "between", "under", "again",
}

STOPWORDS = RUS_STOPWORDS | EN_STOPWORDS


class EntityBased:
    """
    Entity-based retrieval (TF-IDF + keyword/entity overlap).
    Это keyword/entity-level retriever, не graph-движок.
    """

    def __init__(
        self,
        min_entity_length: int = 3,
        max_entities_per_chunk: int = 16,
        tfidf_weight: float = 0.8,
        entity_overlap_weight: float = 0.2,
        min_score: float = 0.03,
        mmr_lambda: float = 0.74,
    ):
        self.min_entity_length = int(min_entity_length)
        self.max_entities_per_chunk = int(max_entities_per_chunk)
        self.tfidf_weight = float(max(0.0, tfidf_weight))
        self.entity_overlap_weight = float(max(0.0, entity_overlap_weight))
        self.min_score = float(max(0.0, min_score))
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))

        self.chunks: List[Dict] = []
        self.chunk_entities: List[List[str]] = []
        self.chunk_tokens: List[set[str]] = []
        self.entity_to_chunks: dict[str, list[int]] = defaultdict(list)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-]{2,}\b",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        return smart_chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=max(80, chunk_size // 6),
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF\-]{1,}", text or "")

    @staticmethod
    def _normalize_token(token: str) -> str:
        value = str(token or "").strip().lower().strip("-_")
        if not value:
            return ""

        ru_suffixes = (
            "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими",
            "ая", "яя", "ое", "ее", "ые", "ие", "ий", "ый", "ой",
            "ам", "ям", "ах", "ях", "ов", "ев", "ом", "ем", "ую", "юю",
            "а", "я", "ы", "и", "е", "о", "у", "ю",
        )
        if re.search(r"[\u0400-\u04FF]", value) and len(value) >= 5:
            for suffix in ru_suffixes:
                if value.endswith(suffix) and len(value) - len(suffix) >= 3:
                    value = value[: -len(suffix)]
                    break

        en_suffixes = ("ing", "edly", "ed", "ies", "es", "s")
        if re.search(r"[a-z]", value) and len(value) >= 5:
            for suffix in en_suffixes:
                if value.endswith(suffix) and len(value) - len(suffix) >= 3:
                    value = f"{value[:-3]}y" if suffix == "ies" else value[: -len(suffix)]
                    break
        return value

    @classmethod
    def _content_tokens_list(cls, text: str) -> List[str]:
        out: List[str] = []
        for token in cls._tokenize(text):
            norm = cls._normalize_token(token)
            if len(norm) < 2 or norm in STOPWORDS:
                continue
            out.append(norm)
        return out

    @classmethod
    def _bigrams_from_tokens(cls, tokens: List[str]) -> set[str]:
        if len(tokens) < 2:
            return set()
        result = set()
        for i in range(len(tokens) - 1):
            left = tokens[i].strip()
            right = tokens[i + 1].strip()
            if not left or not right:
                continue
            result.add(f"{left} {right}")
        return result

    @staticmethod
    def _normalize_feature_list(values: List[float]) -> List[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
        if span <= 1e-9:
            return [1.0 if v > 0 else 0.0 for v in values]
        return [max(0.0, min(1.0, (float(v) - minimum) / span)) for v in values]

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _entity_rarity(self, shared_entities: set[str]) -> float:
        if not shared_entities or not self.chunks:
            return 0.0
        n_chunks = max(1, len(self.chunks))
        max_idf = max(1e-9, math.log(1.0 + n_chunks))
        scores: List[float] = []
        for entity in shared_entities:
            df = len(self.entity_to_chunks.get(entity, []))
            idf = math.log(1.0 + (n_chunks / (1.0 + df)))
            scores.append(max(0.0, min(1.0, idf / max_idf)))
        return sum(scores) / len(scores) if scores else 0.0

    def _diversify_mmr(self, ranked: List[Dict], top_k: int) -> List[Dict]:
        if not ranked:
            return []
        limit = max(1, int(top_k))
        if len(ranked) <= limit:
            return ranked[:limit]

        remaining = list(ranked)
        selected: List[Dict] = []

        while remaining and len(selected) < limit:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_idx = 0
            best_mmr = -10.0
            for idx, candidate in enumerate(remaining):
                cand_score = float(candidate.get("score", 0.0))
                cand_tokens = set(candidate.get("tokens", set()))
                max_sim = 0.0
                for chosen in selected:
                    max_sim = max(max_sim, self._jaccard(cand_tokens, set(chosen.get("tokens", set()))))
                mmr = self.mmr_lambda * cand_score - (1.0 - self.mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    def _extract_entities_from_text(self, text: str) -> List[str]:
        if not text:
            return []

        named_candidates = re.findall(
            r"\b(?:[A-Z\u0410-\u042F\u0401][a-z\u0430-\u044F\u0451]{2,})(?:-[A-Z\u0410-\u042F\u0401]?[a-z\u0430-\u044F\u0451]{2,})?"
            r"(?:\s+(?:[A-Z\u0410-\u042F\u0401][a-z\u0430-\u044F\u0451]{2,})(?:-[A-Z\u0410-\u042F\u0401]?[a-z\u0430-\u044F\u0451]{2,})?){0,2}\b",
            text,
        )
        named_tokens = [self._normalize_token(item) for item in named_candidates if item.strip()]

        filtered = [
            token for token in self._content_tokens_list(text)
            if len(token) >= self.min_entity_length and token not in STOPWORDS
        ]
        freqs = Counter(filtered)

        bigrams: Counter[str] = Counter()
        for i in range(len(filtered) - 1):
            left = filtered[i]
            right = filtered[i + 1]
            if left == right:
                continue
            bigrams[f"{left} {right}"] += 1

        scores: Dict[str, float] = {}
        for value in named_tokens:
            if len(value) < self.min_entity_length:
                continue
            scores[value] = scores.get(value, 0.0) + 2.0
            for part in value.split(" "):
                part = part.strip()
                if len(part) >= self.min_entity_length and part not in STOPWORDS:
                    scores[part] = scores.get(part, 0.0) + 0.65
        for value, count in freqs.items():
            scores[value] = scores.get(value, 0.0) + float(count)
        for value, count in bigrams.items():
            scores[value] = scores.get(value, 0.0) + 0.8 * float(count)

        ranked = sorted(
            scores.items(),
            key=lambda x: (float(x[1]), len(x[0])),
            reverse=True,
        )

        result: List[str] = []
        seen = set()
        for value, _ in ranked:
            item = value.strip().lower()
            if len(item) < self.min_entity_length:
                continue
            if item in STOPWORDS or item in seen:
                continue
            seen.add(item)
            result.append(item)
            if len(result) >= self.max_entities_per_chunk:
                break
        return result

    def add_chunk(
        self,
        chunk: str,
        chunk_id: int | str | None = None,
        doc_id: str = "unknown",
        metadata: Dict | None = None,
    ) -> None:
        if not chunk.strip():
            return

        if chunk_id is None:
            chunk_id = len(self.chunks)

        entities = self._extract_entities_from_text(chunk)
        meta = dict(metadata or {})
        meta.setdefault("source", doc_id)

        payload = {
            "id": chunk_id,
            "doc_id": doc_id,
            "text": chunk,
            "metadata": meta,
            "entities": entities,
        }
        self.chunks.append(payload)
        self.chunk_entities.append(entities)
        self.chunk_tokens.append(set(self._content_tokens_list(chunk)))

        idx = len(self.chunks) - 1
        for entity in set(entities):
            self.entity_to_chunks[entity].append(idx)

        self.tfidf_matrix = None

    def add_document(
        self,
        text: str,
        doc_id: str = "doc1",
        metadata: Dict | None = None,
        chunk_size: int = 900,
        overlap: int = 180,
    ) -> None:
        chunks = self._split_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            chunk_meta = {**(metadata or {}), "chunk_index": i, "doc_id": doc_id, "source": doc_id}
            self.add_chunk(chunk=chunk, chunk_id=f"{doc_id}_chunk_{i}", doc_id=doc_id, metadata=chunk_meta)

    def build_index(self) -> None:
        if not self.chunks:
            self.tfidf_matrix = None
            return
        corpus = [item["text"] for item in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if not query.strip() or not self.chunks:
            return []

        if self.tfidf_matrix is None:
            self.build_index()
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        norm_cosine = self._normalize_feature_list([float(x) for x in cosine_sims])

        query_entities = set(self._extract_entities_from_text(query))
        query_tokens_list = self._content_tokens_list(query)
        query_tokens = set(query_tokens_list)
        query_phrase = " ".join(query_tokens_list) if len(query_tokens_list) >= 2 else ""
        query_bigrams = self._bigrams_from_tokens(query_tokens_list)

        raw_tfidf_w = max(0.0, self.tfidf_weight)
        raw_overlap_w = max(0.0, self.entity_overlap_weight)
        raw_coverage_w = 0.16
        raw_rarity_w = 0.08
        raw_bigram_w = 0.08
        raw_phrase_w = 0.04 if query_phrase else 0.0

        weight_sum = raw_tfidf_w + raw_overlap_w + raw_coverage_w + raw_rarity_w + raw_bigram_w + raw_phrase_w
        if weight_sum <= 0:
            raw_tfidf_w = 1.0
            weight_sum = 1.0

        tfidf_w = raw_tfidf_w / weight_sum
        overlap_w = raw_overlap_w / weight_sum
        coverage_w = raw_coverage_w / weight_sum
        rarity_w = raw_rarity_w / weight_sum
        bigram_w = raw_bigram_w / weight_sum
        phrase_w = raw_phrase_w / weight_sum

        ranked: List[Dict] = []
        for idx, _ in enumerate(cosine_sims):
            item = self.chunks[idx]
            tfidf_score = float(norm_cosine[idx]) if idx < len(norm_cosine) else 0.0
            chunk_entities = set(item.get("entities", []))
            chunk_tokens = self.chunk_tokens[idx] if idx < len(self.chunk_tokens) else set()

            if query_entities:
                shared_entities = query_entities & chunk_entities
                overlap_score = len(shared_entities) / len(query_entities)
            else:
                shared_entities = set()
                overlap_score = 0.0

            if query_tokens:
                overlap_count = len(query_tokens & chunk_tokens)
                token_coverage = overlap_count / len(query_tokens)
                token_density = overlap_count / max(1, len(chunk_tokens))
                coverage_score = 0.75 * token_coverage + 0.25 * token_density
            else:
                coverage_score = 0.0

            rarity_score = self._entity_rarity(shared_entities)

            chunk_tokens_ordered = self._content_tokens_list(str(item.get("text", "")))
            chunk_bigrams = self._bigrams_from_tokens(chunk_tokens_ordered)
            bigram_score = (len(query_bigrams & chunk_bigrams) / len(query_bigrams)) if query_bigrams else 0.0

            phrase_score = 0.0
            if query_phrase:
                normalized_chunk = " ".join(chunk_tokens_ordered)
                if query_phrase in normalized_chunk:
                    phrase_score = 1.0

            final_score = (
                tfidf_w * tfidf_score
                + overlap_w * float(overlap_score)
                + coverage_w * float(coverage_score)
                + rarity_w * float(rarity_score)
                + bigram_w * float(bigram_score)
                + phrase_w * float(phrase_score)
            )

            dynamic_min_score = self.min_score * (0.65 if query_entities else 1.0)
            if final_score < dynamic_min_score:
                continue

            if query_tokens and coverage_score <= 0 and overlap_score <= 0 and tfidf_score < 0.12:
                continue

            ranked.append(
                {
                    "payload": item,
                    "score": max(0.0, float(final_score)),
                    "tokens": chunk_tokens,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        diverse = self._diversify_mmr(ranked, top_k=max(1, int(top_k)))

        output: List[Tuple[Dict, float]] = []
        for row in diverse:
            output.append((row["payload"], float(row["score"])))
        return output

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        self.chunks.clear()
        self.chunk_entities.clear()
        self.chunk_tokens.clear()
        self.entity_to_chunks.clear()
        self.tfidf_matrix = None

