from __future__ import annotations

import re
from typing import Dict, List, Tuple

import chromadb
import matplotlib.pyplot as plt
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.text_splitter import dot_dot_chunk_text, smart_chunk_text

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


class ChunkBased:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "knowledge_base",
        chroma_path: str = "./chroma_db",
        min_chunk_size: int = 200,
        min_chunk_floor: int = 250,
        splitter_mode: str = "smart",
        upsert_batch_size: int = 1000,
        mmr_lambda: float = 0.78,
    ):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.min_chunk_size = int(min_chunk_size)
        self.min_chunk_floor = int(min_chunk_floor)
        self.splitter_mode = str(splitter_mode or "smart").strip().lower()
        self.upsert_batch_size = max(1, int(upsert_batch_size))
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))

        self.chunks: List[Dict] = []
        self._chunk_by_id: Dict[str, Dict] = {}
        self._doc_chunk_index_map: Dict[tuple[str, int], str] = {}

        self._sparse_vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-]{2,}\b",
            ngram_range=(1, 2),
            max_features=35000,
            sublinear_tf=True,
        )
        self._sparse_matrix = None

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    @staticmethod
    def _split_text(
        text: str,
        chunk_size: int,
        overlap: int,
        min_chunk_floor: int = 250,
        splitter_mode: str = "smart",
    ) -> List[str]:
        mode = str(splitter_mode or "smart").strip().lower()
        if mode == "dot_dot":
            return dot_dot_chunk_text(
                text=text,
                chunk_size=chunk_size,
                neighbor_overlap_sentences=1,
                min_chunk_floor=min_chunk_floor,
            )
        return smart_chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=max(80, chunk_size // 5),
            min_chunk_floor=min_chunk_floor,
        )

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        value = max(0.0, float(distance))
        return 1.0 / (1.0 + value)

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
        raw = re.findall(r"[A-Za-z\u0400-\u04FF0-9\-]{2,}", text or "")
        out: List[str] = []
        for token in raw:
            norm = cls._normalize_token(token)
            if len(norm) < 2 or norm in STOPWORDS:
                continue
            out.append(norm)
        return out

    @classmethod
    def _content_tokens_set(cls, text: str) -> set[str]:
        return set(cls._content_tokens_list(text))

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
    def _normalize_feature_map(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        minimum = min(values.values())
        maximum = max(values.values())
        span = maximum - minimum
        if span <= 1e-9:
            return {k: (1.0 if v > 0 else 0.0) for k, v in values.items()}
        return {k: max(0.0, min(1.0, (float(v) - minimum) / span)) for k, v in values.items()}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _keyword_importance_score(self, query_tokens: List[str], chunk_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0

        vocab = getattr(self._sparse_vectorizer, "vocabulary_", {}) or {}
        idf = getattr(self._sparse_vectorizer, "idf_", None)

        total = 0.0
        matched = 0.0
        for token in set(query_tokens):
            weight = 1.0
            if token in vocab and idf is not None:
                try:
                    weight = float(idf[vocab[token]])
                except Exception:
                    weight = 1.0
            total += weight
            if token in chunk_tokens:
                matched += weight

        return 0.0 if total <= 0 else matched / total

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

            best_index = 0
            best_value = -10.0
            for idx, candidate in enumerate(remaining):
                cand_score = float(candidate.get("score", 0.0))
                cand_tokens = set(candidate.get("tokens", set()))
                max_sim = 0.0
                for chosen in selected:
                    max_sim = max(max_sim, self._jaccard(cand_tokens, set(chosen.get("tokens", set()))))
                mmr_value = self.mmr_lambda * cand_score - (1.0 - self.mmr_lambda) * max_sim
                if mmr_value > best_value:
                    best_value = mmr_value
                    best_index = idx

            selected.append(remaining.pop(best_index))

        return selected

    def _mark_sparse_dirty(self) -> None:
        self._sparse_matrix = None

    def _rebuild_doc_chunk_index_map(self) -> None:
        self._doc_chunk_index_map.clear()
        for item in self.chunks:
            metadata = dict(item.get("metadata", {}) or {})
            doc_id = str(metadata.get("doc_id") or metadata.get("source") or "")
            try:
                chunk_index = int(metadata.get("chunk_index"))
            except Exception:
                continue
            if not doc_id:
                continue
            self._doc_chunk_index_map[(doc_id, chunk_index)] = str(item.get("id", ""))

    def _ensure_local_cache_from_collection(self) -> None:
        if self.chunks:
            return
        try:
            total = self.collection.count()
        except Exception:
            return
        if total <= 0:
            return

        restored: List[Dict] = []
        batch_size = 1000
        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []
            ids = batch.get("ids") or []

            for i, text in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                chunk_id = str(ids[i]) if i < len(ids) else f"restored_chunk_{offset + i}"
                restored.append(
                    {
                        "id": chunk_id,
                        "text": text or "",
                        "metadata": metadata or {},
                    }
                )

        self.chunks.extend(restored)
        self._chunk_by_id = {str(item["id"]): item for item in self.chunks}
        self._rebuild_doc_chunk_index_map()
        self._mark_sparse_dirty()

    def _ensure_sparse_index(self) -> None:
        if self._sparse_matrix is not None:
            return
        if not self.chunks:
            self._sparse_matrix = None
            return
        corpus = [str(item.get("text", "") or "") for item in self.chunks]
        self._sparse_matrix = self._sparse_vectorizer.fit_transform(corpus)

    def _inject_neighbor_candidates(self, candidates: Dict[str, Dict], limit_seed: int = 12) -> None:
        if not candidates:
            return
        self._rebuild_doc_chunk_index_map()
        if not self._doc_chunk_index_map:
            return

        seed = sorted(
            candidates.values(),
            key=lambda item: float(item.get("dense_raw", 0.0)) + float(item.get("sparse_raw", 0.0)),
            reverse=True,
        )[: max(1, int(limit_seed))]

        for row in seed:
            metadata = dict(row.get("metadata", {}) or {})
            doc_id = str(metadata.get("doc_id") or metadata.get("source") or "")
            try:
                chunk_index = int(metadata.get("chunk_index"))
            except Exception:
                continue
            if not doc_id:
                continue

            for shift in (-1, 1):
                neighbor_id = self._doc_chunk_index_map.get((doc_id, chunk_index + shift))
                if not neighbor_id or neighbor_id in candidates:
                    continue
                payload = self._chunk_by_id.get(neighbor_id)
                if not payload:
                    continue
                candidates[neighbor_id] = {
                    "id": neighbor_id,
                    "text": str(payload.get("text", "") or ""),
                    "metadata": dict(payload.get("metadata", {}) or {}),
                    "dense_raw": 0.35 * float(row.get("dense_raw", 0.0)),
                    "sparse_raw": 0.35 * float(row.get("sparse_raw", 0.0)),
                }

    def add_document(self, text: str, doc_id: str = "doc1", metadata: Dict | None = None) -> None:
        base_metadata = dict(metadata or {})
        base_metadata.setdefault("source", doc_id)

        chunks = self._split_text(
            text=text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            min_chunk_floor=self.min_chunk_floor,
            splitter_mode=self.splitter_mode,
        )
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for index, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{index}"
            chunk_metadata = {**base_metadata, "chunk_index": index, "doc_id": doc_id}
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append(chunk_metadata)

            payload = {"id": chunk_id, "text": chunk_text, "metadata": chunk_metadata}
            self.chunks.append(payload)
            self._chunk_by_id[str(chunk_id)] = payload
            self._doc_chunk_index_map[(str(doc_id), int(index))] = str(chunk_id)

        batch = self.upsert_batch_size
        for start in range(0, len(ids), batch):
            end = start + batch
            self.collection.upsert(
                documents=documents[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )

        self._mark_sparse_dirty()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []

        self._ensure_local_cache_from_collection()
        if not self.chunks:
            return []

        query_tokens_ordered = self._content_tokens_list(query)
        query_tokens = set(query_tokens_ordered)
        query_phrase = " ".join(query_tokens_ordered) if len(query_tokens_ordered) >= 2 else ""
        query_bigrams = self._bigrams_from_tokens(query_tokens_ordered)

        try:
            total = self.collection.count()
        except Exception:
            total = max(1, int(top_k))
        dense_n = min(max(max(1, int(top_k)) * 12, 30), max(1, total))

        try:
            dense = self.collection.query(
                query_texts=[query],
                n_results=dense_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            dense = {}

        documents = (dense.get("documents") or [[]])[0]
        metadatas = (dense.get("metadatas") or [[]])[0]
        distances = (dense.get("distances") or [[]])[0]
        ids = (dense.get("ids") or [[]])[0]

        candidates: Dict[str, Dict] = {}
        for i, chunk_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            dense_score = self._distance_to_score(distance)
            chunk_id = str(ids[i]) if i < len(ids) else str(
                f"{metadata.get('doc_id', 'doc')}_chunk_{metadata.get('chunk_index', i)}"
            )
            candidates[chunk_id] = {
                "id": chunk_id,
                "text": chunk_text or "",
                "metadata": metadata or {},
                "dense_raw": float(dense_score),
                "sparse_raw": 0.0,
            }

        self._ensure_sparse_index()
        if self._sparse_matrix is not None and self.chunks:
            try:
                query_vec = self._sparse_vectorizer.transform([query])
                sparse_scores = cosine_similarity(query_vec, self._sparse_matrix).flatten()
                top_sparse_idx = sorted(
                    range(len(sparse_scores)),
                    key=lambda idx: float(sparse_scores[idx]),
                    reverse=True,
                )[: max(30, int(top_k) * 12)]

                for idx in top_sparse_idx:
                    score = float(sparse_scores[idx])
                    if score <= 0:
                        continue
                    item = self.chunks[idx]
                    chunk_id = str(item.get("id", f"sparse_chunk_{idx}"))
                    if chunk_id not in candidates:
                        candidates[chunk_id] = {
                            "id": chunk_id,
                            "text": str(item.get("text", "") or ""),
                            "metadata": dict(item.get("metadata", {}) or {}),
                            "dense_raw": 0.0,
                            "sparse_raw": score,
                        }
                    else:
                        current_sparse = float(candidates[chunk_id].get("sparse_raw", 0.0))
                        candidates[chunk_id]["sparse_raw"] = max(current_sparse, score)
            except Exception:
                pass

        self._inject_neighbor_candidates(candidates)
        if not candidates:
            return []

        dense_map = {cid: float(item.get("dense_raw", 0.0)) for cid, item in candidates.items()}
        sparse_map = {cid: float(item.get("sparse_raw", 0.0)) for cid, item in candidates.items()}
        norm_dense = self._normalize_feature_map(dense_map)
        norm_sparse = self._normalize_feature_map(sparse_map)

        qlen = len(query_tokens_ordered)
        if qlen <= 2:
            weights = {
                "dense": 0.45,
                "sparse": 0.25,
                "coverage": 0.12,
                "density": 0.05,
                "importance": 0.08,
                "bigram": 0.03,
                "phrase": 0.02,
            }
        else:
            weights = {
                "dense": 0.36,
                "sparse": 0.22,
                "coverage": 0.17,
                "density": 0.07,
                "importance": 0.11,
                "bigram": 0.05,
                "phrase": 0.02,
            }

        ranked: List[Dict] = []
        for chunk_id, payload in candidates.items():
            text = str(payload.get("text", "") or "")
            tokens_ordered = self._content_tokens_list(text)
            chunk_tokens = set(tokens_ordered)
            overlap_count = len(query_tokens & chunk_tokens) if query_tokens else 0
            coverage = overlap_count / len(query_tokens) if query_tokens else 0.0
            density = overlap_count / max(1, len(chunk_tokens)) if chunk_tokens else 0.0
            importance = self._keyword_importance_score(query_tokens_ordered, chunk_tokens)

            chunk_bigrams = self._bigrams_from_tokens(tokens_ordered)
            bigram_overlap = (len(query_bigrams & chunk_bigrams) / len(query_bigrams)) if query_bigrams else 0.0

            phrase_bonus = 0.0
            if query_phrase:
                normalized_chunk = " ".join(tokens_ordered)
                if query_phrase in normalized_chunk:
                    phrase_bonus = 1.0

            score = (
                weights["dense"] * float(norm_dense.get(chunk_id, 0.0))
                + weights["sparse"] * float(norm_sparse.get(chunk_id, 0.0))
                + weights["coverage"] * float(coverage)
                + weights["density"] * float(density)
                + weights["importance"] * float(importance)
                + weights["bigram"] * float(bigram_overlap)
                + weights["phrase"] * float(phrase_bonus)
            )

            length_ratio = len(text) / max(1, self.chunk_size)
            length_prior = max(0.0, 1.0 - min(1.0, abs(length_ratio - 1.0) / 1.5))
            score *= (0.90 + 0.10 * length_prior)

            if len(text) < max(120, self.min_chunk_size // 2):
                score *= 0.9

            ranked.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": dict(payload.get("metadata", {}) or {}),
                    "tokens": chunk_tokens,
                    "score": max(0.0, float(score)),
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        diverse = self._diversify_mmr(ranked, top_k=max(1, int(top_k)))

        output: List[Tuple[Dict, float]] = []
        for item in diverse:
            output.append(
                (
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "metadata": item["metadata"],
                    },
                    float(item["score"]),
                )
            )
        return output

    def build_index(self) -> None:
        self._ensure_local_cache_from_collection()
        self._ensure_sparse_index()

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,
        )
        self.chunks.clear()
        self._chunk_by_id.clear()
        self._doc_chunk_index_map.clear()
        self._sparse_matrix = None

    def visualize_search(self, query: str, top_k: int = 5, save_path: str = "search_visualization.png") -> None:
        results = self.search(query, top_k)
        if not results:
            raise ValueError("No results to visualize.")

        scores = [score for _, score in results]
        labels = [str(item[0]["metadata"].get("chunk_index", i)) for i, item in enumerate(results)]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(scores)), scores, color="#4ca8af")
        plt.title(f'Relevance for query: "{query}"')
        plt.xlabel("Chunk")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        for index, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"#{labels[index]}\n{scores[index]:.2f}",
                ha="center",
            )

        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    print("ChunkBased ready")

