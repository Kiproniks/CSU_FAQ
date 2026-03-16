import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EntityBased:
    """
    Entity-based подход для RAG.
    Извлекает ключевые слова (сущности) и использует TF-IDF + cosine similarity.
    """

    def __init__(self, min_entity_length: int = 2, max_entities_per_chunk: int = 10):
        self.min_entity_length = min_entity_length
        self.max_entities_per_chunk = max_entities_per_chunk
        
        self.chunks: List[Dict] = []           # [{'id': , 'text': }]
        self.chunk_entities: List[List[str]] = []   # список сущностей для каждого чанка
        self.entity_to_chunks = defaultdict(list)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[a-zа-яё]{2,}\b',
            max_features=5000
        )
        self.tfidf_matrix = None

    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r'\b[a-zа-яё]+\b', text)

    def _extract_entities(self, words: List[str]) -> List[str]:
        """Извлечение самых частых сущностей"""
        if not words:
            return []
        word_freq = Counter(words)
        entities = [
            word for word in set(words) 
            if len(word) >= self.min_entity_length
        ]
        sorted_entities = sorted(entities, key=lambda x: word_freq[x], reverse=True)
        return sorted_entities[:self.max_entities_per_chunk]

    def add_chunk(self, chunk: str, chunk_id: int = None) -> None:
        """Добавление чанка в базу данных."""
        if chunk_id is None:
            chunk_id = len(self.chunks)

        words = self._preprocess_text(chunk)
        entities = self._extract_entities(words)

        self.chunks.append({
            'id': chunk_id, 
            'text': chunk,
            'entities': entities          # ← добавили
        })

        self.chunk_entities.append(entities)

        for entity in entities:
            self.entity_to_chunks[entity].append(chunk_id)

    def build_index(self) -> None:
        corpus = [chunk['text'] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self.tfidf_matrix is None:
            self.build_index()

        query_vec = self.vectorizer.transform([query])
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(cosine_sims)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if cosine_sims[idx] > 0.05:   # небольшой порог, чтобы не брать совсем слабые совпадения
                results.append((self.chunks[idx], float(cosine_sims[idx])))
        return results

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        self.chunks.clear()
        self.chunk_entities.clear()
        self.entity_to_chunks.clear()
        self.tfidf_matrix = None