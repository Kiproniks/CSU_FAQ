import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EntityBased:
    def __init__(self, min_entity_length: int = 2):
        self.min_entity_length = min_entity_length
        self.chunks: List[Dict] = []
        self.entity_to_chunks = defaultdict(list)
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b[a-zа-яё]{2,}\b', max_features=5000)
        self.tfidf_matrix = None

    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        return re.findall(r'\b[a-zа-яё]+\b', text.lower())

    def _extract_entities(self, words: List[str]) -> List[str]:
        if not words: return []
        word_freq = Counter(words)
        entities = [w for w in set(words) if len(w) >= self.min_entity_length]
        return sorted(entities, key=lambda x: word_freq[x], reverse=True)[:10]

    def add_chunk(self, chunk: str, chunk_id: int):
        words = self._preprocess_text(chunk)
        entities = self._extract_entities(words)
        self.chunks.append({'id': chunk_id, 'text': chunk, 'entities': entities})
        for e in entities:
            self.entity_to_chunks[e].append(chunk_id)

    def build_index(self):
        corpus = [c['text'] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float, int]]:
        """Возвращает (chunk, tfidf_score, entity_match_count)"""
        if self.tfidf_matrix is None: self.build_index()

        # 1. TF-IDF cosine (как раньше)
        query_vec = self.vectorizer.transform([query])
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 2. Entity matching (новое!)
        query_words = self._preprocess_text(query)
        query_entities = set(self._extract_entities(query_words))

        results = []
        for idx in np.argsort(cosine_sims)[-top_k:][::-1]:
            if cosine_sims[idx] <= 0.05: continue
            chunk = self.chunks[idx]
            entity_overlap = len(query_entities & set(chunk['entities']))
            results.append((chunk, float(cosine_sims[idx]), entity_overlap))
        return results