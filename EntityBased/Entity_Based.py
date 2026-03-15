import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EntityBased:
    """
    Оптимизированная версия класса для разбиения текста на сущности, группировки чанков
    и поиска релевантных чанков для пользовательского запроса.
    Использует TF‑IDF и косинусное сходство для более точного поиска.
    """

    def __init__(self, min_entity_length: int = 2, max_entities_per_chunk: int = 10):
        self.min_entity_length = min_entity_length
        self.max_entities_per_chunk = max_entities_per_chunk
        self.chunks = []  # список всех чанков
        self.chunk_entities = []  # список сущностей для каждого чанка
        self.entity_to_chunks = defaultdict(list)  # отображение сущности на список чанков

        # Для TF‑IDF векторизации
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[a-zа-яё]{2,}\b',  # слова от 2 букв
            max_features=5000  # ограничение на количество признаков
        )
        self.tfidf_matrix = None  # матрица TF‑IDF для всех чанков

    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        """Предварительная обработка текста."""
        text = text.lower()
        words = re.findall(r'\b[a-zа-яё]+\b', text)
        return words

    def _extract_entities(self, words: List[str]) -> List[str]:
        """Извлечение сущностей (ключевых слов) из списка слов."""
        entities = [
            word for word in set(words)
            if len(word) >= self.min_entity_length
        ]
        word_freq = Counter(words)
        sorted_entities = sorted(
            entities,
            key=lambda x: word_freq[x],
            reverse=True
        )
        return sorted_entities[:self.max_entities_per_chunk]

    def add_chunk(self, chunk: str, chunk_id: int = None) -> None:
        """Добавление чанка в базу данных."""
        if chunk_id is None:
            chunk_id = len(self.chunks)

        self.chunks.append({'id': chunk_id, 'text': chunk})
        words = self._preprocess_text(chunk)
        entities = self._extract_entities(words)
        self.chunk_entities.append(entities)

        for entity in entities:
            self.entity_to_chunks[entity].append(chunk_id)

    def build_index(self) -> None:
        """Построение TF‑IDF индекса для всех чанков."""
        corpus = [chunk['text'] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Поиск наиболее релевантных чанков для запроса с использованием TF‑IDF и косинусного сходства.
        """
        if self.tfidf_matrix is None:
            self.build_index()

        # Векторизация запроса
        query_vec = self.vectorizer.transform([query])

        # Расчёт косинусного сходства
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Сортировка и отбор топ‑K результатов
        top_indices = np.argsort(cosine_sims)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if cosine_sims[idx] > 0:  # игнорируем нулевые сходства
                results.append((self.chunks[idx], cosine_sims[idx]))

        return results

    def get_chunks(self) -> List[Dict]:
        """Получить все добавленные чанки."""
        return self.chunks

    def clear(self) -> None:
        """Очистить базу данных чанков и сущностей."""
        self.chunks.clear()
        self.chunk_entities.clear()
        self.entity_to_chunks.clear()
        self.tfidf_matrix = None
