import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

class ChunkBased:
    """
    Chunk-based подход для RAG.
    - Разбивает текст на чанки с пересечением (overlap)
    - Использует embeddings (sentence-transformers)
    - Хранит в Chroma (векторная база)
    - Полностью совместим с EntityBased
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "knowledge_base"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict] = []                    # список всех чанков
        self.client = chromadb.PersistentClient(path="./chroma_db")  # сохраняется на диске
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Разбивает текст на чанки с пересечением."""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap
            if len(chunk) < chunk_size // 2:  # последний маленький чанк не нужен
                break
        return chunks

    def add_document(self, text: str, doc_id: str = "doc1", metadata: Dict = None) -> None:
        """Добавляет целый документ (книгу) — разбивает на чанки и индексирует."""
        if metadata is None:
            metadata = {"source": doc_id}

        chunks = self._split_text(text, self.chunk_size, self.overlap)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": i}
            })
            
            self.collection.add(
                documents=[chunk_text],
                ids=[chunk_id],
                metadatas=[{**metadata, "chunk_index": i}]
            )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Поиск по embeddings + косинусному сходству."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        for i in range(len(results["documents"][0])):
            chunk_text = results["documents"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            
            # Превращаем distance в score (0-1, чем выше — лучше)
            score = 1 - min(distance, 2.0) / 2.0
            
            output.append((
                {"id": metadata.get("chunk_index"), "text": chunk_text, "metadata": metadata},
                score
            ))
        return output

    def build_index(self):
        """Для совместимости с EntityBased — Chroma индексирует автоматически."""
        pass

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        """Полная очистка."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        self.chunks.clear()

    def visualize_search(self, query: str, top_k: int = 5, save_path: str = "search_visualization.png"):
        """Красивая визуализация: барчарт релевантности + примеры чанков."""
        results = self.search(query, top_k)
        
        scores = [score for _, score in results]
        chunk_indices = [r[0]["metadata"].get("chunk_index", i) for i, r in enumerate(results)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(scores)), scores, color='skyblue')
        plt.title(f'Релевантность чанков для запроса: "{query}"')
        plt.xlabel('Чанк (индекс)')
        plt.ylabel('Score (0–1)')
        plt.ylim(0, 1)
        
        # Подписываем топ-чанк
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'Chunk {chunk_indices[i]}\n{scores[i]:.2f}', ha='center')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        print(f"✅ Визуализация сохранена: {save_path}")
        # Показываем лучший чанк
        best = results[0]
        print(f"\n🔥 Лучший чанк (score {best[1]:.3f}):\n{best[0]['text'][:500]}...\n")

# ==================== Пример использования внизу файла ====================
if __name__ == "__main__":
    print("ChunkBased готов к использованию!")