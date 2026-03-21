import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple

class ChunkBased:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200, collection_name: str = "harry_potter_collection"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        # твой старый метод (без изменений)
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap
            if len(chunk) < chunk_size // 2: break
        return chunks

    def add_document(self, text: str, doc_id: str, metadata: Dict = None):
        # твой старый метод (без изменений)
        ...

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        output = []
        for i in range(len(results["documents"][0])):
            chunk_text = results["documents"][0][i]
            distance = results["distances"][0][i]
            score = 1 - min(distance, 2.0) / 2.0
            metadata = results["metadatas"][0][i]
            output.append(({"text": chunk_text, "metadata": metadata}, score))
        return output