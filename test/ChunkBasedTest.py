import os
from typing import List, Dict, Tuple

import chromadb
from chromadb.utils import embedding_functions


class ChunkBased:
    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
        collection_name: str = "harry_potter_collection",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        base_dir = os.path.dirname(__file__)
        self.db_path = os.path.join(base_dir, "chroma_db_test")

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        step = max(1, chunk_size - overlap)

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if len(chunk) < chunk_size // 2:
                break

            start += step

        return chunks

    def add_document(self, text: str, doc_id: str, metadata: Dict = None):
        if metadata is None:
            metadata = {}

        chunks = self._split_text(text, self.chunk_size, self.overlap)
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_id}_{i}")
            documents.append(chunk)
            metadatas.append({
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            chunk_text = docs[i]
            distance = distances[i]
            score = 1 - min(distance, 2.0) / 2.0
            metadata = metas[i] if metas else {}
            output.append(({"text": chunk_text, "metadata": metadata}, score))

        return output