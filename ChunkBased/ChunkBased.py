import re
from typing import Dict, List, Tuple

import chromadb
import matplotlib.pyplot as plt
from chromadb.utils import embedding_functions


class ChunkBased:
    """Chunk-based retrieval with sentence-aware chunking and Chroma storage."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "knowledge_base",
        chroma_path: str = "./chroma_db",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict] = []
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with paragraph/sentence boundaries and overlap fallback."""
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

    def add_document(self, text: str, doc_id: str = "doc1", metadata: Dict = None) -> None:
        if metadata is None:
            metadata = {"source": doc_id}

        chunks = self._split_text(text, self.chunk_size, self.overlap)

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = {**metadata, "chunk_index": i}
            self.chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

            self.collection.add(
                documents=[chunk_text],
                ids=[chunk_id],
                metadatas=[chunk_metadata],
            )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i in range(len(documents)):
            chunk_text = documents[i]
            distance = distances[i]
            metadata = metadatas[i]
            score = 1 - min(distance, 2.0) / 2.0

            output.append(
                (
                    {
                        "id": metadata.get("chunk_index"),
                        "text": chunk_text,
                        "metadata": metadata,
                    },
                    score,
                )
            )
        return output

    def build_index(self):
        # Chroma indexes on add; method is kept for API compatibility.
        return None

    def get_chunks(self) -> List[Dict]:
        return self.chunks

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function,
        )
        self.chunks.clear()

    def visualize_search(
        self,
        query: str,
        top_k: int = 5,
        save_path: str = "search_visualization.png",
        show: bool = False,
    ):
        results = self.search(query, top_k)
        if not results:
            return

        scores = [score for _, score in results]
        chunk_indices = [
            r[0]["metadata"].get("chunk_index", i)
            for i, r in enumerate(results)
        ]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(scores)), scores, color="skyblue")
        plt.title(f'Chunk relevance for query: "{query}"')
        plt.xlabel("Chunk index")
        plt.ylabel("Score (0-1)")
        plt.ylim(0, 1)

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"Chunk {chunk_indices[i]}\\n{scores[i]:.2f}",
                ha="center",
            )

        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    print("ChunkBased is ready")
