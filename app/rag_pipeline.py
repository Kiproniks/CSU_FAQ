from __future__ import annotations

from typing import Any, Dict, List, Literal

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.Entity_Based import EntityBased
from app.config import settings
from app.llm_service import LLMService

RetrievalMode = Literal["hybrid", "chunk", "entity"]


class RAGPipeline:
    def __init__(self) -> None:
        self.chunk_engine = self._init_chunk_engine()
        self.entity_engine = EntityBased()
        self.llm = LLMService()

        if settings.bootstrap_entity_from_chroma:
            self._bootstrap_entity_index_from_chroma()

    def _init_chunk_engine(self) -> ChunkBased:
        primary = ChunkBased(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            collection_name=settings.chunk_collection,
            chroma_path=settings.chroma_path,
        )

        try:
            primary_count = primary.collection.count()
        except Exception:
            return primary

        # Backward compatibility: previously hardcoded path/collection.
        if (
            primary_count == 0
            and (settings.chroma_path != "./chroma_db" or settings.chunk_collection != "harry_potter_collection")
        ):
            legacy = ChunkBased(
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                collection_name="harry_potter_collection",
                chroma_path="./chroma_db",
            )
            try:
                if legacy.collection.count() > 0:
                    return legacy
            except Exception:
                pass

        return primary

    def _bootstrap_entity_index_from_chroma(self) -> None:
        try:
            collection = self.chunk_engine.collection
            total = collection.count()
        except Exception:
            return

        if total == 0:
            return

        batch_size = 500
        for offset in range(0, total, batch_size):
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []

            for index, text in enumerate(documents):
                metadata = (metadatas[index] or {}) if index < len(metadatas) else {}
                source = metadata.get("source") or "unknown"
                self.entity_engine.add_chunk(
                    chunk=text,
                    chunk_id=len(self.entity_engine.chunks),
                    doc_id=source,
                    metadata=metadata,
                )

        self.entity_engine.build_index()

    def index_document(self, text: str, doc_id: str, metadata: Dict[str, Any] | None = None) -> None:
        safe_metadata = metadata or {}
        self.chunk_engine.add_document(text, doc_id=doc_id, metadata=safe_metadata)
        self.entity_engine.add_document(
            text,
            doc_id=doc_id,
            metadata=safe_metadata,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    @staticmethod
    def _normalize_chunk_results(raw_results: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in raw_results:
            if isinstance(item, tuple) and len(item) == 2:
                payload, score = item
                text = (payload or {}).get("text", "")
                metadata = (payload or {}).get("metadata", {})
            else:
                payload = item if isinstance(item, dict) else {}
                text = payload.get("text", "")
                metadata = payload.get("metadata", {})
                score = payload.get("score", 0.0)

            normalized.append(
                {
                    "strategy": "chunk-based",
                    "text": text,
                    "score": float(score),
                    "metadata": metadata or {},
                    "source": (metadata or {}).get("source", "unknown"),
                    "entities": [],
                }
            )
        return normalized

    @staticmethod
    def _normalize_entity_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in raw_results:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            normalized.append(
                {
                    "strategy": "entity-based-tfidf",
                    "text": item.get("text", "") if isinstance(item, dict) else "",
                    "score": float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                    "metadata": metadata or {},
                    "source": (metadata or {}).get("source", item.get("doc_id", "unknown")) if isinstance(item, dict) else "unknown",
                    "entities": item.get("entities", []) if isinstance(item, dict) else [],
                }
            )
        return normalized

    @staticmethod
    def _build_context(chunk_results: List[Dict[str, Any]], entity_results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []

        if chunk_results:
            parts.append("=== Chunk-based retrieval ===")
            for index, item in enumerate(chunk_results, start=1):
                parts.append(
                    f"{index}. score={item['score']:.4f} | source={item['source']}\n{item['text']}"
                )

        if entity_results:
            parts.append("=== Entity-based retrieval (TF-IDF, keyword-level) ===")
            for index, item in enumerate(entity_results, start=1):
                entities = ", ".join(item.get("entities", [])[:8])
                if not entities:
                    entities = "-"
                parts.append(
                    f"{index}. score={item['score']:.4f} | source={item['source']} | entities={entities}\n{item['text']}"
                )

        if not parts:
            return "No relevant context was found."

        return "\n\n".join(parts)

    def retrieve(
        self,
        query: str,
        top_k_chunks: int | None = None,
        top_k_entities: int | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        chunk_k = top_k_chunks or settings.top_k_chunks
        entity_k = top_k_entities or settings.top_k_entities

        raw_chunk_results = self.chunk_engine.search(query, top_k=chunk_k)
        raw_entity_results = self.entity_engine.search(query, top_k=entity_k)

        chunk_results = self._normalize_chunk_results(raw_chunk_results)
        entity_results = self._normalize_entity_results(raw_entity_results)
        combined = sorted(
            chunk_results + entity_results,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        return {
            "chunk_results": chunk_results,
            "entity_results": entity_results,
            "combined": combined,
        }

    @staticmethod
    def _normalize_mode(mode: str) -> RetrievalMode:
        value = (mode or "").strip().lower()
        if value in {"chunk", "chunkbased"}:
            return "chunk"
        if value in {"entity", "entitybased"}:
            return "entity"
        return "hybrid"

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        retrieval = self.retrieve(
            query=query,
            top_k_chunks=top_k,
            top_k_entities=top_k,
        )

        normalized_mode = self._normalize_mode(mode)

        selected_chunk_results = retrieval["chunk_results"]
        selected_entity_results = retrieval["entity_results"]
        selected_hits = retrieval["combined"]

        if normalized_mode == "chunk":
            selected_entity_results = []
            selected_hits = selected_chunk_results
        elif normalized_mode == "entity":
            selected_chunk_results = []
            selected_hits = selected_entity_results

        context = self._build_context(
            selected_chunk_results,
            selected_entity_results,
        )
        llm_result = self.llm.generate_answer(query=query, context=context)

        return {
            "question": query,
            "answer": llm_result.answer,
            "provider": llm_result.provider,
            "model": llm_result.model,
            "mode": normalized_mode,
            "hits": selected_hits,
            "chunk_results": retrieval["chunk_results"],
            "entity_results": retrieval["entity_results"],
            "context": context,
        }

    def ask(self, query: str, top_k: int = 3, mode: str = "hybrid") -> str:
        return self.answer(query=query, top_k=top_k, mode=mode).get("answer", "")
