from __future__ import annotations
import inspect
from typing import Any, Dict, List, Literal

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.EntityBased import EntityBased
from app.config import settings
from app.llm_service import LLMService
from app.query_decomposer import QueryDecomposer   # ← обязательно должен существовать

RetrievalMode = Literal["hybrid", "chunk", "entity"]


class RAGPipeline:
    def __init__(self) -> None:
        self.chunk_engine = self._init_chunk_engine()
        self.entity_engine = EntityBased()
        self.llm = LLMService()
        self.decomposer = QueryDecomposer()                     # ← добавлено для разбиения сложных вопросов

        if settings.bootstrap_entity_from_chroma:
            self._bootstrap_entity_index_from_chroma()

    # ====================== ВСЁ ТВОЁ СТАРОЕ (полностью вставлено) ======================
    def _init_chunk_engine(self) -> ChunkBased:
        init_params = inspect.signature(ChunkBased.__init__).parameters
        kwargs: Dict[str, Any] = {
            "chunk_size": settings.chunk_size,
            "overlap": settings.chunk_overlap,
            "collection_name": settings.chunk_collection,
        }
        if "chroma_path" in init_params:
            kwargs["chroma_path"] = settings.chroma_path

        primary = ChunkBased(**kwargs)

        try:
            primary_count = primary.collection.count()
        except Exception:
            return primary

        if (
            primary_count == 0
            and (settings.chroma_path != "./chroma_db" or settings.chunk_collection != "harry_potter_collection")
        ):
            legacy_kwargs: Dict[str, Any] = {
                "chunk_size": settings.chunk_size,
                "overlap": settings.chunk_overlap,
                "collection_name": "harry_potter_collection",
            }
            if "chroma_path" in init_params:
                legacy_kwargs["chroma_path"] = "./chroma_db"
            legacy = ChunkBased(**legacy_kwargs)
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
                try:
                    self.entity_engine.add_chunk(
                        chunk=text,
                        chunk_id=len(self.entity_engine.chunks),
                        doc_id=source,
                        metadata=metadata,
                    )
                except TypeError:
                    self.entity_engine.add_chunk(text, len(self.entity_engine.chunks))
        self.entity_engine.build_index()

    def index_document(self, text: str, doc_id: str, metadata: Dict[str, Any] | None = None) -> None:
        safe_metadata = metadata or {}
        self.chunk_engine.add_document(text, doc_id=doc_id, metadata=safe_metadata)

        add_document = getattr(self.entity_engine, "add_document", None)
        if callable(add_document):
            add_document(
                text,
                doc_id=doc_id,
                metadata=safe_metadata,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            return

        split_text = getattr(self.entity_engine, "_split_text", None)
        if callable(split_text):
            chunks = split_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        else:
            chunks = ChunkBased._split_text(text, settings.chunk_size, settings.chunk_overlap)

        for chunk in chunks:
            self.entity_engine.add_chunk(chunk, len(self.entity_engine.chunks))

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
    def _normalize_entity_results(raw_results: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in raw_results:
            if isinstance(item, tuple) and len(item) == 2:
                payload, score = item
                payload = payload or {}
                metadata = payload.get("metadata", {})
                entities = payload.get("entities", payload.get("entity", []))
                text = payload.get("text", "")
                source = (metadata or {}).get("source", payload.get("doc_id", "unknown"))
            else:
                payload = item if isinstance(item, dict) else {}
                metadata = payload.get("metadata", {})
                entities = payload.get("entities", [])
                text = payload.get("text", "")
                score = payload.get("score", 0.0)
                source = (metadata or {}).get("source", payload.get("doc_id", "unknown"))

            normalized.append(
                {
                    "strategy": "entity-based-tfidf",
                    "text": text,
                    "score": float(score),
                    "metadata": metadata or {},
                    "source": source,
                    "entities": entities if isinstance(entities, list) else [],
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

    # ====================== НОВОЕ: разбиение сложных вопросов ======================
    def _single_answer(self, query: str, mode: str = "hybrid", top_k: int | None = None) -> Dict[str, Any]:
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

        context = self._build_context(selected_chunk_results, selected_entity_results)
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

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        # ВСЕГДА разбиваем на под-вопросы (Chunk, Entity и Hybrid)
        sub_queries = self.decomposer.decompose(query)

        if len(sub_queries) == 1:
            return self._single_answer(sub_queries[0], mode, top_k)

        results = []
        for sq in sub_queries:
            sub_result = self._single_answer(sq, mode, top_k)
            results.append(sub_result)
        return results

    def ask(self, query: str, top_k: int = 3, mode: str = "hybrid") -> str:
        result = self.answer(query=query, top_k=top_k, mode=mode)
        if isinstance(result, list):
            # Если вопрос сложный → несколько ответов
            return "\n\n".join([r.get("answer", "") for r in result])
        return result.get("answer", "")