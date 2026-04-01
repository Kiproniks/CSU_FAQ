from __future__ import annotations

import inspect
from typing import Any, Dict, List, Literal

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.EntityBased import EntityBased
from app.config import settings
from app.db import get_database
from app.llm_service import LLMService

RetrievalMode = Literal["hybrid", "chunk", "entity"]


class RAGPipeline:
    def __init__(self) -> None:
        # Инициализируем оба ретривера и LLM-сервис один раз на пайплайн.
        self.chunk_engine = self._init_chunk_engine()
        self.entity_engine = EntityBased(
            min_entity_length=settings.entity_min_length,
            max_entities_per_chunk=settings.entity_max_entities_per_chunk,
            tfidf_weight=settings.entity_tfidf_weight,
            entity_overlap_weight=settings.entity_overlap_weight,
            min_score=settings.entity_min_score,
            mmr_lambda=settings.entity_mmr_lambda,
        )
        self.llm = LLMService()

        # Сначала поднимаем индексы из PostgreSQL (authoritative storage).
        loaded_from_db = self._bootstrap_from_database()

        # Fallback: если в БД пусто, можно поднять entity-индекс из Chroma.
        if settings.bootstrap_entity_from_chroma:
            if not loaded_from_db:
                self._bootstrap_entity_index_from_chroma()

    def _init_chunk_engine(self) -> ChunkBased:
        # Основной chunk-движок использует значения из .env/config.
        # Поддерживаем обе версии конструктора (с chroma_path и без него).
        init_params = inspect.signature(ChunkBased.__init__).parameters
        kwargs: Dict[str, Any] = {
            "chunk_size": settings.chunk_size,
            "overlap": settings.chunk_overlap,
            "splitter_mode": settings.chunk_splitter_mode,
            "mmr_lambda": settings.chunk_mmr_lambda,
            "collection_name": settings.chunk_collection,
            "embedding_model": settings.chunk_embedding_model,
        }
        if "chroma_path" in init_params:
            kwargs["chroma_path"] = settings.chroma_path

        primary = ChunkBased(**kwargs)

        try:
            primary_count = primary.collection.count()
        except Exception:
            return primary

        # Обратная совместимость: ранее путь и коллекция были захардкожены.
        if (
            primary_count == 0
            and (settings.chroma_path != "./chroma_db" or settings.chunk_collection != "harry_potter_collection")
        ):
            legacy_kwargs: Dict[str, Any] = {
                "chunk_size": settings.chunk_size,
                "overlap": settings.chunk_overlap,
                "splitter_mode": settings.chunk_splitter_mode,
                "mmr_lambda": settings.chunk_mmr_lambda,
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
        # Восстанавливаем TF-IDF состояние entity-индекса из сохраненных чанков Chroma пакетами.
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
                    # Новая сигнатура (если есть doc_id/metadata).
                    self.entity_engine.add_chunk(
                        chunk=text,
                        chunk_id=len(self.entity_engine.chunks),
                        doc_id=source,
                        metadata=metadata,
                    )
                except TypeError:
                    # Старый tro-part: add_chunk(chunk, chunk_id).
                    self.entity_engine.add_chunk(text, len(self.entity_engine.chunks))

        self.entity_engine.build_index()

    def _bootstrap_from_database(self) -> bool:
        """Восстанавливает chunk/entity из PostgreSQL. Возвращает True, если что-то загружено."""
        try:
            rows = get_database().load_chunks(limit=300000)
        except Exception:
            rows = []

        if not rows:
            return False

        chunk_rows = []
        for row in rows:
            chunk_id = str((row or {}).get("chunk_id", "")).strip()
            text = str((row or {}).get("text", "") or "")
            metadata = (row or {}).get("metadata", {}) or {}
            doc_id = str((row or {}).get("doc_id", "") or metadata.get("doc_id") or metadata.get("source") or "unknown")
            try:
                chunk_index = int((row or {}).get("chunk_index", metadata.get("chunk_index", 0)) or 0)
            except Exception:
                chunk_index = 0
            if not chunk_id or not text.strip():
                continue
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "metadata": metadata,
                }
            )

        if not chunk_rows:
            return False

        # Прогреваем chunk-движок из DB данных (и Chroma, если доступна).
        try:
            self.chunk_engine.chunks = []
            self.chunk_engine._chunk_by_id = {}
            self.chunk_engine._doc_chunk_index_map = {}
            ids = []
            documents = []
            metadatas = []
            for row in chunk_rows:
                chunk_id = row["chunk_id"]
                metadata = dict(row["metadata"] or {})
                metadata.setdefault("doc_id", row["doc_id"])
                metadata.setdefault("chunk_index", row["chunk_index"])
                ids.append(chunk_id)
                documents.append(row["text"])
                metadatas.append(metadata)
                payload = {"id": chunk_id, "text": row["text"], "metadata": metadata}
                self.chunk_engine.chunks.append(payload)
                self.chunk_engine._chunk_by_id[str(chunk_id)] = payload
                self.chunk_engine._doc_chunk_index_map[(str(metadata.get("doc_id")), int(metadata.get("chunk_index", 0)))] = str(chunk_id)

            try:
                batch = max(1, int(getattr(self.chunk_engine, "upsert_batch_size", 1000)))
                for start in range(0, len(ids), batch):
                    end = start + batch
                    self.chunk_engine.collection.upsert(
                        ids=ids[start:end],
                        documents=documents[start:end],
                        metadatas=metadatas[start:end],
                    )
            except Exception:
                pass

            self.chunk_engine._mark_sparse_dirty()
        except Exception:
            return False

        # Пересобираем entity-индекс из chunk-текста.
        try:
            self.entity_engine.chunks.clear()
            self.entity_engine.chunk_entities.clear()
            self.entity_engine.chunk_tokens.clear()
            self.entity_engine.entity_to_chunks.clear()
            self.entity_engine.tfidf_matrix = None
            for row in chunk_rows:
                self.entity_engine.add_chunk(
                    chunk=row["text"],
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    metadata=row["metadata"],
                )
            self.entity_engine.build_index()
        except Exception:
            return False

        return True

    def index_document(self, text: str, doc_id: str, metadata: Dict[str, Any] | None = None) -> None:
        # Держим оба индекса синхронизированными при добавлении нового документа.
        safe_metadata = metadata or {}
        self.chunk_engine.add_document(text, doc_id=doc_id, metadata=safe_metadata)

        # Для tro-part версии EntityBased add_document может отсутствовать.
        add_document = getattr(self.entity_engine, "add_document", None)
        if callable(add_document):
            add_document(
                text,
                doc_id=doc_id,
                metadata=safe_metadata,
                chunk_size=settings.entity_chunk_size,
                overlap=settings.entity_chunk_overlap,
            )
            return

        split_text = getattr(self.entity_engine, "_split_text", None)
        if callable(split_text):
            chunks = split_text(
                text,
                chunk_size=settings.entity_chunk_size,
                overlap=settings.entity_chunk_overlap,
            )
        else:
            chunks = ChunkBased._split_text(text, settings.entity_chunk_size, settings.entity_chunk_overlap)

        for chunk in chunks:
            self.entity_engine.add_chunk(chunk, len(self.entity_engine.chunks))

    @staticmethod
    def _normalize_chunk_results(raw_results: List[Any]) -> List[Dict[str, Any]]:
        # Нормализуем вывод ChunkBased к общей схеме ответа.
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
        # Нормализуем вывод EntityBased к той же схеме, что и chunk-результаты.
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
        # Формируем текстовый блок контекста, который передается в LLM.
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
        # Запускаем оба ретривера и возвращаем раздельные и объединенные результаты.
        chunk_k = top_k_chunks or settings.top_k_chunks
        entity_k = top_k_entities or settings.top_k_entities

        try:
            raw_chunk_results = self.chunk_engine.search(query, top_k=chunk_k)
        except Exception as exc:
            print(f"[RAGPipeline] chunk retrieval error: {exc}")
            raw_chunk_results = []

        try:
            raw_entity_results = self.entity_engine.search(query, top_k=entity_k)
        except Exception as exc:
            print(f"[RAGPipeline] entity retrieval error: {exc}")
            raw_entity_results = []

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
        # Поддерживаем удобные алиасы режимов из UI и Telegram.
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
        # Полный поток: retrieve -> сборка контекста -> запрос к LLM.
        normalized_mode = self._normalize_mode(mode)
        chunk_k = top_k or settings.top_k_chunks
        entity_k = top_k or settings.top_k_entities

        retrieval: Dict[str, List[Dict[str, Any]]]
        if normalized_mode == "chunk":
            # Быстрый путь: считаем только chunk retrieval.
            try:
                raw_chunk_results = self.chunk_engine.search(query, top_k=chunk_k)
            except Exception as exc:
                print(f"[RAGPipeline] chunk retrieval error: {exc}")
                raw_chunk_results = []
            chunk_results = self._normalize_chunk_results(raw_chunk_results)
            retrieval = {
                "chunk_results": chunk_results,
                "entity_results": [],
                "combined": list(chunk_results),
            }
        elif normalized_mode == "entity":
            # Быстрый путь: считаем только entity retrieval.
            try:
                raw_entity_results = self.entity_engine.search(query, top_k=entity_k)
            except Exception as exc:
                print(f"[RAGPipeline] entity retrieval error: {exc}")
                raw_entity_results = []
            entity_results = self._normalize_entity_results(raw_entity_results)
            retrieval = {
                "chunk_results": [],
                "entity_results": entity_results,
                "combined": list(entity_results),
            }
        else:
            retrieval = self.retrieve(
                query=query,
                top_k_chunks=top_k,
                top_k_entities=top_k,
            )

        selected_chunk_results = retrieval["chunk_results"]
        selected_entity_results = retrieval["entity_results"]
        selected_hits = retrieval["combined"]

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
        # Упрощенный обертка-метод, когда нужен только финальный текст ответа.
        return self.answer(query=query, top_k=top_k, mode=mode).get("answer", "")


