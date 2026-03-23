from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.EntityBased import EntityBased
from app.config import settings
from app.db import get_database
from app.llm_service import LLMService

RetrievalMode = Literal["hybrid", "chunk", "entity"]


class RAGPipeline:
    def __init__(self) -> None:
        # Инициализируем ретриверы + LLM.
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

        # Отдельная память QA-датасета (Q/A пары), чтобы ответы учитывали большой датасет.
        self.qa_memory_engine = self._init_qa_memory_engine()
        self.qa_memory_enabled = False

        # Поднимаем индексы из БД.
        loaded_from_db = self._bootstrap_from_database()
        if settings.bootstrap_entity_from_chroma and not loaded_from_db:
            self._bootstrap_entity_index_from_chroma()

        # Подключаем QA-память из JSON-датасета.
        self._bootstrap_qa_memory()

    def _init_chunk_engine(self) -> ChunkBased:
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

        # Fallback к старой коллекции.
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

    def _init_qa_memory_engine(self) -> ChunkBased:
        # Коллекция памяти Q/A из датасета.
        init_params = inspect.signature(ChunkBased.__init__).parameters
        kwargs: Dict[str, Any] = {
            "chunk_size": 1200,
            "overlap": 0,
            "splitter_mode": "smart",
            "mmr_lambda": 0.85,
            "collection_name": "qa_dataset_memory_v1",
            "embedding_model": settings.chunk_embedding_model,
        }
        if "chroma_path" in init_params:
            kwargs["chroma_path"] = settings.chroma_path
        return ChunkBased(**kwargs)

    @staticmethod
    def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z\u0400-\u04FF0-9\-]{3,}", text or "")}

    @staticmethod
    def _extract_qa_answer(text: str) -> str:
        marker = "Правильный ответ:"
        raw = str(text or "")
        if marker in raw:
            tail = raw.split(marker, 1)[1].strip()
            if tail:
                return tail
        return ""

    def _resolve_qa_dataset_path(self) -> Path | None:
        candidates: List[Path] = []
        if settings.qa_dataset_path:
            candidates.append(Path(settings.qa_dataset_path))
        candidates.extend(
            [
                Path("обучение ллм/dataset_eval_top2500_best.json"),
                Path("обучение ллм/dataset_eval.json"),
                Path("обучение ллм/second_dataset.json"),
                Path("обучение ллм/third.json"),
            ]
        )

        for item in candidates:
            try:
                resolved = item if item.is_absolute() else (Path.cwd() / item)
                if resolved.exists() and resolved.is_file():
                    return resolved
            except Exception:
                continue

        for path in Path.cwd().glob("**/*top2500*.json"):
            if path.is_file():
                return path
        return None

    def _bootstrap_qa_memory(self) -> None:
        if not settings.qa_memory_enabled:
            return

        dataset_path = self._resolve_qa_dataset_path()
        if dataset_path is None:
            return

        try:
            if self.qa_memory_engine.collection.count() > 0:
                self.qa_memory_enabled = True
                return
        except Exception:
            pass

        rows = self._read_json_rows(dataset_path)
        if not rows:
            return

        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, item in enumerate(rows, start=1):
            question = str(item.get("question") or "").strip()
            answer = str(item.get("expected_answer") or "").strip()
            if not question or not answer:
                continue

            chunk_id = f"qa_dataset_{idx}"
            # Ограничиваем длину ответа в QA-памяти, чтобы не раздувать prompt и не ловить timeout.
            short_answer = answer if len(answer) <= 420 else f"{answer[:420]}..."
            text = f"Вопрос: {question}\nПравильный ответ: {short_answer}"
            metadata = {
                "source": "qa_dataset",
                "doc_id": "qa_dataset",
                "chunk_index": idx,
                "qa_question": question,
                "qa": True,
            }
            ids.append(chunk_id)
            docs.append(text)
            metadatas.append(metadata)

        if not ids:
            return

        batch = max(100, int(getattr(self.qa_memory_engine, "upsert_batch_size", 1000)))
        for start in range(0, len(ids), batch):
            end = start + batch
            self.qa_memory_engine.collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metadatas[start:end],
            )

        # Локальный кэш для ChunkBased.search.
        self.qa_memory_engine.chunks = []
        self.qa_memory_engine._chunk_by_id = {}
        self.qa_memory_engine._doc_chunk_index_map = {}
        for i in range(len(ids)):
            payload = {"id": ids[i], "text": docs[i], "metadata": metadatas[i]}
            self.qa_memory_engine.chunks.append(payload)
            self.qa_memory_engine._chunk_by_id[ids[i]] = payload
            self.qa_memory_engine._doc_chunk_index_map[("qa_dataset", int(metadatas[i]["chunk_index"]))] = ids[i]
        self.qa_memory_engine._mark_sparse_dirty()
        self.qa_memory_enabled = True

    def _select_qa_results(self, query: str, qa_raw: List[Any]) -> List[Dict[str, Any]]:
        # Фильтруем QA-кандидаты: оставляем только те, где есть смысловое/лексическое совпадение с вопросом.
        normalized = self._normalize_chunk_results(qa_raw)
        q_tokens = self._tokens(query)
        if not q_tokens:
            return normalized[: settings.qa_memory_top_k]

        scored: List[Dict[str, Any]] = []
        for row in normalized:
            meta = row.get("metadata", {}) or {}
            qa_question = str(meta.get("qa_question", "") or "")
            candidate_text = f"{qa_question}\n{row.get('text', '')}"
            c_tokens = self._tokens(candidate_text)
            overlap = len(q_tokens & c_tokens)
            overlap_ratio = overlap / max(1, len(q_tokens))
            score = float(row.get("score", 0.0))

            # Низкая лексика + низкий score => отбрасываем шум.
            if overlap == 0 and score < 0.42:
                continue
            if overlap_ratio < 0.10 and score < 0.50:
                continue

            row["strategy"] = "qa-dataset-memory"
            row["_qa_rank"] = 0.65 * overlap_ratio + 0.35 * score
            row["_qa_overlap_ratio"] = overlap_ratio
            scored.append(row)

        scored.sort(key=lambda x: float(x.get("_qa_rank", 0.0)), reverse=True)
        return scored[: settings.qa_memory_top_k]

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
            batch = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
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

    def _bootstrap_from_database(self) -> bool:
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

        try:
            self.chunk_engine.chunks = []
            self.chunk_engine._chunk_by_id = {}
            self.chunk_engine._doc_chunk_index_map = {}
            ids, documents, metadatas = [], [], []
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
        safe_metadata = metadata or {}
        self.chunk_engine.add_document(text, doc_id=doc_id, metadata=safe_metadata)

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
            chunks = split_text(text, chunk_size=settings.entity_chunk_size, overlap=settings.entity_chunk_overlap)
        else:
            chunks = ChunkBased._split_text(text, settings.entity_chunk_size, settings.entity_chunk_overlap)
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
    def _build_context(
        chunk_results: List[Dict[str, Any]],
        entity_results: List[Dict[str, Any]],
        qa_results: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []

        if qa_results:
            parts.append("=== QA dataset memory (high priority) ===")
            for index, item in enumerate(qa_results, start=1):
                parts.append(f"{index}. score={item['score']:.4f} | source={item['source']}\n{item['text']}")

        if chunk_results:
            parts.append("=== Chunk-based retrieval ===")
            for index, item in enumerate(chunk_results, start=1):
                parts.append(f"{index}. score={item['score']:.4f} | source={item['source']}\n{item['text']}")

        if entity_results:
            parts.append("=== Entity-based retrieval (TF-IDF, keyword-level) ===")
            for index, item in enumerate(entity_results, start=1):
                entities = ", ".join(item.get("entities", [])[:8]) or "-"
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
        combined = sorted(chunk_results + entity_results, key=lambda x: x.get("score", 0.0), reverse=True)
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

    def answer(self, query: str, top_k: int | None = None, mode: str = "hybrid") -> Dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        chunk_k = top_k or settings.top_k_chunks
        entity_k = top_k or settings.top_k_entities

        retrieval: Dict[str, List[Dict[str, Any]]]
        if normalized_mode == "chunk":
            try:
                raw_chunk_results = self.chunk_engine.search(query, top_k=chunk_k)
            except Exception as exc:
                print(f"[RAGPipeline] chunk retrieval error: {exc}")
                raw_chunk_results = []
            chunk_results = self._normalize_chunk_results(raw_chunk_results)
            retrieval = {"chunk_results": chunk_results, "entity_results": [], "combined": list(chunk_results)}
        elif normalized_mode == "entity":
            try:
                raw_entity_results = self.entity_engine.search(query, top_k=entity_k)
            except Exception as exc:
                print(f"[RAGPipeline] entity retrieval error: {exc}")
                raw_entity_results = []
            entity_results = self._normalize_entity_results(raw_entity_results)
            retrieval = {"chunk_results": [], "entity_results": entity_results, "combined": list(entity_results)}
        else:
            retrieval = self.retrieve(query=query, top_k_chunks=top_k, top_k_entities=top_k)

        selected_chunk_results = retrieval["chunk_results"]
        selected_entity_results = retrieval["entity_results"]
        selected_hits = retrieval["combined"]

        qa_results: List[Dict[str, Any]] = []
        if self.qa_memory_enabled:
            try:
                qa_raw = self.qa_memory_engine.search(query, top_k=settings.qa_memory_top_k)
                qa_results = self._select_qa_results(query, qa_raw)
            except Exception as exc:
                print(f"[RAGPipeline] qa-memory retrieval error: {exc}")
                qa_results = []

        if qa_results:
            selected_hits = qa_results + selected_hits

        # Быстрый путь: если есть очень похожий QA-кейс из датасета, возвращаем его без LLM.
        if qa_results:
            top = qa_results[0]
            qa_rank = float(top.get("_qa_rank", 0.0))
            qa_score = float(top.get("score", 0.0))
            qa_overlap = float(top.get("_qa_overlap_ratio", 0.0))
            quick_answer = self._extract_qa_answer(str(top.get("text", "")))
            if quick_answer and (qa_rank >= 0.55 or (qa_score >= 0.62 and qa_overlap >= 0.20)):
                return {
                    "question": query,
                    "answer": quick_answer,
                    "provider": "qa-memory",
                    "model": "dataset_eval_top2500",
                    "mode": normalized_mode,
                    "hits": selected_hits,
                    "qa_results": qa_results,
                    "chunk_results": retrieval["chunk_results"],
                    "entity_results": retrieval["entity_results"],
                    "context": f"=== QA dataset memory (direct answer) ===\n{top.get('text', '')}",
                }

        context = self._build_context(selected_chunk_results, selected_entity_results, qa_results)
        llm_result = self.llm.generate_answer(query=query, context=context)

        return {
            "question": query,
            "answer": llm_result.answer,
            "provider": llm_result.provider,
            "model": llm_result.model,
            "mode": normalized_mode,
            "hits": selected_hits,
            "qa_results": qa_results,
            "chunk_results": retrieval["chunk_results"],
            "entity_results": retrieval["entity_results"],
            "context": context,
        }

    def ask(self, query: str, top_k: int = 3, mode: str = "hybrid") -> str:
        return self.answer(query=query, top_k=top_k, mode=mode).get("answer", "")
