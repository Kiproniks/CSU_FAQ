from __future__ import annotations

from typing import Any

from flask import Flask, render_template, request

from app.config import settings
from app.rag_pipeline import RAGPipeline

# Веб-приложение и лениво создаваемый пайплайн, общий для всех запросов.
app = Flask(__name__)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    # Создаем пайплайн один раз, чтобы не переинициализировать индексы на каждый запрос.
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _snippet(text: str, limit: int = 260) -> str:
    # Короткий предпросмотр фрагмента в блоке «На чем основан ответ».
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    # Одностраничный сценарий: принять вопрос, запустить RAG в выбранном режиме, отрисовать результат.
    result: dict[str, Any] | None = None
    basis_hits: list[dict[str, Any]] = []
    basis_label = "ChunkBased"
    error = ""
    query = ""
    mode = "chunk"
    top_k = 3

    if request.method == "POST":
        # Чтение и валидация пользовательских параметров.
        query = (request.form.get("query") or "").strip()
        mode = (request.form.get("mode") or "chunk").strip().lower()
        if mode not in {"chunk", "entity"}:
            mode = "chunk"

        top_k_raw = request.form.get("top_k") or "3"
        try:
            top_k = max(1, int(top_k_raw))
        except ValueError:
            top_k = 3

        if not query:
            error = "Введите вопрос."
        else:
            try:
                # Основной вызов пайплайна поиска и генерации.
                result = get_pipeline().answer(query=query, top_k=top_k, mode=mode)
            except Exception as exc:
                error = f"Ошибка пайплайна: {exc}"

    if result:
        # Показ фрагментов, на которых основан ответ.
        basis_hits = result.get("hits", []) or []
        basis_label = "EntityBased (TF-IDF)" if mode == "entity" else "ChunkBased"

    return render_template(
        "index.html",
        query=query,
        mode=mode,
        top_k=top_k,
        result=result,
        basis_hits=basis_hits,
        basis_label=basis_label,
        error=error,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        snippet=_snippet,
    )

