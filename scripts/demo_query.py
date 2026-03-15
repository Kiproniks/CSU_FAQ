from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag_pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo for chunk/entity retrieval and final RAG answer")
    parser.add_argument(
        "query",
        nargs="?",
        default="Who is Severus Snape?",
        help="Question for retrieval and answer",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k per retrieval strategy",
    )
    parser.add_argument(
        "--plot-chunks",
        action="store_true",
        help="Generate chunk relevance visualization PNG",
    )
    parser.add_argument(
        "--plot-path",
        default=str(BASE_DIR / "data" / "chunk_search_demo.png"),
        help="Path to save chunk plot",
    )
    return parser.parse_args()


def _snippet(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def print_chunk_results(results: list[dict]) -> None:
    print("\n=== CHUNK-BASED RESULTS ===")
    if not results:
        print("No results")
        return

    for index, hit in enumerate(results, start=1):
        print(f"{index}. score={hit['score']:.4f} | source={hit['source']}")
        print(f"   {_snippet(hit['text'])}")


def print_entity_results(results: list[dict]) -> None:
    print("\n=== ENTITY-BASED RESULTS (TF-IDF, not entity graph) ===")
    if not results:
        print("No results")
        return

    for index, hit in enumerate(results, start=1):
        entities = ", ".join(hit.get("entities", [])[:10]) or "-"
        print(f"{index}. score={hit['score']:.4f} | source={hit['source']}")
        print(f"   entities: {entities}")
        print(f"   {_snippet(hit['text'])}")


def main() -> None:
    args = parse_args()
    pipeline = RAGPipeline()

    chunk_count = pipeline.chunk_engine.collection.count()
    if chunk_count == 0:
        print("Chunk index is empty. Run: python scripts/reindex_harry_potter.py")
        sys.exit(1)

    result = pipeline.answer(args.query, top_k=args.top_k)

    print(f"Question: {args.query}")
    print(f"LLM provider/model: {result['provider']} / {result['model']}")
    print(f"Indexed chunks in Chroma: {chunk_count}")

    print_chunk_results(result["chunk_results"])
    print_entity_results(result["entity_results"])

    if args.plot_chunks:
        try:
            plot_path = Path(args.plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline.chunk_engine.visualize_search(args.query, top_k=args.top_k, save_path=str(plot_path))
            print(f"\nChunk plot saved to: {plot_path}")
        except Exception as exc:
            print(f"\nChunk plot skipped: {exc}")

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
