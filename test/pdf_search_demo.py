"""Compare both retrieval methods on a synthetic PDF."""

from pathlib import Path
import sys

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ChunkBased.ChunkBased import ChunkBased
from EntityBased.EntityBased import EntityBased
from vizualize_scripts.visualize_chunks import visualize_both


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    return "\n\n".join(text for page in reader.pages if (text := page.extract_text())).strip()


def main() -> None:
    pdf_path = Path(__file__).with_name("fixtures") / "faq-sample.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunk_size, overlap = 1000, 200

    entity_system = EntityBased()
    for start in range(0, len(text), chunk_size - overlap):
        chunk = text[start:start + chunk_size]
        if len(chunk) > 100:
            entity_system.add_chunk(chunk, start)
    entity_system.build_index()

    chunk_system = ChunkBased(
        chunk_size=chunk_size, overlap=overlap, collection_name="pdf_search_demo"
    )
    chunk_system.add_document(text, doc_id="faq_sample")

    query = "How do I renew a library loan?"
    print(f"Query: {query}")
    for name, system in (("EntityBased", entity_system), ("ChunkBased", chunk_system)):
        print(f"\n{name}:")
        for rank, (chunk, score) in enumerate(system.search(query, top_k=3), start=1):
            excerpt = " ".join(chunk["text"].split())[:200]
            print(f"{rank}. {score:.3f} | {excerpt}")

    visualize_both(text)


if __name__ == "__main__":
    main()
