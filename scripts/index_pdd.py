from __future__ import annotations
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.pdf_utils import extract_text_from_pdf
from app.rag_pipeline import RAGPipeline

def index_pdd():
    pipeline = RAGPipeline()
    books_dir = BASE_DIR / "pdd_docs"
    
    if not books_dir.exists():
        print(f"Folder '{books_dir}' does not exist. Creating it...")
        books_dir.mkdir()
        print(f"Please place your PDD PDF files in '{books_dir}/' and run again.")
        return
    
    pdf_files = sorted(books_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {books_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"\nIndexing: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            print("  Warning: No text extracted")
            continue
        
        doc_id = pdf_path.stem
        
        # Index in ChunkBased
        pipeline.chunk_engine.add_document(
            text=text,
            doc_id=doc_id,
            metadata={"source": pdf_path.name, "type": "pdd"}
        )
        
        # Index in EntityBased
        pipeline.entity_engine.add_document(
            text=text,
            doc_id=doc_id,
            metadata={"source": pdf_path.name, "type": "pdd"}
        )
        
        print(f"  Added {len(text):,} characters")
    
    print(f"\n✅ Done! Total chunks: {pipeline.chunk_engine.collection.count()}")

if __name__ == "__main__":
    index_pdd()