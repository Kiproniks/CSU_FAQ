# CSU FAQ / Harry Potter RAG

University demo project for intelligent search over PDF books with two retrieval strategies:

1. `chunk-based` (vector search in Chroma)
2. `entity-based` (currently TF-IDF keyword/entity-like retrieval, not a full entity graph)

## Project Structure

```text
CSU_FAQ_clean/
  app/
    config.py
    llm_service.py
    pdf_utils.py
    rag_pipeline.py
    telegram_bot.py
  ChunkBased/
    ChunkBased.py
  EntityBased/
    Entity_Based.py
  scripts/
    reindex_harry_potter.py
    demo_query.py
    visualize_splitting.py
  harry_potter/
    .gitkeep               # put your PDFs here locally
  chroma_db/               # current persistent vector index
  run_bot.py
  requirements.txt
  .env.example
```

## Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create `.env` from template:

```powershell
Copy-Item .env.example .env
```

Key variables:

- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `LLM_PROVIDER` - `echo`, `ollama`, or `openai`
- `LLM_MODEL` - model name (`llama3.1:8b` for Ollama, or OpenAI model)
- `OLLAMA_BASE_URL` - usually `http://localhost:11434`
- `OPENAI_API_KEY` - required for `openai`
- `CHROMA_PATH` - Chroma folder (default `./chroma_db`)
- `CHUNK_COLLECTION` - collection name (default `harry_potter_collection`)

## Reindex Harry Potter Books

Place source PDFs into `harry_potter/` before running reindexing (PDFs are not committed to git).

```powershell
python scripts/reindex_harry_potter.py
```

Optional:

```powershell
python scripts/reindex_harry_potter.py --clear-chunk-index --clear-entity-index
```

## Test Search and Pipeline (Demo for Teacher)

Single reproducible script:

```powershell
python scripts/demo_query.py "Who is Dumbledore?" --top-k 3
```

With chunk relevance visualization:

```powershell
python scripts/demo_query.py "Who is Dumbledore?" --top-k 3 --plot-chunks
```

Demo output includes:

- chunk-based hits with scores and sources
- entity-based hits with scores and extracted entities
- final pipeline answer

## Split Visualization

Create an output folder with chunk splits for both strategies:

```powershell
python scripts/visualize_splitting.py --limit 0
```

Artifacts are saved to `data/split_visualization/`:

- `chunk_based/*.txt|*.json` - how `ChunkBased` split each book
- `entity_based/*.txt|*.json` - how `EntityBased` split each book
- `plots/*_lengths.png` - chunk length comparison charts
- `SUMMARY.md` - per-book split statistics

Note: `data/split_visualization/` is generated locally and ignored in git.

## Local LLM via Ollama

1. Install and run Ollama
2. Pull model (example):

```powershell
ollama pull llama3.1:8b
```

3. In `.env` set:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

If Ollama is unavailable, system automatically falls back to `echo` mode (no crash).

## Telegram Bot

1. Set token in `.env`:

```env
TELEGRAM_BOT_TOKEN=your_token_here
```

2. Run:

```powershell
python run_bot.py
```

Bot flow:

- receives user message
- `/start` shows buttons with two modes: `ChunkBased` and `EntityBased`
- stores selected mode per chat
- runs RAG pipeline in selected mode
- returns answer + top sources
- if LLM is unavailable, returns short best-matching fragments from indexed files
- handles runtime errors gracefully

## Notes About Current Architecture

- `chunk-based` is the stronger retrieval strategy for this project state.
- `entity-based` is currently TF-IDF/keyword-style and is shown honestly as such.
- Core indexing pipeline is preserved; changes are incremental and compatibility-focused.

## Cleanup Notes

Safe-to-remove or ignore artifacts (not used by runtime pipeline):

- `venv/`
- `__pycache__/` folders
- `#U041f#U0440#U043e#U0447#U0442#U0438#U041c#U0435#U043d#U044f.txt` (local note file)
- `data/chroma_db/` and `ChunkBased/chroma_db/` (legacy/local test artifacts)

Current working index path is `chroma_db/` with collection `harry_potter_collection`.
