# Approach

This project implements a stateless conversational recommender for SHL Individual
Test Solution assessments.

## Architecture

Requests enter FastAPI through `POST /chat`. The API validates the stable schema,
passes the full message history to `ChatService`, and returns a fixed response
shape containing `reply`, `recommendations`, and `end_of_conversation`.

The chat pipeline is:

1. Guardrail check for prompt injection, off-topic requests, and restricted advice.
2. Intent detection for clarification, recommendation, refinement, or comparison.
3. Query construction from the complete stateless conversation history.
4. Retrieval from the SHL catalog using a persisted local vector index.
5. Optional Gemini/OpenRouter response generation grounded in retrieved context.
6. Deterministic fallback response using only validated catalog records.

## Retrieval

Catalog records are normalized into `Assessment` objects. The loader accepts both
the legacy normalized schema and the current SHL scraper schema, mapping `link`
to `url`, `keys` to test type/searchable skills, and `remote`/`adaptive` to
booleans. Each assessment exposes a `search_text()` representation combining
name, description, measured skills, test type, duration, support flags, job
levels, and languages.

`CatalogRetriever` builds a deterministic sparse vector index directly from
`data/catalog.json`. The index is stored as JSON under `vector_index/` and is
rebuilt automatically when missing or stale. No browser, SHL network access,
FAISS package, or downloaded embedding model is required at runtime. If the index
cannot be loaded, the retriever falls back to lexical scoring from the same local
catalog.

## Grounding and Guardrails

The chatbot never recommends outside `data/catalog.json`. Structured
recommendations are validated by exact catalog name and URL before being returned.
The LLM receives a strict system prompt plus retrieved catalog context; if the LLM
is unavailable, the deterministic fallback still returns catalog-only results.

The service refuses prompt injection, roleplay attacks, general hiring advice,
legal, medical, and financial requests.

## Scraping

`scraper/scrape_catalog.py` is a manual maintenance command only. The FastAPI app
never calls it during startup or request handling. When `data/catalog.json`
already exists, the scraper exits without making live SHL requests unless
`--force` is supplied.
