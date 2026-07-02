# SHL AI Assessment Recommender

Production-ready FastAPI chatbot for recommending SHL Individual Test Solution
assessments from a catalog-grounded retrieval pipeline.

## Features

- Stateless `POST /chat` API using the full supplied conversation history
- Clarification before recommending when requirements are vague
- Recommendation, refinement, and comparison flows
- Catalog-only structured recommendations with SHL URLs
- Prompt injection, off-topic, legal, medical, finance, and general hiring refusals
- Offline local vector retrieval built from `data/catalog.json`
- Optional Gemini or OpenRouter LLM response generation when explicitly enabled
- Deterministic fallback when no LLM key is configured
- Pytest coverage for core API behavior
- Render deployment config

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in keys as needed.

```bash
copy .env.example .env
```

Important variables:

- `GEMINI_API_KEY`: enables Gemini responses
- `OPENROUTER_API_KEY`: enables OpenRouter when `LLM_PROVIDER=openrouter`
- `ENABLE_LLM`: set to `true` only when external LLM responses are desired
- `CATALOG_PATH`: path to the catalog JSON file
- `TOP_K`: number of retrieved assessments, capped at 10

## Run Locally

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## API

### Health

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

### Chat

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a Java developer with SQL and reasoning skills\"}]}"
```

Response schema:

```json
{
  "reply": "....",
  "recommendations": [
    {
      "name": "...",
      "url": "...",
      "test_type": "..."
    }
  ],
  "end_of_conversation": false
}
```

## Catalog Source

```bash
python scraper/scrape_catalog.py --output data/catalog.json
```

The committed `data/catalog.json` is the runtime source of truth. The app never
scrapes during startup or request handling. The optional scraper skips live SHL
requests when the output catalog already exists; pass `--force` only when you are
manually refreshing the catalog and have installed optional scraper dependencies.
Install `playwright`, `beautifulsoup4`, and `requests` only in a maintenance
environment that needs to refresh the catalog.

The loader supports both the legacy normalized catalog fields and the current SHL
scraper format with `link`, `keys`, `remote`, and `adaptive` fields.

## Build Vector Index

The local JSON vector index is built automatically from `data/catalog.json` when
`vector_index/index.json` is missing or stale. It does not require browser
binaries, SHL network access, FAISS, or downloaded embedding models.

## Tests

```bash
pytest
```

## Deployment on Render

1. Push the repository to GitHub.
2. Create a new Render web service from the repository.
3. Render reads `render.yaml`.
4. Keep `ENABLE_LLM=false` for fully offline catalog-only responses, or add
   `GEMINI_API_KEY`/`OPENROUTER_API_KEY` and set `ENABLE_LLM=true` for optional
   generated wording.

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Project Structure

```text
app/
  main.py
  api.py
  chat.py
  retriever.py
  prompts.py
  guardrails.py
  models.py
  utils.py
scraper/
  scrape_catalog.py
data/
  catalog.json
vector_index/
docs/
  approach.md
tests/
requirements.txt
README.md
.env.example
.gitignore
render.yaml
Procfile
```
