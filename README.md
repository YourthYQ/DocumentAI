# DocuAI

**RAG-based document Q&A API** — ingest PDFs or text, ask questions in natural language, get answers grounded in your documents with source citations.

DocuAI turns your own documents (PDFs, plain text) into a searchable knowledge base and answers questions in natural language, with each claim tied to specific source chunks. It uses **Retrieval Augmented Generation (RAG)**: a vector store (Pinecone) for semantic search over document chunks, an LLM (OpenAI) to generate answers from the retrieved context, and structured storage (MongoDB, Redis) for documents, chat logs, and session history. The backend is a REST API (FastAPI) with OpenAPI docs, health checks, and Docker Compose for local and deployment use. It’s built to be extensible—modular retrieval, LLM, and storage layers—and to keep answers grounded in your data instead of generic model knowledge.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

---

## Features

| Feature | Description |
|--------|-------------|
| **Document ingestion** | Raw text `POST` or file upload (PDF, TXT). Chunking with configurable overlap. |
| **Vector search** | Embeddings via OpenAI, stored and queried in Pinecone. |
| **RAG chat** | Retrieve top‑k chunks → LLM (GPT‑4 or 3.5) → answer + citations. |
| **Session & audit** | Per-session history in Redis (TTL). All turns logged to MongoDB. |
| **APIs** | REST with OpenAPI at `/docs`. Health checks for MongoDB, Redis, OpenAI, Pinecone. |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic |
| LLM & embeddings | LangChain, OpenAI (Chat + `text-embedding-ada-002`) |
| Vector DB | Pinecone (LangChain `PineconeVectorStore`) |
| Document DB | MongoDB (documents, chat logs) |
| Session store | Redis (`RedisChatMessageHistory`) |
| Doc processing | pdfminer.six, `RecursiveCharacterTextSplitter` |
| Runtime | Docker, Docker Compose |

---

## Architecture

```
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   /api/v1/...   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐   ┌───────────────┐   ┌─────────────────┐
  │  MongoDB    │   │  Pinecone     │   │  Redis          │
  │  docs, logs │   │  embeddings   │   │  chat sessions  │
  └─────────────┘   └───────┬───────┘   └─────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  OpenAI       │
                    │  embed + LLM  │
                    └───────────────┘
```

**RAG flow:** `user_message` → vector similarity (Pinecone, k=5) → `format_docs` → prompt (context + question) → `ChatOpenAI` → `StrOutputParser` → `ai_response` + `retrieved_docs`.

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, [OpenAI](https://platform.openai.com) and [Pinecone](https://app.pinecone.io) API keys.

```bash
git clone https://github.com/yourth/DocumentAI.git
cd DocumentAI/docuai

cp env.example .env
# Edit .env: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT.
# Optional: PINECONE_INDEX_NAME (default: docuai-index).
# MongoDB/Redis defaults work with docker-compose.

docker-compose up --build -d
```

- **API:** http://localhost:8000  
- **Swagger:** http://localhost:8000/docs  
- **Health:** http://localhost:8000/api/v1/health  

---

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/` | Add document from JSON `{ "content", "metadata?" }`. |
| `POST` | `/api/v1/documents/upload_file/` | Upload PDF or TXT; chunk, embed, and index. |
| `GET` | `/api/v1/documents/{doc_id}` | Fetch document by id (MongoDB; for docs added via `POST /documents/`). |
| `POST` | `/api/v1/chat/` | `{ "session_id", "user_message" }` → RAG reply + `retrieved_docs`. |
| `GET` | `/api/v1/sessions/` | List sessions (last interaction, message count). |
| `GET` | `/api/v1/chat/{session_id}/history` | Chat history for a session (from MongoDB). |
| `GET` | `/api/v1/health` | Status of MongoDB, Redis, OpenAI, Pinecone. |

Full request/response schemas: **http://localhost:8000/docs**.

---

## Project Structure

```
DocumentAI/
├── docuai/                 # App and Docker
│   ├── app/
│   │   ├── api/            # FastAPI routes, Pydantic models
│   │   ├── core/           # Config (env)
│   │   ├── data/           # MongoDB: documents, chat_logs
│   │   ├── llm/            # RAG chain (retriever → prompt → LLM → parser)
│   │   ├── retrieval/      # Pinecone + LangChain embeddings, index helpers
│   │   └── services/       # document_processor (PDF/TXT, chunking), conversation_manager (Redis)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── env.example
│   └── requirements.txt
├── tests/                  # API, config, storage, RAG, retrieval, conversation_manager
├── .github/workflows/      # CI (pytest on push/PR)
└── README.md
```

---

## Development

**Run tests** (from repo root):

```bash
pip install -r docuai/requirements.txt
PYTHONPATH=docuai pytest tests/ -v
```

**Env (`.env` in `docuai/`):**

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key. |
| `PINECONE_API_KEY` | ✅ | Pinecone API key. |
| `PINECONE_ENVIRONMENT` | ✅ | Pinecone env/region (e.g. `gcp-starter`, `us-west1-gcp`). |
| `PINECONE_INDEX_NAME` | | Index name; default `docuai-index`. Created on first run if missing. |
| `MONGO_URI`, `MONGO_DB_NAME` | | Defaults work with `docker-compose` (`mongo`, `docuai_db`). |
| `REDIS_HOST`, `REDIS_PORT` | | Defaults work with `docker-compose` (`redis`, `6379`). |
| `DEFAULT_LLM_MODEL` | | e.g. `gpt-4` or `gpt-3.5-turbo`. |

---

## Design Choices

- **RAG only from provided context:** If retrieval returns nothing, the model is instructed to say the answer is not in the documents.
- **Sessions:** Redis holds in-session turns (TTL); MongoDB holds all turns for audit and `GET /sessions` and `GET /chat/{id}/history`. Conversation history is not yet fed back into the RAG context (single-turn RAG per request).
- **File uploads:** PDF/TXT are chunked and embedded per chunk; the assembled file is not stored in MongoDB. `GET /documents/{id}` applies to docs created via `POST /documents/` (full text in MongoDB).
- **Index creation:** On startup, if `PINECONE_INDEX_NAME` does not exist, the app attempts to create a serverless index (dim=1536, cosine).

---

## License

Licensed under the [MIT License](LICENSE).