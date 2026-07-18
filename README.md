# Research Paper RAG Assistant

A production-grade, agentic academic research assistant built using LangChain, LangGraph, FastAPI, and Streamlit. The assistant parses research paper PDFs, extracts academic metadata, structures data into hybrid indices (BM25 + ChromaDB), reranks retrieved context using Cross-Encoders, and leverages LangGraph workflows to answer academic queries with exact page-level citations.

---

## Project Structure

```
research-paper-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── core/                      # Configuration and settings
│   │   ├── services/                  # Business logic (parsing, embeddings, retriever, reranking)
│   │   ├── agent/                     # LangGraph agent state, nodes, and routing edges
│   │   ├── api/                       # API routes (upload, query, evaluate)
│   │   └── evaluation/                # RAGAS metrics evaluator
│   └── Dockerfile
├── frontend/
│   ├── utils/                     # UI helper clients
│   ├── app.py                     # Streamlit application layout
│   └── Dockerfile
├── data/
│   ├── raw/                       # Staged uploaded PDFs
│   └── vector_db/                 # Persistent database directory (Chroma)
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Configure Environments
Copy `.env.example` to `.env` and adjust variables (specifically adding your `GROQ_API_KEY` if using Groq, or adjusting settings for Ollama):
```bash
cp .env.example .env
```

### 2. Local Setup
We recommend using Python virtual environments:

#### Backend:
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app/main.py
```

#### Frontend:
```bash
cd frontend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Running with Docker Compose
To run both backend and frontend automatically inside Docker:
```bash
docker-compose up --build
```
- FastAPI Backend: `http://localhost:8005/docs` (Swagger API docs)
- Streamlit UI: `http://localhost:8501`

---

## Production Best Practices

1. **Non-blocking Async Loops**: FastAPI event loop is kept fully responsive. Heavy CPU-bound blocking operations (like PDF parsing or embedding benchmarks) run inside Python's async threadpool using `asyncio.to_thread`.
2. **Model Caching**: NLP models (SentenceTransformers and Cross-Encoders) are instantiated as singletons and cached in memory. This eliminates model loading overhead (saving ~2GB loading overhead per request) and ensures query latencies remain sub-second.
3. **ChromaDB Segregation**: Multi-embedding architectures isolate vector databases inside separate directories. This guarantees that collections don't run into dimension collisions (e.g., 384 vs 1024).
4. **Fault Tolerance & Resilient Evaluators**: Network API exceptions (e.g. LLM rate limits or invalid keys) are caught at boundaries, logged, and return clean JSON payloads rather than crashing service workers. The RAGAS evaluation runner implements a fallback scoring mechanism to prevent dashboard disruption when API keys are placeholders.

---

## Evaluation Methodology (RAGAS & Embeddings)

Our final year project features a dual-layer evaluation framework to benchmark both the retrieval and generation phases:

### 1. RAGAS Pipeline Quality Assessment
We leverage the **RAGAS (Retrieval Augmented Generation Assessment)** framework to evaluate pipeline performance. The evaluation is run by wrapping our configured LLM (via `LangchainLLMWrapper`) and Embeddings models (via `LangchainEmbeddingsWrapper`) so that evaluations utilize the exact same models that drive the main application:
- **Faithfulness (Grounding)**: Measures if the output claims are supported entirely by the retrieved document chunks.
- **Answer Relevance**: Analyzes if the generated text answers the prompt directly or diverges.
- **Context Precision & Recall**: Determines if the hybrid search retriever and Cross-Encoder rank highly relevant chunks first and capture all necessary source facts.

### 2. Embedding Benchmarks
We compare embedding models (e.g., `all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`) across key operational dimensions:
- **Encoding Latency**: Measures average query and document encoding duration (ms).
- **Cosine Similarity Spread**: Benchmarks average, maximum, and minimum cosine distance bounds on standard academic query-context pairs.

---

## Future Improvements

1. **Graph RAG (Neo4j)**: Constructing an entity-relation knowledge graph to link documents' semantic properties together and answer multi-hop questions.
2. **Contextual Compression**: Using LLMs to compress long chunks before feeding them to generation templates.
3. **Thread State Memory**: Mounting a SqliteSaver checkpoint saver to enable multi-turn agentic chat memory.


