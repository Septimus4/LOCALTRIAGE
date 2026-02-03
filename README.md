# LOCALTRIAGE

A self-hosted, privacy-preserving customer support triage platform powered by local LLMs.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)

## Overview

LOCALTRIAGE is an end-to-end customer support system that:

- **🏷️ Automatically triages** incoming tickets by category and priority
- **📝 Drafts responses** using RAG (Retrieval-Augmented Generation) with citations
- **📊 Generates insights** to identify emerging product issues
- **🔒 Runs entirely locally** - no data leaves your infrastructure

## Features

### Intelligent Routing
- TF-IDF + Logistic Regression baseline classifier
- Multi-class categorization (Billing, Technical, Account, etc.)
- Priority prediction (P1-P4)
- Confidence scoring for human escalation

### RAG-Powered Drafting
- Hybrid retrieval (dense + sparse search with RRF fusion)
- Local LLM generation (Qwen2.5-14B or 7B)
- Automatic citation of knowledge base sources
- Confidence indicators for agent review

### Analytics & Insights
- Weekly trend analysis
- Emerging issue detection via topic clustering
- Resolution time tracking
- Performance dashboards

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 32GB+ RAM (64GB recommended)
- GPU with 24GB+ VRAM (optional, for faster inference)

### 1. Clone and Configure

```bash
git clone https://github.com/your-org/localtriage.git
cd localtriage
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Services

```bash
# CPU-only deployment
docker compose --profile cpu up -d

# With GPU (uncomment llm-gpu in docker-compose.yml)
docker compose up -d
```

### 3. Initialize Database

```bash
# Apply schema
docker compose exec api python -m ingestion.setup

# Ingest sample data (optional)
docker compose exec api python -m ingestion.ingest --sample
```

### 4. Access the Dashboard

Open http://localhost:8501 in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│                    (http://localhost:8501)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                         │
│                    (http://localhost:8080)                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Triage  │  │  Drafting │  │ Retrieval│  │  Analytics │  │
│  │  Router  │  │    RAG    │  │  Hybrid  │  │   Engine   │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬─────┘  │
└───────┼──────────────┼─────────────┼───────────────┼────────┘
        │              │             │               │
┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌──────▼──────┐
│  PostgreSQL  │ │  Ollama   │ │   Qdrant  │ │ BGE-Large   │
│   Database   │ │ qwen3:32b │ │  Vectors  │ │ Embeddings  │
│  (port 5432) │ │(port 11434)│ │(port 6333)│ │ (1024 dim)  │
└──────────────┘ └───────────┘ └───────────┘ └─────────────┘
```

### Tech Stack
- **LLM**: Qwen3:32B via Ollama (22GB VRAM)
- **Embeddings**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **Vector Store**: Qdrant
- **Database**: PostgreSQL 16
- **API Framework**: FastAPI
- **GPU**: NVIDIA RTX 5090 (32GB VRAM, CUDA 13.0)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and component status |
| `/triage` | POST | Classify ticket category and priority |
| `/draft` | POST | Generate RAG-based response draft |
| `/similar` | POST | Find similar historical tickets |
| `/feedback` | POST | Submit agent feedback on drafts |
| `/metrics` | GET | Get system performance metrics |
| `/tickets` | GET | List tickets with pagination |

### Example: Generate Draft

```bash
curl -X POST http://localhost:8080/draft \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot reset password",
    "body": "The reset email never arrived",
    "use_llm": true
  }'
```

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `LLM_BASE_URL` | http://localhost:11434/v1 | LLM API endpoint (Ollama) |
| `LLM_MODEL` | qwen3:32b | Model to use |
| `EMBEDDING_MODEL` | BAAI/bge-large-en-v1.5 | Embedding model (1024 dim) |
| `VECTOR_STORE_TYPE` | qdrant | Vector store (qdrant/faiss) |
| `USE_LLM` | true | Enable LLM drafting |

See [.env.example](.env.example) for full configuration.

## Project Structure

```
localtriage/
├── src/
│   ├── api/          # FastAPI application
│   ├── ingestion/    # Data loading & schema
│   ├── retrieval/    # BM25 & vector search
│   ├── triage/       # Classification/routing
│   ├── rag/          # LLM drafting
│   ├── evaluation/   # Metrics & testing
│   └── ui/           # Streamlit dashboard
├── notebooks/        # Jupyter notebooks
├── docs/             # Documentation
├── infra/            # Docker & monitoring
├── data/             # Data storage
└── tests/            # Test suite
```

## Evaluation

Run the evaluation harness:

```bash
# Full evaluation
python scripts/run_evaluation.py

# Run tests
pytest tests/ -v

# E2E tests (requires API running)
E2E_BASE_URL=http://localhost:8080 pytest tests/e2e/ -v
```

### Current Metrics (2026-02-03)

| Metric | Baseline | Current | Improvement | Notes |
|--------|----------|---------|-------------|-------|
| Routing Accuracy | 30% (majority) | 60.0% | **+30pp** | TF-IDF+LogReg, 30 test samples |
| Retrieval Recall@5 | 70.8% (BM25) | 91.7% | **+20.8pp** | Vector search with BGE-Large |
| Draft Quality (1-5) | 1.5 (template) | **4.74** | **+3.2** | LLM+RAG vs generic templates |
| P95 Latency | <100ms | ~20s | N/A | Trade-off for quality |

**Baselines are now measured, not invented:**
- **Routing**: Random=20%, Majority class=30%, Our TF-IDF+LogReg=60%
- **Retrieval**: BM25=70.8%, Vector (BGE-Large)=91.7%
- **Draft**: Template=1.5/5, LLM+RAG=4.74/5

### Test Suite Status

| Suite | Tests | Status |
|-------|-------|--------|
| Unit Tests | 28 | ✅ All pass |
| API Tests | 17 | ✅ All pass |
| Data Validation | 10 | ✅ All pass |
| Integration Tests | 10 | ✅ All pass |
| E2E Tests | 26 | ✅ All pass (2 skipped*) |
| **Total** | **91** | **✅ Pass** |

*Feedback tests skipped pending draft persistence implementation.

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start development server
uvicorn src.api.api:app --reload --port 8080
```

### Running Notebooks

```bash
jupyter lab --notebook-dir=notebooks
```

## Documentation

- [Business Requirements](docs/BRD.md)
- [Technical Context](docs/CONTEXT.md)
- [Decision Matrix](docs/decision_matrix.md)
- [Project Plan](docs/project_plan.md)
- [Risk Register](docs/risk_register.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

Built with ❤️ for privacy-conscious customer support teams
