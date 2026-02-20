# Context Analysis Document
## Local LLM Customer Support Triage Platform

**Version:** 1.0  
**Date:** February 1, 2026

---

## 1. Industry Context

### 1.1 Customer Support Landscape

The customer support industry is undergoing rapid transformation driven by:

- **Volume Growth**: Average enterprise handles 50-200% more tickets YoY
- **Expectation Shift**: Customers expect <1hr first response, 24/7 availability
- **Agent Burnout**: 40% annual turnover in support roles
- **Knowledge Decay**: Product changes outpace documentation updates

### 1.2 AI Adoption Barriers

Despite clear benefits, enterprise AI adoption faces friction:

| Barrier | % Citing | Our Mitigation |
|---------|----------|----------------|
| Data privacy concerns | 68% | Local-only processing |
| Integration complexity | 54% | Simple API, file-based ingestion |
| Unpredictable quality | 47% | Evaluation harness, baselines |
| Cost uncertainty | 41% | Self-hosted, cost tracking |
| Lack of explainability | 39% | Citation requirements, audit logs |

---

## 2. Technical Context

### 2.1 Local LLM Ecosystem (2026)

The local LLM landscape has matured significantly:

#### Model Options (within 32B constraint)
| Model | Parameters | Strengths | Considerations |
|-------|------------|-----------|----------------|
| Qwen2.5-32B-Instruct | 32B | Strong reasoning, multilingual | Highest quality, needs GPU |
| Qwen2.5-14B-Instruct | 14B | Good balance | Recommended starting point |
| Qwen2.5-7B-Instruct | 7B | Fast, efficient | May need prompt engineering |
| Mistral-Nemo-12B | 12B | Strong English | Good for English-only use |
| Mistral-7B-Instruct | 7B | Very fast | Baseline option |

#### Quantization Options
| Format | Memory Reduction | Quality Impact | Use Case |
|--------|------------------|----------------|----------|
| FP16 | Baseline | None | Production with GPU |
| INT8 | 50% | Minimal | Memory-constrained GPU |
| INT4 (GPTQ) | 75% | Moderate | Large models on limited VRAM |
| Q4_K_M (GGUF) | 75% | Low-Moderate | CPU or mixed inference |

#### Serving Options
| Tool | Strengths | Considerations |
|------|-----------|----------------|
| vLLM | High throughput, PagedAttention | GPU required, best for production |
| llama.cpp | CPU support, quantization | Lower throughput, more flexible |
| Ollama | Easy setup | Less control, good for prototyping |
| Text Generation Inference | Production-ready | Docker-native |

### 2.2 RAG Architecture Patterns

Modern RAG systems combine multiple retrieval strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Processing                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Query       │→ │ Query       │→ │ Hybrid Retrieval        │ │
│  │ Analysis    │  │ Expansion   │  │ (Dense + Sparse)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Retrieval Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Vector Store │  │ BM25/TF-IDF │  │ Similar Ticket Search  ││
│  │ (KB Chunks)  │  │ (Keywords)   │  │ (Historical Context)   ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Reranking & Fusion                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Cross-encoder reranking │ Reciprocal Rank Fusion │ Dedup   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Generation Layer                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Prompt Assembly │ LLM Generation │ Citation Extraction     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Embedding Model Options

| Model | Dimensions | Speed | Quality | Notes |
|-------|------------|-------|---------|-------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Great baseline |
| bge-base-en-v1.5 | 768 | Fast | Very Good | Previous candidate |
| bge-large-en-v1.5 | 1024 | Moderate | Excellent | Previous choice |
| nomic-embed-text | 768 | Fast | Very Good | Long context |
| Qwen3-Embedding-8B | 4096 | Moderate | SOTA | **Selected** — MTEB #1 (70.58), 32K ctx, MRL support |

---

## 3. Organizational Context

### 3.1 Current Support Operations

```
┌─────────────────────────────────────────────────────────────────┐
│                     Current Ticket Flow                         │
│                                                                 │
│  Customer     Ticket      Manual       Agent        Resolution  │
│  Contact  →  Created  →  Triage   →  Assignment →  & Response   │
│              (Auto)      (5-15m)     (Variable)    (10-60m)     │
│                            │                                    │
│                            ▼                                    │
│                    ┌───────────────┐                            │
│                    │ Pain Points:  │                            │
│                    │ • Inconsistent│                            │
│                    │ • Slow        │                            │
│                    │ • Knowledge   │                            │
│                    │   loss        │                            │
│                    └───────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Target Operations

```
┌─────────────────────────────────────────────────────────────────┐
│                      Target Ticket Flow                         │
│                                                                 │
│  Customer     Ticket      Auto        Draft        Agent        │
│  Contact  →  Created  →  Triage  →  Generated →  Review &       │
│              (Auto)      (<5s)      (<30s)       Send           │
│                │           │           │                        │
│                ▼           ▼           ▼                        │
│           ┌─────────────────────────────────┐                   │
│           │ Benefits:                        │                  │
│           │ • Consistent routing             │                  │
│           │ • Grounded responses             │                  │
│           │ • Knowledge preserved            │                  │
│           │ • Insights generated             │                  │
│           └─────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Sources Available

| Source | Format | Volume | Labels | Quality |
|--------|--------|--------|--------|---------|
| Historical tickets | CSV export | 20,000 | Category, Priority | Good |
| KB articles | Markdown | 200 | Tags | Variable |
| Resolution notes | Text field | 15,000 | None | Sparse |
| Agent feedback | None yet | N/A | N/A | To collect |

---

## 4. Competitive Landscape

### 4.1 Alternative Solutions

| Solution | Approach | Pros | Cons |
|----------|----------|------|------|
| Zendesk AI | Cloud SaaS | Integrated, turnkey | Data leaves premises |
| Intercom Fin | Cloud SaaS | Good UX | Privacy concerns |
| Custom GPT-4 | API calls | High quality | Cost, latency, privacy |
| **This Project** | Local LLM | Privacy, control, cost | Setup complexity |

### 4.2 Differentiation

Our solution is differentiated by:

1. **Complete Data Sovereignty**: All processing local
2. **Transparent Costs**: Fixed infrastructure, no per-token fees
3. **Full Auditability**: Every decision logged and traceable
4. **Customizable**: Models and prompts fully under control
5. **No Vendor Lock-in**: Open source components throughout

---

## 5. Technical Constraints & Decisions

### 5.1 Hardware Assumptions

**Minimum Configuration (CPU-focused):**
- 32GB RAM
- 8+ CPU cores
- 100GB SSD
- Model: Qwen2.5-7B-Q4_K_M

**Recommended Configuration (GPU):**
- 64GB RAM
- 16+ CPU cores
- NVIDIA GPU with 24GB VRAM (RTX 4090 / A10)
- 500GB SSD
- Model: Qwen2.5-14B-Instruct

**Production Configuration:**
- 128GB RAM
- 32+ CPU cores
- NVIDIA GPU with 48GB+ VRAM (A40 / A100)
- 1TB NVMe
- Model: Qwen2.5-32B-Instruct

### 5.2 Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary database | PostgreSQL | Mature, full-text search, JSON support |
| Vector store | Qdrant | Production-ready, filtering, easy setup |
| Embedding model | Qwen3-Embedding-8B | SOTA quality (MTEB #1), 4096-dim, 32K context |
| LLM serving | vLLM (GPU) / llama.cpp (CPU) | Flexibility for different hardware |
| API framework | FastAPI | Modern, async, auto-docs |
| UI framework | Streamlit | Rapid prototyping, data-friendly |

### 5.3 Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Request parsing | <10ms | FastAPI overhead |
| Embedding generation | <100ms | Batch if possible |
| Vector search | <50ms | Qdrant optimized |
| BM25 search | <20ms | PostgreSQL FTS |
| Reranking | <200ms | Optional cross-encoder |
| LLM generation | <25s | Primary bottleneck |
| Response formatting | <10ms | JSON serialization |
| **Total** | **<30s** | p95 target |

---

## 6. Success Factors

### 6.1 Critical Success Factors

1. **Data Quality**: Clean, labeled data for evaluation
2. **KB Coverage**: Comprehensive knowledge base
3. **Evaluation Rigor**: Systematic quality measurement
4. **User Adoption**: Agent buy-in and feedback
5. **Performance Tuning**: Meeting latency targets

### 6.2 Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM hallucination | High | High | Citation requirements, confidence thresholds |
| Poor retrieval | Medium | High | Hybrid search, iterative tuning |
| Slow inference | Medium | Medium | Quantization, caching, async |
| Scope creep | High | Medium | Strict MVP definition |
| Data quality issues | Medium | High | Validation pipelines |

---

## 7. Assumptions & Dependencies

### 7.1 Assumptions

- Support team will provide feedback during development
- Historical ticket data can be exported and anonymized
- GPU hardware available for production (CPU acceptable for MVP)
- KB articles are reasonably accurate and up-to-date
- English-only support is acceptable for MVP

### 7.2 Dependencies

| Dependency | Type | Status | Contingency |
|------------|------|--------|-------------|
| Ticket export | Data | Pending | Use public dataset |
| KB access | Data | Confirmed | Create sample KB |
| GPU server | Infrastructure | Optional | Use quantized CPU models |
| Agent availability | Human | Pending | Use simulated feedback |

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **BM25** | Best Matching 25 - sparse retrieval algorithm |
| **Dense Retrieval** | Semantic search using embeddings |
| **GGUF** | GPT-Generated Unified Format - quantized model format |
| **PagedAttention** | Memory optimization technique for LLM serving |
| **RAG** | Retrieval-Augmented Generation |
| **SLA** | Service Level Agreement |
| **Vector Store** | Database optimized for similarity search |
| **vLLM** | High-throughput LLM serving framework |
