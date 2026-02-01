# Business Requirements Document (BRD)
## Local LLM Customer Support Triage + Response Drafting + Analytics Platform

**Version:** 1.0  
**Date:** February 1, 2026  
**Author:** Support Engineering Team  
**Status:** Draft

---

## 1. Executive Summary

This document outlines the business requirements for a self-hosted, privacy-preserving customer support triage system. The platform will leverage local Large Language Models (LLMs) to automate ticket routing, draft responses using Retrieval-Augmented Generation (RAG), and provide actionable product insights—all while maintaining complete data sovereignty.

---

## 2. Business Context

### 2.1 Problem Statement

Current support operations face several challenges:
- **Manual triage delays:** Agents spend 15-30 minutes per shift on ticket routing
- **Inconsistent responses:** Quality varies significantly across agents and shifts
- **Knowledge silos:** Institutional knowledge trapped in individual agents' heads
- **Reactive insights:** Product issues discovered late, often from escalations
- **Privacy constraints:** Cannot use cloud-based AI due to data sensitivity

### 2.2 Business Opportunity

By deploying a local LLM-powered system, we can:
- Reduce time-to-first-response by 40-60%
- Improve routing accuracy to >90%
- Ensure consistent, policy-compliant response quality
- Proactively surface product issues before they escalate
- Maintain complete control over customer data

---

## 3. Stakeholder Analysis

| Stakeholder | Role | Primary Needs | Success Metrics |
|-------------|------|---------------|-----------------|
| **Support Agent** | End user | Faster drafts, relevant context | Draft acceptance rate, time saved |
| **Support Lead** | Manager | Routing accuracy, SLA visibility | Correct routing %, SLA compliance |
| **Product Manager** | Consumer | Issue trends, evidence | Theme detection coverage |
| **Security/IT** | Gatekeeper | Local processing, audit trails | Zero data egress, full traceability |

---

## 4. Target Business KPIs

### 4.1 Efficiency Metrics
| KPI | Baseline | Target | Measurement |
|-----|----------|--------|-------------|
| Time-to-first-draft | 8 min | 2 min | Avg time from ticket creation to draft ready |
| Tickets processed/hour/agent | 6 | 12 | Throughput tracking |

### 4.2 Quality Metrics
| KPI | Baseline | Target | Measurement |
|-----|----------|--------|-------------|
| Correct routing rate | 72% | 92% | Routing prediction vs final assignment |
| Draft acceptance rate | N/A | 70% | Drafts used with <20% modification |
| Hallucination rate | N/A | <2% | Claims without KB grounding |
| Policy violation rate | N/A | <0.5% | Automated + manual audit |

### 4.3 Insight Metrics
| KPI | Baseline | Target | Measurement |
|-----|----------|--------|-------------|
| Theme detection coverage | 40% | 85% | Issues surfaced before escalation |
| Emerging issue lead time | 5 days | 1 day | Time from first ticket to theme alert |

### 4.4 Operational Metrics
| KPI | Target | Measurement |
|-----|--------|-------------|
| Compute cost per 1k tickets | <$5 equivalent | GPU-hours × rate |
| System availability | 99.5% | Uptime monitoring |
| End-to-end latency (p95) | <30s | Request logging |

---

## 5. Functional Requirements

### 5.1 Data Ingestion (FR-ING)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-ING-001 | Ingest tickets from CSV format | Must | Successfully parse and store 100% of valid records |
| FR-ING-002 | Ingest tickets from JSON format | Must | Support nested JSON structures |
| FR-ING-003 | Store normalized ticket records | Must | Schema validation passes |
| FR-ING-004 | Ingest KB articles (Markdown) | Must | Preserve formatting, extract metadata |
| FR-ING-005 | Version KB content | Should | Track changes, support rollback |
| FR-ING-006 | Chunk KB for retrieval | Must | Configurable chunk size, overlap |

### 5.2 Ticket Triage (FR-TRI)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-TRI-001 | Predict ticket category | Must | Multi-class classification, top-3 predictions |
| FR-TRI-002 | Predict ticket priority | Must | P1-P4 with confidence scores |
| FR-TRI-003 | Suggest assignee queue | Should | Based on category + skills matrix |
| FR-TRI-004 | Detect SLA risk | Should | Flag tickets at risk of breach |
| FR-TRI-005 | Explain routing decision | Should | Provide reasoning for predictions |

### 5.3 Response Drafting (FR-DRF)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-DRF-001 | Retrieve relevant KB chunks | Must | Top-k retrieval with similarity scores |
| FR-DRF-002 | Retrieve similar past tickets | Must | Include resolution if available |
| FR-DRF-003 | Generate grounded response draft | Must | All claims cite sources |
| FR-DRF-004 | Include citations in output | Must | Inline references to KB/tickets |
| FR-DRF-005 | Provide confidence indicators | Should | Flag low-confidence sections |
| FR-DRF-006 | Support template override | Should | Agent can select response template |

### 5.4 Analytics (FR-ANA)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-ANA-001 | Volume trend visualization | Must | Daily/weekly/monthly views |
| FR-ANA-002 | Category distribution | Must | Breakdown with drill-down |
| FR-ANA-003 | Emerging cluster detection | Should | Auto-detect new issue patterns |
| FR-ANA-004 | SLA compliance dashboard | Should | Real-time risk indicators |
| FR-ANA-005 | Weekly insights report | Must | Automated generation |

### 5.5 Feedback Loop (FR-FBK)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-FBK-001 | Capture draft usefulness rating | Must | 1-5 scale + optional comment |
| FR-FBK-002 | Track agent corrections | Should | Diff between draft and sent |
| FR-FBK-003 | Store feedback for retraining | Should | Structured storage with labels |

---

## 6. Non-Functional Requirements

### 6.1 Privacy & Security (NFR-SEC)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| NFR-SEC-001 | Fully local inference | Must | Zero external API calls for LLM |
| NFR-SEC-002 | No data egress | Must | Network audit shows no external transfers |
| NFR-SEC-003 | Audit trail for all operations | Must | Complete request/response logging |
| NFR-SEC-004 | Role-based access control | Should | Admin, Lead, Agent roles |

### 6.2 Performance (NFR-PRF)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| NFR-PRF-001 | Draft generation <30s (p95) | Must | Performance tests pass |
| NFR-PRF-002 | Triage prediction <5s (p95) | Must | Performance tests pass |
| NFR-PRF-003 | Support 10 concurrent users | Should | Load test passes |
| NFR-PRF-004 | Process 1000 tickets/hour | Should | Batch processing benchmark |

### 6.3 Traceability (NFR-TRC)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| NFR-TRC-001 | Log prompt for every generation | Must | Queryable log store |
| NFR-TRC-002 | Log retrieved documents | Must | Document IDs and scores |
| NFR-TRC-003 | Log model config per request | Must | Model, params, quantization |
| NFR-TRC-004 | Reproducible generation | Should | Same inputs → same outputs (temp=0) |

### 6.4 Maintainability (NFR-MNT)
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| NFR-MNT-001 | Docker-based deployment | Must | Single docker-compose up |
| NFR-MNT-002 | Baseline rollback capability | Must | Switch to non-LLM mode in <5min |
| NFR-MNT-003 | Model hot-swap | Should | Change model without restart |
| NFR-MNT-004 | Configuration externalization | Must | All configs in env/yaml files |

---

## 7. Scope Boundaries

### 7.1 In Scope
- CSV and JSON ticket ingestion
- Markdown KB article ingestion
- Local LLM inference (Qwen/Mistral up to 32B)
- RAG-based response drafting
- Basic analytics dashboard
- Feedback collection
- Docker-based deployment

### 7.2 Out of Scope
- Real-time ticket system integration (webhooks)
- Multi-language support (future phase)
- Voice/chat channel support
- Customer-facing chatbot
- Fine-tuning infrastructure
- Production-grade authentication

### 7.3 Assumptions
- GPU hardware available (or CPU acceptable for demo)
- Labeled data available for evaluation
- KB content is reasonably current and accurate
- Support team available for feedback sessions

### 7.4 Constraints
- Maximum model size: 32B parameters
- Local-only deployment
- 2-3 week implementation timeline
- Single-node deployment (no distributed systems)

---

## 8. Acceptance Criteria

### 8.1 Minimum Viable Product (MVP)
- [ ] Ingest 2,000+ tickets from CSV/JSON
- [ ] Ingest 50+ KB articles
- [ ] Route tickets with >80% accuracy
- [ ] Generate drafts with citations
- [ ] Display basic analytics
- [ ] Collect agent feedback
- [ ] Deploy via Docker Compose

### 8.2 Target Product
- [ ] Route tickets with >90% accuracy
- [ ] Draft acceptance rate >60%
- [ ] Hallucination rate <5%
- [ ] End-to-end latency <30s (p95)
- [ ] Weekly insights report generation
- [ ] Full audit logging

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM quality insufficient | Medium | High | Baseline fallback, multiple model options |
| Latency exceeds targets | Medium | Medium | Quantization, caching, async processing |
| KB coverage gaps | High | Medium | Confidence thresholds, escalation paths |
| GPU unavailable | Low | High | CPU fallback with smaller models |
| Evaluation data insufficient | Medium | High | Create annotation guidelines, synthetic data |

---

## 10. Requirement Traceability Matrix

| Requirement | Deliverable | Test Method |
|-------------|-------------|-------------|
| FR-ING-001 | ingestion/csv_loader.py | Unit tests + integration |
| FR-ING-004 | ingestion/kb_loader.py | Unit tests + integration |
| FR-TRI-001 | triage/classifier.py | Evaluation harness |
| FR-DRF-001 | retrieval/vector_search.py | Recall@k metrics |
| FR-DRF-003 | rag/drafter.py | Rubric evaluation |
| FR-ANA-001 | ui/dashboard.py | Manual inspection |
| NFR-SEC-001 | docker-compose.yml | Network audit |
| NFR-TRC-001 | monitoring/logger.py | Log inspection |

---

## 11. Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Sponsor | | | |
| Support Lead | | | |
| Security | | | |
| Engineering Lead | | | |

---

## Appendix A: Glossary

- **RAG**: Retrieval-Augmented Generation - combining search with LLM generation
- **KB**: Knowledge Base - curated articles and documentation
- **SLA**: Service Level Agreement - contractual response time commitments
- **Triage**: Process of categorizing and prioritizing support tickets
- **Hallucination**: LLM generating factually incorrect or unsupported claims
