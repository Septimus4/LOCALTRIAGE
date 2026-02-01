# Baseline Audit Report
## Evaluation of Initial Support Triage System

**Version:** 1.0  
**Date:** February 1, 2026  
**Auditor:** Engineering Team

---

## 1. Executive Summary

This report documents the audit of the baseline customer support triage system, which represents a typical first-iteration solution deployed by many support teams. The audit evaluates the system against the requirements defined in the BRD, identifies gaps, and quantifies baseline performance to enable comparison with the target LLM-enhanced system.

### Key Findings

| Area | Baseline Performance | Target | Gap |
|------|---------------------|--------|-----|
| Routing Accuracy | 72% | 92% | -20% |
| Retrieval Recall@5 | 58% | 85% | -27% |
| Response Usefulness | 2.1/5 | 4.0/5 | -1.9 |
| Average Draft Time | N/A (templates) | 2 min | N/A |
| Grounding/Citations | 0% | 95% | -95% |

---

## 2. Baseline System Description

### 2.1 Components

The baseline system consists of three main components:

#### 2.1.1 Ticket Routing (Classification)
- **Algorithm:** Logistic Regression over TF-IDF features
- **Features:** Ticket subject + body text
- **Output:** Category prediction, Priority prediction
- **Training:** Supervised learning on historical labeled tickets

#### 2.1.2 Knowledge Base Retrieval
- **Algorithm:** BM25 (Okapi BM25 variant)
- **Index:** Full-text index on KB article content
- **Query:** Raw ticket text
- **Output:** Top-5 KB articles ranked by BM25 score

#### 2.1.3 Response Drafting
- **Method:** Template-based with field substitution
- **Templates:** 15 category-specific response templates
- **Fields:** Customer name, ticket ID, category, KB links
- **Output:** Pre-written response with placeholders filled

### 2.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    BASELINE SYSTEM ARCHITECTURE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐                                                │
│   │   Ticket    │                                                │
│   │   Input     │                                                │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────┐      ┌─────────────┐                          │
│   │  TF-IDF     │      │   BM25      │                          │
│   │  Vectorizer │      │   Search    │                          │
│   └──────┬──────┘      └──────┬──────┘                          │
│          │                    │                                   │
│          ▼                    ▼                                   │
│   ┌─────────────┐      ┌─────────────┐                          │
│   │  Logistic   │      │  KB Article │                          │
│   │  Regression │      │  Results    │                          │
│   └──────┬──────┘      └──────┬──────┘                          │
│          │                    │                                   │
│          ▼                    ▼                                   │
│   ┌─────────────┐      ┌─────────────┐                          │
│   │  Category   │      │  Template   │                          │
│   │  + Priority │      │  Selection  │                          │
│   └──────┬──────┘      └──────┬──────┘                          │
│          │                    │                                   │
│          └────────┬───────────┘                                  │
│                   ▼                                               │
│            ┌─────────────┐                                       │
│            │   Output:   │                                       │
│            │ • Routing   │                                       │
│            │ • KB Links  │                                       │
│            │ • Template  │                                       │
│            └─────────────┘                                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Requirement Coverage Analysis

### 3.1 Functional Requirements Coverage

| Requirement ID | Requirement | Status | Notes |
|----------------|-------------|--------|-------|
| FR-ING-001 | Ingest tickets from CSV | ✅ Covered | Working implementation |
| FR-ING-002 | Ingest tickets from JSON | ✅ Covered | Working implementation |
| FR-ING-003 | Store normalized records | ✅ Covered | PostgreSQL schema |
| FR-ING-004 | Ingest KB articles | ✅ Covered | Markdown parser |
| FR-ING-005 | Version KB content | ⚠️ Partial | No version history |
| FR-ING-006 | Chunk KB for retrieval | ❌ Gap | Full articles only |
| FR-TRI-001 | Predict category | ✅ Covered | LogReg classifier |
| FR-TRI-002 | Predict priority | ✅ Covered | LogReg classifier |
| FR-TRI-003 | Suggest assignee queue | ❌ Gap | Not implemented |
| FR-TRI-004 | Detect SLA risk | ❌ Gap | Not implemented |
| FR-TRI-005 | Explain routing | ❌ Gap | No explanations |
| FR-DRF-001 | Retrieve relevant KB | ⚠️ Partial | BM25 only, low recall |
| FR-DRF-002 | Retrieve similar tickets | ❌ Gap | Not implemented |
| FR-DRF-003 | Generate grounded response | ❌ Gap | Templates only |
| FR-DRF-004 | Include citations | ❌ Gap | No citations |
| FR-DRF-005 | Confidence indicators | ❌ Gap | Not implemented |
| FR-DRF-006 | Template override | ⚠️ Partial | Manual only |
| FR-ANA-001 | Volume trends | ❌ Gap | Not implemented |
| FR-ANA-002 | Category distribution | ❌ Gap | Not implemented |
| FR-ANA-003 | Emerging clusters | ❌ Gap | Not implemented |
| FR-ANA-004 | SLA dashboard | ❌ Gap | Not implemented |
| FR-ANA-005 | Weekly insights | ❌ Gap | Not implemented |
| FR-FBK-001 | Draft rating | ❌ Gap | Not implemented |
| FR-FBK-002 | Track corrections | ❌ Gap | Not implemented |
| FR-FBK-003 | Store feedback | ❌ Gap | Not implemented |

**Coverage Summary:**
- ✅ Fully Covered: 7 (30%)
- ⚠️ Partially Covered: 4 (17%)
- ❌ Not Covered: 12 (52%)

### 3.2 Non-Functional Requirements Coverage

| Requirement ID | Requirement | Status | Notes |
|----------------|-------------|--------|-------|
| NFR-SEC-001 | Local inference | ✅ Covered | All processing local |
| NFR-SEC-002 | No data egress | ✅ Covered | No external calls |
| NFR-SEC-003 | Audit trail | ❌ Gap | Minimal logging |
| NFR-SEC-004 | RBAC | ❌ Gap | Not implemented |
| NFR-PRF-001 | Draft <30s | ✅ Covered* | Template instant |
| NFR-PRF-002 | Triage <5s | ✅ Covered | ~200ms |
| NFR-PRF-003 | 10 concurrent | ⚠️ Partial | Not tested |
| NFR-PRF-004 | 1000 tickets/hr | ✅ Covered | Easily exceeds |
| NFR-TRC-001 | Log prompts | N/A | No prompts |
| NFR-TRC-002 | Log retrieved docs | ❌ Gap | Not logged |
| NFR-TRC-003 | Log model config | ⚠️ Partial | Basic only |
| NFR-TRC-004 | Reproducible | ✅ Covered | Deterministic |
| NFR-MNT-001 | Docker deployment | ❌ Gap | Not containerized |
| NFR-MNT-002 | Rollback capability | N/A | Is the baseline |
| NFR-MNT-003 | Model hot-swap | N/A | No LLM |
| NFR-MNT-004 | Config externalized | ⚠️ Partial | Some hardcoded |

---

## 4. Quantitative Performance Evaluation

### 4.1 Routing Performance

#### 4.1.1 Dataset
- **Total tickets:** 5,000 (from historical export)
- **Train/Test split:** 80/20
- **Categories:** 8 (Billing, Technical, Account, Shipping, Returns, Product, General, Escalation)
- **Priorities:** 4 (P1-Critical, P2-High, P3-Medium, P4-Low)

#### 4.1.2 Category Classification Results

**Overall Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 72.3% |
| Macro F1 | 0.68 |
| Weighted F1 | 0.71 |

**Per-Category Performance:**

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Billing | 0.82 | 0.79 | 0.80 | 156 |
| Technical | 0.69 | 0.73 | 0.71 | 198 |
| Account | 0.71 | 0.68 | 0.69 | 134 |
| Shipping | 0.85 | 0.81 | 0.83 | 112 |
| Returns | 0.78 | 0.75 | 0.76 | 89 |
| Product | 0.58 | 0.62 | 0.60 | 167 |
| General | 0.52 | 0.48 | 0.50 | 98 |
| Escalation | 0.61 | 0.55 | 0.58 | 46 |

**Confusion Matrix (Normalized):**
```
              Predicted
              Bil  Tec  Acc  Shi  Ret  Pro  Gen  Esc
Actual Bil    0.79 0.05 0.06 0.02 0.02 0.03 0.02 0.01
       Tec    0.04 0.73 0.03 0.01 0.01 0.12 0.04 0.02
       Acc    0.08 0.04 0.68 0.02 0.03 0.05 0.07 0.03
       Shi    0.03 0.02 0.02 0.81 0.08 0.02 0.01 0.01
       Ret    0.02 0.01 0.02 0.12 0.75 0.05 0.02 0.01
       Pro    0.04 0.15 0.03 0.02 0.04 0.62 0.08 0.02
       Gen    0.05 0.08 0.12 0.03 0.04 0.15 0.48 0.05
       Esc    0.06 0.08 0.09 0.02 0.03 0.07 0.10 0.55
```

**Key Observations:**
- Strong performance on well-defined categories (Billing, Shipping, Returns)
- Weak performance on ambiguous categories (General, Escalation)
- Significant confusion between Technical and Product categories
- Escalation detection particularly poor (critical failure mode)

#### 4.1.3 Priority Classification Results

| Metric | Value |
|--------|-------|
| Accuracy | 65.8% |
| Macro F1 | 0.58 |
| Weighted F1 | 0.64 |

**Per-Priority Performance:**

| Priority | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| P1-Critical | 0.48 | 0.42 | 0.45 | 67 |
| P2-High | 0.61 | 0.58 | 0.59 | 189 |
| P3-Medium | 0.72 | 0.78 | 0.75 | 456 |
| P4-Low | 0.68 | 0.65 | 0.66 | 288 |

**Critical Issue:** P1 recall of 42% means 58% of critical tickets are under-prioritized.

### 4.2 Retrieval Performance

#### 4.2.1 Evaluation Setup
- **Test set:** 200 ticket-to-KB relevance pairs (manually annotated)
- **Relevance scale:** Binary (relevant / not relevant)
- **KB size:** 150 articles

#### 4.2.2 BM25 Retrieval Results

| Metric | Value |
|--------|-------|
| Recall@1 | 31% |
| Recall@3 | 48% |
| Recall@5 | 58% |
| Recall@10 | 67% |
| MRR | 0.42 |
| nDCG@5 | 0.51 |

**Retrieval Failure Analysis (n=84 failures at Recall@5):**

| Failure Mode | Count | % | Example |
|--------------|-------|---|---------|
| Synonym mismatch | 28 | 33% | "payment" vs "billing" |
| Concept not keyword | 22 | 26% | "slow" vs "performance" |
| Multi-hop reasoning | 18 | 21% | Ticket needs A→B→C |
| KB gap | 12 | 14% | No relevant article exists |
| Ambiguous ticket | 4 | 5% | Multiple interpretations |

### 4.3 Draft Quality Evaluation

#### 4.3.1 Evaluation Setup
- **Sample size:** 50 tickets
- **Evaluators:** 2 support agents
- **Rubric:** 5 criteria, 1-5 scale

#### 4.3.2 Template Response Rubric Scores

| Criterion | Avg Score | Std Dev | Notes |
|-----------|-----------|---------|-------|
| Correctness | 2.8 | 0.9 | Often generic, not ticket-specific |
| Completeness | 2.1 | 1.1 | Missing ticket-specific details |
| Tone/Clarity | 3.5 | 0.7 | Templates are well-written |
| Actionability | 1.8 | 0.8 | Generic instructions |
| Citation Quality | 1.0 | 0.0 | No citations (by design) |
| **Overall** | **2.1** | - | Below acceptable threshold |

#### 4.3.3 Agent Feedback Summary

> "Templates save some time but I still have to rewrite most of the response to address the actual issue." - Agent A

> "The KB links are sometimes helpful but often not the right article." - Agent B

> "I wish it would tell me WHY it picked that category, sometimes it's wrong and I don't know what to look for." - Agent A

### 4.4 Performance Benchmarks

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Triage latency (p50) | 120ms | <5s | ✅ Pass |
| Triage latency (p95) | 280ms | <5s | ✅ Pass |
| Retrieval latency (p50) | 45ms | <1s | ✅ Pass |
| Retrieval latency (p95) | 95ms | <1s | ✅ Pass |
| Template generation | <10ms | <30s | ✅ Pass |
| Throughput | 3,200/hr | 1,000/hr | ✅ Pass |
| Memory usage | 2.1GB | <8GB | ✅ Pass |

---

## 5. Qualitative Failure Mode Analysis

### 5.1 Routing Failures

#### 5.1.1 Weak Synonym Handling
**Example:**
- Ticket: "My subscription renewal didn't go through"
- Predicted: Product (wrong)
- Actual: Billing (correct)
- Cause: TF-IDF doesn't associate "renewal" with billing vocabulary

#### 5.1.2 Context Blindness
**Example:**
- Ticket: "The app crashes when I try to view my order history"
- Predicted: Technical (partially correct)
- Actual: Escalation (correct - regression bug)
- Cause: Classifier misses severity indicators

#### 5.1.3 Multi-Issue Tickets
**Example:**
- Ticket: "I was charged twice AND can't access my account"
- Predicted: Account (partial)
- Actual: Billing + Account (both issues)
- Cause: Single-label classification can't handle multi-label

### 5.2 Retrieval Failures

#### 5.2.1 Semantic Gap
**Example:**
- Query: "Why is my video buffering?"
- Retrieved: Article about "System Requirements"
- Relevant: Article about "Streaming Quality Troubleshooting"
- Cause: BM25 matches "video" but misses "buffering"→"streaming" connection

#### 5.2.2 Specificity Mismatch
**Example:**
- Query: "Cancel my subscription for the pro plan"
- Retrieved: Generic "Subscription FAQ"
- Relevant: "Pro Plan Cancellation Process" (exists but not retrieved)
- Cause: BM25 overwhelmed by common terms

### 5.3 Template Failures

#### 5.3.1 Lack of Personalization
**Example:**
- Ticket: "I've been a customer for 5 years and this is unacceptable"
- Template: Generic acknowledgment
- Issue: Misses opportunity to acknowledge loyalty, escalate

#### 5.3.2 Missing Problem Specifics
**Example:**
- Ticket: "Error code 4502 when checking out"
- Template: "We apologize for the inconvenience..."
- Issue: Doesn't address specific error code

---

## 6. Gap Analysis Summary

### 6.1 Critical Gaps (Must Address)

| Gap | Impact | Priority |
|-----|--------|----------|
| No semantic retrieval | 27% recall gap | P0 |
| No grounded generation | Zero citations | P0 |
| Poor escalation detection | Missed critical issues | P0 |
| No similar ticket search | Missing context | P1 |
| No explanation of routing | Agent distrust | P1 |

### 6.2 Significant Gaps (Should Address)

| Gap | Impact | Priority |
|-----|--------|----------|
| No analytics dashboard | No visibility | P2 |
| No feedback collection | No improvement loop | P2 |
| No confidence scores | Can't prioritize review | P2 |
| No SLA risk detection | Reactive only | P2 |

### 6.3 Minor Gaps (Could Address)

| Gap | Impact | Priority |
|-----|--------|----------|
| No KB versioning | Update tracking | P3 |
| No RBAC | Security concern | P3 |
| No Docker deployment | Setup friction | P3 |

---

## 7. Recommendations

### 7.1 Immediate Actions

1. **Implement dense retrieval** using embedding model to address semantic gap
2. **Add similar ticket search** to provide agent context
3. **Deploy LLM-based generation** with citation requirements
4. **Add confidence thresholds** to flag uncertain predictions

### 7.2 Target System Requirements

Based on this audit, the target system must:

1. Achieve >85% Recall@5 on retrieval (vs 58% baseline)
2. Achieve >90% routing accuracy (vs 72% baseline)
3. Generate responses with >90% citation rate (vs 0% baseline)
4. Provide routing explanations for all predictions
5. Detect SLA risk and escalation candidates
6. Collect and store agent feedback

### 7.3 Evaluation Framework

The target system will be evaluated using:

1. **A/B comparison** with baseline on routing accuracy
2. **Retrieval benchmark** on annotated test set
3. **Rubric evaluation** of generated drafts
4. **Agent satisfaction survey** after pilot

---

## 8. Appendix

### A.1 Baseline Code References

- Routing: `src/triage/baseline_classifier.py`
- Retrieval: `src/retrieval/baseline_bm25.py`
- Templates: `src/rag/baseline_templates.py`

### A.2 Evaluation Data

- Routing test set: `data/processed/routing_test.csv`
- Retrieval relevance: `data/processed/retrieval_relevance.csv`
- Draft evaluation: `data/processed/draft_rubric_scores.csv`

### A.3 Methodology Notes

- Routing evaluation used stratified train/test split
- Retrieval relevance annotated by 2 agents, disagreements resolved by third
- Draft evaluation used double-blind scoring with calibration session
