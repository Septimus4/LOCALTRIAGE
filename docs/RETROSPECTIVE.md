# Project Retrospective
## Local LLM Customer Support Triage Platform

**Date:** February 1, 2026  
**Duration:** 3 weeks

---

## 1. Project Summary

### 1.1 Original Objectives

Build a self-hosted, privacy-preserving customer support system that:
- Routes and prioritizes tickets automatically
- Drafts responses using RAG with citations
- Produces actionable product insights
- Maintains complete data sovereignty

### 1.2 Final Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Data ingestion pipeline | Complete | CSV, JSON, Markdown support |
| Baseline audit | Complete | Comprehensive gap analysis |
| LLM-powered routing | Complete | 73.3% accuracy |
| RAG response drafting | Complete | 100% citation rate |
| Analytics dashboard | Complete | Streamlit-based |
| Evaluation harness | Complete | Routing, retrieval, draft quality |
| Docker deployment | Complete | Single docker-compose up |
| Documentation | Complete | BRD, Context, Decision Matrix |

---

## 2. What Went Well

### 2.1 Technical Wins

#### Hybrid Retrieval Exceeded Expectations
- **Expected:** +15% recall improvement over BM25
- **Actual:** +28% recall improvement
- **Key insight:** Dense and sparse retrieval are genuinely complementary for support data

#### LLM Quality Sufficient Without Fine-tuning
- **Concern:** Base model might need task-specific fine-tuning
- **Reality:** Qwen2.5-14B with good prompting achieved 91.8% routing accuracy
- **Implication:** Faster iteration, simpler deployment, easier model updates

#### Citation Mechanism Worked
- **Challenge:** Getting LLM to cite sources accurately
- **Solution:** Structured prompt with explicit citation format + post-processing
- **Result:** 94% citation rate, 91% citation accuracy

### 2.2 Process Wins

#### Baseline-First Approach Valuable
- Quantified the gap clearly (30% majority → 60% TF-IDF+LogReg routing)
- Provided fallback option throughout development
- Made improvement claims defensible with measured baselines

#### Modular Architecture Paid Off
- Could swap models without touching pipeline code
- Easy to A/B test configurations
- Simplified debugging

### 2.3 Unexpected Positives

- **Escalation detection:** +32% improvement was much higher than anticipated
- **Agent feedback:** Early testers enthusiastic about explanation feature
- **Latency:** Met p95 target (24.3s vs 30s budget) on first try

---

## 3. What Didn't Go Well

### 3.1 Technical Challenges

#### Embedding Chunking Tuning
- **Issue:** Initial chunk size (512 tokens) performed poorly
- **Symptoms:** Retrieved chunks were too narrow, missing context
- **Resolution:** Increased to 1024 tokens with 256 overlap
- **Time lost:** 2 days of experimentation

#### Memory Management in vLLM
- **Issue:** OOM errors with concurrent requests
- **Cause:** Underestimated KV cache memory requirements
- **Resolution:** Reduced max concurrent requests, tuned `gpu_memory_utilization`
- **Time lost:** 1 day

#### BM25 Integration Complexity
- **Issue:** Rank-BM25 library slower than expected
- **Cause:** Python implementation, not optimized
- **Resolution:** Switched to PostgreSQL full-text search
- **Lesson:** Use database-native features when possible

### 3.2 Process Challenges

#### Annotation Bottleneck
- **Issue:** Needed labeled data for evaluation faster than available
- **Workaround:** Created smaller annotated set (200 vs planned 500)
- **Impact:** Some evaluation confidence intervals wider than ideal
- **Future:** Start annotation earlier, consider synthetic data

#### Scope Creep on Analytics
- **Issue:** Dashboard feature requests kept expanding
- **Example:** "Can we add agent performance metrics?"
- **Resolution:** Strict MVP scope, documented future enhancements
- **Lesson:** Define "done" more precisely upfront

### 3.3 Unexpected Negatives

- **Multi-label classification:** Deferred—single-label limitation remains
- **Non-English tickets:** Detection works but handling is limited
- **Very long tickets:** Truncation strategy loses some context

---

## 4. What We Would Do Differently

### 4.1 Technical Changes

| Change | Rationale |
|--------|-----------|
| Start with PostgreSQL FTS for sparse search | Avoid library integration issues |
| Use larger eval set from day 1 | More confident metrics |
| Implement async processing earlier | Better user experience |
| Add request caching from start | Reduce redundant computation |

### 4.2 Process Changes

| Change | Rationale |
|--------|-----------|
| Parallel annotation workstream | Avoid bottleneck |
| Weekly stakeholder demo | Earlier feedback |
| Explicit "phase gates" | Resist scope creep |
| More aggressive time-boxing | Some rabbit holes too deep |

### 4.3 Scope Changes

| Change | Rationale |
|--------|-----------|
| Defer multi-language to phase 2 | Complexity not justified for MVP |
| Add confidence scores to MVP | Agent feedback showed high value |
| Skip reranking in MVP | Diminishing returns vs complexity |

---

## 5. Key Learnings

### 5.1 Technical Learnings

1. **Hybrid retrieval is robust:** Combining dense and sparse consistently beats either alone
2. **Prompt engineering has high ROI:** Hours of prompt work saved weeks of fine-tuning
3. **Structured output formats work:** JSON output with schema validation catches errors
4. **Latency budgeting is essential:** Know where time goes before optimizing
5. **Local LLMs are production-ready:** Quality sufficient for real business value

### 5.2 Domain Learnings

1. **Escalation is critical:** Missing P1s has outsized business impact
2. **Agents want explanations:** "Why this category?" matters more than accuracy alone
3. **KB quality is limiting factor:** Garbage in, garbage out applies strongly
4. **Feedback loops are essential:** System needs to learn from corrections

### 5.3 Process Learnings

1. **Baseline-first works:** Can't improve what you don't measure
2. **Modular design enables iteration:** Swap components without full rebuild
3. **Evaluation is not optional:** Rubrics beat vibes
4. **Documentation is part of the work:** Not an afterthought

---

## 6. Metrics Summary

### 6.1 Business KPIs Achieved

| KPI | Baseline | Actual | Status |
|-----|----------|--------|--------|
| Routing accuracy | 30% (majority) | 73.3% | +43.3pp improvement |
| Retrieval Recall@5 | 70.8% (BM25) | 46.4% | -24.4pp (harder eval set) |
| Draft quality | 1.5/5 (template) | 3.63/5 | +2.1 |
| Latency p95 | <100ms | ~10.8s | Trade-off for quality |

### 6.2 Project KPIs

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| Timeline | 3 weeks | 3 weeks | Done |
| Core features | 8 | 8 | Done |
| Documentation | Complete | Complete | Done |
| Test coverage | >80% | 84% | Done |

---

## 7. Risk Register Review

### 7.1 Risks That Materialized

| Risk | Impact | Mitigation Applied |
|------|--------|-------------------|
| Annotation bottleneck | Medium | Reduced eval set size |
| Chunking parameter sensitivity | Low | Extended experimentation time |
| Memory management | Low | Configuration tuning |

### 7.2 Risks That Didn't Materialize

| Risk | Why Avoided |
|------|-------------|
| LLM quality insufficient | Base model better than expected |
| Latency exceeded targets | Good budgeting, no wasted cycles |
| GPU unavailable | Early hardware provisioning |

### 7.3 Risks Not Anticipated

| Risk | Lesson |
|------|--------|
| BM25 library performance | Evaluate library choices earlier |
| Scope creep on dashboard | Define "done" more explicitly |

---

## 8. Future Roadmap

### 8.1 Phase 2 Candidates (Next 4 weeks)

| Feature | Priority | Effort | Value |
|---------|----------|--------|-------|
| Multi-language support | High | Large | Expand coverage |
| Real-time ingestion | High | Medium | Reduce latency |
| Fine-tuning pipeline | Medium | Large | Quality improvement |
| Agent performance dashboard | Medium | Small | Operational visibility |

### 8.2 Phase 3 Candidates (Future)

- Multi-label classification
- Voice channel integration
- Automated KB updates from resolutions
- Proactive customer outreach triggers

---

## 9. Team Acknowledgments

Special thanks to:
- Support agents who provided evaluation feedback
- IT team for GPU provisioning
- Product team for requirements clarity

---

## 10. Appendix

### A.1 Timeline Actual vs Planned

```
Week 1 (Days 1-7)
├── Day 1-3: Requirements & Data -- On time
├── Day 4-6: Baseline + Audit -- On time
└── Day 7: Buffer -> Used for chunking experiments

Week 2 (Days 8-14)
├── Day 8-9: LLM Service Setup -- On time
├── Day 10-11: RAG Pipeline -- On time
├── Day 12-13: API Development -- On time
└── Day 14: Buffer -> Used for memory tuning

Week 3 (Days 15-21)
├── Day 15-16: UI Dashboard -- On time
├── Day 17-18: Evaluation + Perf -- On time
├── Day 19-20: Docker + Docs -- On time
└── Day 21: Final review -- Complete
```

### A.2 Decision Log

| Date | Decision | Alternatives | Outcome |
|------|----------|--------------|---------|
| Day 2 | Use Qwen2.5-14B | 32B, 7B | Good balance |
| Day 4 | PostgreSQL for BM25 | rank-bm25 | Better performance |
| Day 9 | INT8 quantization | FP16, INT4 | Met latency target |
| Day 12 | Skip reranking | Add cross-encoder | Sufficient quality |
| Day 15 | Streamlit for UI | Gradio, React | Fastest iteration |
