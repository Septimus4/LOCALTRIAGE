# LOCALTRIAGE - Risk Register

## Risk Assessment Framework

### Probability Scale
- **Low (1):** < 20% chance of occurrence
- **Medium (2):** 20-60% chance of occurrence  
- **High (3):** > 60% chance of occurrence

### Impact Scale
- **Low (1):** Minor inconvenience, workaround available
- **Medium (2):** Significant delay or quality reduction
- **High (3):** Project failure or major rework required

### Risk Score
`Risk Score = Probability × Impact`
- 1-2: Low priority, monitor
- 3-4: Medium priority, active mitigation
- 6-9: High priority, immediate action required

---

## Identified Risks

### R001: LLM Model Size Exceeds Available VRAM

| Attribute | Value |
|-----------|-------|
| **Category** | Technical |
| **Probability** | Medium (2) |
| **Impact** | High (3) |
| **Risk Score** | 6 |
| **Status** | Active |

**Description:**  
The target model (Qwen2.5-14B) requires ~28GB VRAM for full precision inference. Many development machines have limited GPU memory.

**Mitigation Strategies:**
1. Use quantized models (AWQ/GPTQ) to reduce memory by 50-75%
2. Implement llama.cpp CPU fallback for development
3. Use smaller model (7B) for initial development
4. Cloud GPU for production (A100 80GB)

**Contingency:**  
Deploy Qwen2.5-7B if 14B cannot be accommodated; accept slight quality reduction.

**Owner:** ML Engineer

---

### R002: Response Latency Exceeds User Tolerance

| Attribute | Value |
|-----------|-------|
| **Category** | Performance |
| **Probability** | Medium (2) |
| **Impact** | Medium (2) |
| **Risk Score** | 4 |
| **Status** | Active |

**Description:**  
End-to-end drafting may exceed 5-second target due to sequential retrieval + generation steps.

**Mitigation Strategies:**
1. Pre-warm models and maintain persistent connections
2. Implement async retrieval while LLM processes previous request
3. Cache frequent query patterns
4. Use speculative decoding for faster generation
5. Optimize batch sizes for throughput

**Contingency:**  
Return partial response with loading indicator; allow agents to view retrieved docs while draft generates.

**Owner:** Backend Developer

---

### R003: Retrieval Quality Insufficient for Accurate Drafts

| Attribute | Value |
|-----------|-------|
| **Category** | Quality |
| **Probability** | Low (1) |
| **Impact** | High (3) |
| **Risk Score** | 3 |
| **Status** | Monitoring |

**Description:**  
Retrieved KB chunks may not contain relevant information, leading to hallucinated or unhelpful drafts.

**Mitigation Strategies:**
1. Implement hybrid search (dense + sparse) with RRF fusion
2. Add reranking step with cross-encoder
3. Confidence scoring to flag low-quality retrievals
4. Human-in-the-loop for edge cases
5. Continuous KB quality monitoring

**Contingency:**  
Fall back to template responses when retrieval confidence is low; flag for human review.

**Owner:** ML Engineer

---

### R004: Training Data Quality Issues

| Attribute | Value |
|-----------|-------|
| **Category** | Data |
| **Probability** | Medium (2) |
| **Impact** | Medium (2) |
| **Risk Score** | 4 |
| **Status** | Active |

**Description:**  
Historical ticket data may contain noise, incorrect labels, or PII that affects model training and evaluation.

**Mitigation Strategies:**
1. Implement data validation pipeline
2. Manual audit of sample data
3. PII detection and masking
4. Label quality checks with inter-annotator agreement
5. Active learning to identify problematic examples

**Contingency:**  
Use synthetic data for initial development; crowdsource labeling if needed.

**Owner:** Data Engineer

---

### R005: LLM Hallucination in Generated Drafts

| Attribute | Value |
|-----------|-------|
| **Category** | Quality |
| **Probability** | Medium (2) |
| **Impact** | High (3) |
| **Risk Score** | 6 |
| **Status** | Active |

**Description:**  
LLM may generate plausible-sounding but factually incorrect information not grounded in KB.

**Mitigation Strategies:**
1. Strict RAG prompting with explicit grounding instructions
2. Citation requirement for all factual claims
3. Post-generation fact-checking against retrieved content
4. Confidence scoring to flag uncertain generations
5. Agent review before sending to customer

**Contingency:**  
Require human approval for all drafts initially; implement automatic checking pipeline.

**Owner:** ML Engineer

---

### R006: Integration Complexity with Existing Systems

| Attribute | Value |
|-----------|-------|
| **Category** | Integration |
| **Probability** | Low (1) |
| **Impact** | Medium (2) |
| **Risk Score** | 2 |
| **Status** | Monitoring |

**Description:**  
Integration with existing ticketing system, CRM, or knowledge base may be more complex than anticipated.

**Mitigation Strategies:**
1. Design modular API with standard interfaces
2. Support multiple input formats (CSV, JSON, webhook)
3. Document integration requirements early
4. Create adapter layer for different systems

**Contingency:**  
Deploy as standalone system with manual data sync if integration blocked.

**Owner:** Backend Developer

---

### R007: Model Drift Over Time

| Attribute | Value |
|-----------|-------|
| **Category** | Operational |
| **Probability** | Medium (2) |
| **Impact** | Medium (2) |
| **Risk Score** | 4 |
| **Status** | Monitoring |

**Description:**  
Model performance may degrade as ticket patterns, products, or customer language evolves.

**Mitigation Strategies:**
1. Implement continuous evaluation pipeline
2. Monitor key metrics with alerting
3. Scheduled model retraining/fine-tuning
4. A/B testing framework for model updates
5. Feedback loop from agent ratings

**Contingency:**  
Maintain versioned models for quick rollback; increase human review if drift detected.

**Owner:** ML Engineer

---

### R008: Security and Privacy Concerns

| Attribute | Value |
|-----------|-------|
| **Category** | Security |
| **Probability** | Low (1) |
| **Impact** | High (3) |
| **Risk Score** | 3 |
| **Status** | Active |

**Description:**  
System handles sensitive customer data that must be protected. Local LLM mitigates cloud risks but introduces new considerations.

**Mitigation Strategies:**
1. All processing local (no data to external APIs)
2. Database encryption at rest
3. Role-based access control
4. Audit logging for all data access
5. Regular security reviews

**Contingency:**  
Isolate system on dedicated network; implement data anonymization layer.

**Owner:** Security Lead

---

### R009: Resource Constraints Delay Delivery

| Attribute | Value |
|-----------|-------|
| **Category** | Project |
| **Probability** | Medium (2) |
| **Impact** | Medium (2) |
| **Risk Score** | 4 |
| **Status** | Active |

**Description:**  
Limited team size or competing priorities may delay project milestones.

**Mitigation Strategies:**
1. Clear prioritization of features (MoSCoW)
2. Buffer time in schedule (15-20%)
3. Identify critical path and focus resources
4. Reduce scope rather than quality
5. Regular stakeholder communication

**Contingency:**  
Deliver MVP with core features; defer advanced analytics to Phase 2.

**Owner:** Project Manager

---

## Risk Matrix

```
Impact
   ^
 3 |  R003  |  R001,R005  |
   |        |             |
 2 |  R006  |  R002,R004  |  
   |        |  R007,R009  |
 1 |        |             |
   +--------+-------------+-->
       1          2        3  Probability
```

---

## Risk Review Schedule

| Review Type | Frequency | Participants |
|-------------|-----------|--------------|
| Quick scan | Daily standup | Dev team |
| Full review | Weekly | Project leads |
| Deep dive | Bi-weekly | All stakeholders |

---

## Change Log

| Date | Risk ID | Change | Author |
|------|---------|--------|--------|
| [TBD] | - | Initial risk register created | [Name] |

---

*Last Updated: [Date]*
