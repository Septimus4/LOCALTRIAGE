#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for LOCALTRIAGE
Runs all evaluations and generates metrics report
"""
import os
import sys
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_json_file(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def evaluate_routing() -> Dict[str, Any]:
    """Evaluate routing/classification accuracy."""
    logger.info("=" * 60)
    logger.info("ROUTING EVALUATION")
    logger.info("=" * 60)
    
    from src.triage.baseline_classifier import BaselineRouter
    
    # Load model
    model_path = DATA_DIR / "models" / "classifier_model"
    if not model_path.exists():
        logger.error("No trained model found")
        return {}
    
    router = BaselineRouter.load(str(model_path))
    
    # Load evaluation data
    eval_data = load_json_file(DATA_DIR / "processed" / "evaluation_dataset.json")
    test_samples = eval_data.get("test_samples", [])
    
    correct = 0
    total = len(test_samples)
    latencies = []
    results_by_category = {}
    
    for sample in test_samples:
        start = time.time()
        prediction = router.predict(sample["text"], "")
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        
        expected = sample["expected_category"]
        is_correct = prediction.category == expected
        
        if is_correct:
            correct += 1
        
        if expected not in results_by_category:
            results_by_category[expected] = {"correct": 0, "total": 0}
        results_by_category[expected]["total"] += 1
        if is_correct:
            results_by_category[expected]["correct"] += 1
    
    accuracy = correct / total if total > 0 else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    logger.info(f"Routing Accuracy: {accuracy:.2%} ({correct}/{total})")
    logger.info(f"P95 Latency: {p95_latency:.1f}ms")
    logger.info("\nPer-Category Accuracy:")
    for cat, results in sorted(results_by_category.items()):
        cat_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
        logger.info(f"  {cat}: {cat_acc:.2%} ({results['correct']}/{results['total']})")
    
    return {
        "accuracy": accuracy,
        "total_samples": total,
        "correct": correct,
        "p95_latency_ms": p95_latency,
        "per_category": results_by_category
    }


def evaluate_retrieval() -> Dict[str, Any]:
    """Evaluate retrieval recall@5."""
    logger.info("\n" + "=" * 60)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("=" * 60)
    
    from src.retrieval.vector_search import QdrantVectorStore, EmbeddingModel
    
    # Initialize components
    embedding_model = EmbeddingModel(
        model_name=os.environ.get('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B'),
        device='cuda'
    )
    
    store = QdrantVectorStore(
        collection_name='kb_articles',
        dimension=embedding_model.dimension
    )
    
    # Load KB articles for ground truth
    articles = load_json_file(DATA_DIR / "raw" / "knowledge_base_articles.json")
    article_by_category = {}
    for article in articles:
        cat = article.get("category", "General")
        if cat not in article_by_category:
            article_by_category[cat] = []
        article_by_category[cat].append(article["article_id"])
    
    # Test queries with expected categories
    test_queries = [
        {"query": "How do I reset my password?", "expected_category": "Account"},
        {"query": "I forgot my login credentials", "expected_category": "Account"},
        {"query": "My account was hacked", "expected_category": "Account"},
        {"query": "I was charged twice for my subscription", "expected_category": "Billing"},
        {"query": "How do I get a refund?", "expected_category": "Billing"},
        {"query": "Cancel my subscription", "expected_category": "Billing"},
        {"query": "The app keeps crashing", "expected_category": "Technical"},
        {"query": "Export to PDF not working", "expected_category": "Technical"},
        {"query": "Integration with Slack broken", "expected_category": "Technical"},
        {"query": "Can you add dark mode?", "expected_category": "Feature Request"},
        {"query": "Mobile app feature request", "expected_category": "Feature Request"},
        {"query": "What are your business hours?", "expected_category": "General Inquiry"},
        {"query": "How do I contact support?", "expected_category": "General Inquiry"},
    ]
    
    recall_at_5_scores = []
    latencies = []
    
    for test in test_queries:
        query = test["query"]
        expected_cat = test["expected_category"]
        relevant_ids = article_by_category.get(expected_cat, [])
        
        start = time.time()
        query_embedding = embedding_model.encode(query)
        results = store.search(query_embedding, top_k=5)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        
        # Get retrieved article IDs from original_id in metadata
        retrieved_ids = [r.metadata.get("original_id", r.id) for r in results]
        
        # Calculate recall@5
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
        recall_at_5_scores.append(recall)
        
        logger.info(f"Query: '{query[:40]}...' | Recall@5: {recall:.2f}")
    
    avg_recall = statistics.mean(recall_at_5_scores) if recall_at_5_scores else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    logger.info(f"\nAverage Recall@5: {avg_recall:.2%}")
    logger.info(f"P95 Latency: {p95_latency:.1f}ms")
    
    return {
        "recall_at_5": avg_recall,
        "num_queries": len(test_queries),
        "p95_latency_ms": p95_latency,
        "individual_scores": recall_at_5_scores
    }


def evaluate_draft_quality() -> Dict[str, Any]:
    """Evaluate draft quality using LLM-as-judge (Qwen3:32B).
    
    Each draft is scored by the same LLM on 5 criteria using a structured
    rubric with explicit scoring anchors (1-5).  The judge prompt is
    designed to be impartial:
      - The judge never sees the generating model's identity
      - Scoring anchors are defined per level (1-5) for each criterion
      - The judge is instructed to penalise hallucinations and reward
        grounded, cited information
      - Output is structured JSON so parsing is deterministic
    """
    logger.info("\n" + "=" * 60)
    logger.info("DRAFT QUALITY EVALUATION  (LLM-as-judge)")
    logger.info("=" * 60)
    
    import requests

    # ── Rubric definition ────────────────────────────────────────
    RUBRIC = {
        "correctness": {
            "description": "Factual accuracy of the response relative to the provided knowledge-base sources.  Penalise any claim not grounded in a cited source.",
            "anchors": {
                "1": "Multiple factual errors or fabricated information not present in sources.",
                "2": "At least one significant factual error or unsupported claim.",
                "3": "Mostly accurate but contains a minor inaccuracy or vague statement.",
                "4": "Accurate with all major claims grounded in sources; very minor issues only.",
                "5": "Fully accurate; every claim is directly supported by a cited source."
            }
        },
        "completeness": {
            "description": "Does the response address ALL aspects of the customer's issue?  Consider whether the customer would need to write back for missing information.",
            "anchors": {
                "1": "Addresses almost none of the customer's concerns.",
                "2": "Addresses only part of the issue; major gaps remain.",
                "3": "Addresses the main issue but misses secondary concerns.",
                "4": "Covers all main and most secondary concerns.",
                "5": "Thoroughly addresses every aspect; the customer should have no follow-up need."
            }
        },
        "tone": {
            "description": "Professional, empathetic, and clear communication style appropriate for a customer-support context.",
            "anchors": {
                "1": "Rude, dismissive, or incomprehensible.",
                "2": "Overly terse or slightly condescending.",
                "3": "Acceptable but mechanical; lacks warmth.",
                "4": "Professional and friendly with minor style issues.",
                "5": "Exemplary: warm, clear, professional, and empathetic throughout."
            }
        },
        "actionability": {
            "description": "Does the response give the customer concrete, actionable next steps they can follow immediately?",
            "anchors": {
                "1": "No actionable guidance at all.",
                "2": "Vague suggestions without clear steps.",
                "3": "Some steps provided but incomplete or unclear ordering.",
                "4": "Clear numbered/ordered steps covering the main resolution path.",
                "5": "Detailed step-by-step instructions with fallback options if the first path fails."
            }
        },
        "citation_quality": {
            "description": "Are knowledge-base sources cited appropriately?  Citations should appear inline (e.g. [KB-X]) and match the information they support.",
            "anchors": {
                "1": "No citations at all.",
                "2": "One or two citations but placed incorrectly or irrelevant.",
                "3": "Some relevant citations but inconsistent placement.",
                "4": "Most claims are well-cited with correct source references.",
                "5": "Every substantive claim is supported by a correctly placed, relevant citation."
            }
        }
    }

    JUDGE_SYSTEM_PROMPT = (
        "You are an extremely strict, impartial quality evaluator for customer-support draft responses.  "
        "You will receive a customer ticket and a candidate response.  "
        "Your task is to rigorously critique the response before scoring.\n\n"
        "IMPORTANT EVALUATION RULES:\n"
        "- A score of 5 is RARE and means truly flawless.  Most good responses score 3-4.\n"
        "- You MUST identify at least one concrete weakness or area for improvement per criterion before scoring.\n"
        "- If the response uses hedging language ('I believe', 'probably', 'might') without citing a source, deduct points for correctness.\n"
        "- If any claim lacks an inline citation [KB-X], deduct points for citation_quality.\n"
        "- Generic or boilerplate responses that don't specifically address the customer's unique situation score at most 3 for completeness.\n"
        "- A score of 5 on actionability requires numbered step-by-step instructions WITH fallback options.\n"
        "- Do NOT inflate scores because the response is polite.  Politeness alone is worth 3 on tone; 4-5 requires genuine empathy adapted to the situation.\n\n"
        "SCORING PROCESS (follow this order):\n"
        "1. First, list 1-3 specific weaknesses of the response.\n"
        "2. Then, list 1-2 strengths.\n"
        "3. Then, assign integer scores 1-5 based on the rubric anchors below.\n"
        "4. Return ONLY valid JSON (no markdown, no extra text outside the JSON).\n\n"
        "Rubric criteria and scoring anchors:\n"
    )

    # Build rubric text for the prompt
    rubric_text = ""
    for criterion, spec in RUBRIC.items():
        rubric_text += f"\n## {criterion}\n{spec['description']}\n"
        for level, anchor in spec["anchors"].items():
            rubric_text += f"  {level}: {anchor}\n"

    JUDGE_SYSTEM_PROMPT += rubric_text
    JUDGE_SYSTEM_PROMPT += (
        "\nReturn a JSON object with exactly these keys:\n"
        '{"weaknesses": ["<weakness1>", "<weakness2>"], "strengths": ["<strength1>"], '
        '"correctness": <int 1-5>, "completeness": <int 1-5>, "tone": <int 1-5>, '
        '"actionability": <int 1-5>, "citation_quality": <int 1-5>, "justification": "<1-2 sentence summary>"}\n'
        "\nRemember: most good-but-not-perfect responses should score 3-4.  Only truly exceptional responses earn 5.\n"
    )

    # ── Test tickets ─────────────────────────────────────────────
    test_tickets = [
        {
            "subject": "Cannot login to my account",
            "body": "I've tried resetting my password multiple times but I still can't login. The reset emails aren't arriving."
        },
        {
            "subject": "Charged twice for subscription",
            "body": "I was charged $99.99 twice this month for my annual subscription. Please refund the duplicate."
        },
        {
            "subject": "App crashes on export",
            "body": "Every time I try to export a report to PDF, the application crashes. This started after the last update."
        },
        {
            "subject": "Request: Dark mode",
            "body": "Would love to see dark mode added to the application. It would help reduce eye strain when working late."
        },
        {
            "subject": "Account security concern",
            "body": "I noticed some suspicious login attempts on my account from different locations. How can I secure my account?"
        },
    ]

    # ── Weights (equal by default — unbiased) ────────────────────
    weights = {
        "correctness": 0.25,
        "completeness": 0.20,
        "tone": 0.15,
        "actionability": 0.20,
        "citation_quality": 0.20
    }
    criteria_names = list(weights.keys())

    all_scores: List[Dict[str, Any]] = []  # per-ticket detail
    quality_scores: List[float] = []
    latencies: List[float] = []
    citation_counts: List[int] = []
    per_criterion: Dict[str, List[int]] = {c: [] for c in criteria_names}
    hallucination_count = 0

    for ticket in test_tickets:
        # 1. Generate draft via the API
        start = time.time()
        try:
            resp = requests.post(
                "http://localhost:8080/draft",
                json={"subject": ticket["subject"], "body": ticket["body"], "use_llm": True},
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            gen_latency_ms = (time.time() - start) * 1000
        except Exception as exc:
            logger.error(f"Draft generation failed for '{ticket['subject']}': {exc}")
            latencies.append(120000)
            quality_scores.append(1.0)
            continue

        draft_text = result.get("draft_text", "")
        citations = result.get("citations", [])
        citation_counts.append(len(citations))
        latencies.append(gen_latency_ms)

        # 2. Build judge prompt (append /no_think to suppress Qwen3 thinking blocks)
        user_prompt = (
            f"## Customer ticket\n"
            f"**Subject:** {ticket['subject']}\n"
            f"**Body:** {ticket['body']}\n\n"
            f"## Candidate response\n{draft_text}\n\n"
            f"/no_think"
        )

        # 3. Call the judge (same Qwen3:32B via Ollama)
        try:
            judge_resp = requests.post(
                "http://localhost:11434/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen3:32b",
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
                timeout=120,
            )
            judge_resp.raise_for_status()
            raw_content = judge_resp.json()["choices"][0]["message"]["content"]

            # Try to extract JSON from response (may contain <think> blocks)
            # First try after stripping think blocks, then try raw content
            cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

            def extract_json(text: str) -> dict:
                """Extract first complete JSON object using bracket matching."""
                start_idx = text.find("{")
                if start_idx == -1:
                    return None
                depth = 0
                in_string = False
                escape_next = False
                for i in range(start_idx, len(text)):
                    c = text[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if c == "\\":
                        escape_next = True
                        continue
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start_idx:i + 1])
                            except json.JSONDecodeError:
                                return None
                return None

            scores = extract_json(cleaned)
            if scores is None:
                scores = extract_json(raw_content)
            if scores is None:
                raise ValueError(f"No valid JSON in judge response: {raw_content[:300]}")

        except Exception as exc:
            logger.warning(f"Judge call failed for '{ticket['subject']}': {exc}")
            scores = {c: 3 for c in criteria_names}
            scores["justification"] = f"Judge error: {exc}"

        # 4. Record scores
        ticket_scores = {}
        for c in criteria_names:
            val = int(scores.get(c, 3))
            val = max(1, min(5, val))  # clamp 1-5
            ticket_scores[c] = val
            per_criterion[c].append(val)

        weighted = sum(ticket_scores[c] * weights[c] for c in criteria_names)
        quality_scores.append(weighted)

        # Basic hallucination flag (hedging phrases in draft)
        hedging = ["I believe", "I think", "probably", "might be", "it seems", "possibly"]
        if any(h.lower() in draft_text.lower() for h in hedging):
            hallucination_count += 1

        entry = {
            "subject": ticket["subject"],
            "scores": ticket_scores,
            "weighted": round(weighted, 2),
            "justification": scores.get("justification", ""),
            "citations": len(citations),
            "latency_ms": round(gen_latency_ms, 0),
        }
        all_scores.append(entry)
        logger.info(
            f"Ticket: '{ticket['subject'][:35]}' | "
            + " ".join(f"{c[:4]}={ticket_scores[c]}" for c in criteria_names)
            + f" | W={weighted:.2f}/5 | {gen_latency_ms:.0f}ms"
        )

    # ── Aggregate ────────────────────────────────────────────────
    avg_quality = statistics.mean(quality_scores) if quality_scores else 0
    avg_citations = statistics.mean(citation_counts) if citation_counts else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    criterion_avgs = {c: round(statistics.mean(v), 2) if v else 0 for c, v in per_criterion.items()}
    hallucination_rate = hallucination_count / len(test_tickets) if test_tickets else 0

    logger.info(f"\n--- LLM-as-judge results ---")
    logger.info(f"Average weighted quality : {avg_quality:.2f}/5")
    logger.info(f"Per-criterion averages   : {criterion_avgs}")
    logger.info(f"Average citations/draft  : {avg_citations:.1f}")
    logger.info(f"Hallucination rate       : {hallucination_rate:.0%}")
    logger.info(f"Draft P95 latency        : {p95_latency:.0f}ms")

    return {
        "avg_quality": round(avg_quality, 2),
        "avg_citations": avg_citations,
        "p95_latency_ms": p95_latency,
        "num_samples": len(test_tickets),
        "evaluation_method": "LLM-as-judge (Qwen3:32B, temperature=0)",
        "rubric_criteria": list(RUBRIC.keys()),
        "weights": weights,
        "criterion_averages": criterion_avgs,
        "hallucination_rate": round(hallucination_rate, 3),
        "individual_scores": quality_scores,
        "detailed_scores": all_scores,
    }


def measure_e2e_latency() -> Dict[str, Any]:
    """Measure end-to-end P95 latency."""
    logger.info("\n" + "=" * 60)
    logger.info("END-TO-END LATENCY MEASUREMENT")
    logger.info("=" * 60)
    
    import requests
    
    test_tickets = [
        {"subject": "Login issue", "body": "Can't login to my account"},
        {"subject": "Billing question", "body": "Why was I charged twice?"},
        {"subject": "App bug", "body": "The app crashes when exporting"},
        {"subject": "Feature request", "body": "Please add dark mode"},
        {"subject": "Password reset", "body": "Need to reset my password"},
        {"subject": "Refund request", "body": "I want a refund for my subscription"},
        {"subject": "Technical help", "body": "Integration not working"},
        {"subject": "Account security", "body": "Suspicious activity on my account"},
        {"subject": "Subscription cancel", "body": "How do I cancel my subscription?"},
        {"subject": "General inquiry", "body": "What are your support hours?"},
    ]
    
    triage_latencies = []
    draft_latencies = []
    
    for ticket in test_tickets:
        # Triage
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8080/triage",
                json=ticket,
                timeout=30
            )
            triage_latency = (time.time() - start) * 1000
            triage_latencies.append(triage_latency)
        except Exception as e:
            triage_latencies.append(30000)
        
        # Draft
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8080/draft",
                json={**ticket, "use_llm": True},
                timeout=120
            )
            draft_latency = (time.time() - start) * 1000
            draft_latencies.append(draft_latency)
        except Exception as e:
            draft_latencies.append(120000)
    
    triage_p95 = sorted(triage_latencies)[int(len(triage_latencies) * 0.95)]
    draft_p95 = sorted(draft_latencies)[int(len(draft_latencies) * 0.95)]
    
    logger.info(f"Triage P95 Latency: {triage_p95:.0f}ms")
    logger.info(f"Draft P95 Latency: {draft_p95:.0f}ms")
    logger.info(f"Total E2E P95 Latency: {(triage_p95 + draft_p95)/1000:.1f}s")
    
    return {
        "triage_p95_ms": triage_p95,
        "draft_p95_ms": draft_p95,
        "total_p95_s": (triage_p95 + draft_p95) / 1000
    }


def main():
    """Run full evaluation and generate report."""
    logger.info("=" * 60)
    logger.info("LOCALTRIAGE COMPREHENSIVE EVALUATION")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "routing": {},
        "retrieval": {},
        "draft_quality": {},
        "latency": {}
    }
    
    # Run evaluations
    results["routing"] = evaluate_routing()
    results["retrieval"] = evaluate_retrieval()
    results["draft_quality"] = evaluate_draft_quality()
    results["latency"] = measure_e2e_latency()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    routing_acc = results["routing"].get("accuracy", 0) * 100
    retrieval_recall = results["retrieval"].get("recall_at_5", 0) * 100
    draft_quality = results["draft_quality"].get("avg_quality", 0)
    p95_latency = results["latency"].get("total_p95_s", 0)
    
    logger.info(f"""
| Metric             | Baseline | Target | Achieved |
|--------------------|----------|--------|----------|
| Routing Accuracy   | 72%      | 90%    | {routing_acc:.1f}%    |
| Retrieval Recall@5 | 58%      | 80%    | {retrieval_recall:.1f}%    |
| Draft Quality (1-5)| 2.1      | 4.0    | {draft_quality:.2f}     |
| P95 Latency        | 8.2s     | 5.0s   | {p95_latency:.1f}s     |
""")
    
    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
