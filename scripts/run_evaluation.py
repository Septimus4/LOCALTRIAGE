#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for LOCALTRIAGE
Runs all evaluations and generates metrics report
"""
import os
import sys
import json
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
        model_name=os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
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
    """Evaluate draft quality and latency."""
    logger.info("\n" + "=" * 60)
    logger.info("DRAFT QUALITY EVALUATION")
    logger.info("=" * 60)
    
    import requests
    
    test_tickets = [
        {
            "subject": "Cannot login to my account",
            "body": "I've tried resetting my password multiple times but I still can't login. The reset emails aren't arriving.",
            "expected_quality_indicators": ["password", "reset", "email"]
        },
        {
            "subject": "Charged twice for subscription",
            "body": "I was charged $99.99 twice this month for my annual subscription. Please refund the duplicate.",
            "expected_quality_indicators": ["refund", "charge", "billing"]
        },
        {
            "subject": "App crashes on export",
            "body": "Every time I try to export a report to PDF, the application crashes. This started after the last update.",
            "expected_quality_indicators": ["export", "PDF", "crash"]
        },
        {
            "subject": "Request: Dark mode",
            "body": "Would love to see dark mode added to the application. It would help reduce eye strain when working late.",
            "expected_quality_indicators": ["dark mode", "feature", "feedback"]
        },
        {
            "subject": "Account security concern",
            "body": "I noticed some suspicious login attempts on my account from different locations. How can I secure my account?",
            "expected_quality_indicators": ["security", "login", "two-factor"]
        },
    ]
    
    quality_scores = []
    latencies = []
    citation_counts = []
    
    for ticket in test_tickets:
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:8080/draft",
                json={
                    "subject": ticket["subject"],
                    "body": ticket["body"],
                    "use_llm": True
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            draft_text = result.get("draft_text", "").lower()
            citations = result.get("citations", [])
            citation_counts.append(len(citations))
            
            # Score based on:
            # 1. Contains expected keywords (40%)
            # 2. Has citations (30%)
            # 3. Reasonable length (15%)
            # 4. Has follow-up questions (15%)
            
            keyword_score = sum(
                1 for kw in ticket["expected_quality_indicators"]
                if kw.lower() in draft_text
            ) / len(ticket["expected_quality_indicators"])
            
            citation_score = min(len(citations) / 3, 1.0)  # Expect ~3 citations
            
            length_score = 1.0 if 100 < len(draft_text) < 2000 else 0.5
            
            followup_score = 1.0 if result.get("follow_up_questions") else 0.0
            
            quality = (
                keyword_score * 0.40 +
                citation_score * 0.30 +
                length_score * 0.15 +
                followup_score * 0.15
            ) * 5  # Scale to 1-5
            
            quality_scores.append(quality)
            logger.info(f"Ticket: '{ticket['subject'][:30]}...' | Quality: {quality:.2f}/5 | Latency: {latency_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"Error evaluating ticket '{ticket['subject']}': {e}")
            latencies.append(120000)  # Timeout
            quality_scores.append(1.0)  # Minimum score
    
    avg_quality = statistics.mean(quality_scores) if quality_scores else 0
    avg_citations = statistics.mean(citation_counts) if citation_counts else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    logger.info(f"\nAverage Draft Quality: {avg_quality:.2f}/5")
    logger.info(f"Average Citations: {avg_citations:.1f}")
    logger.info(f"P95 Latency: {p95_latency:.0f}ms")
    
    return {
        "avg_quality": avg_quality,
        "avg_citations": avg_citations,
        "p95_latency_ms": p95_latency,
        "num_samples": len(test_tickets),
        "individual_scores": quality_scores
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
