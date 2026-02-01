"""
Evaluation Harness for LOCALTRIAGE
Comprehensive evaluation of routing, retrieval, and draft quality
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import csv

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    precision_recall_fscore_support
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str = "evaluation"
    run_type: str = "full"  # 'routing', 'retrieval', 'draft', 'full'
    
    # Data paths
    test_data_path: Optional[str] = None
    relevance_data_path: Optional[str] = None
    
    # Evaluation settings
    routing_test_size: int = 1000
    retrieval_test_size: int = 200
    draft_test_size: int = 100
    
    # Retrieval settings
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Draft rubric weights
    draft_rubric_weights: Dict[str, float] = field(default_factory=lambda: {
        'correctness': 0.25,
        'completeness': 0.20,
        'tone': 0.15,
        'actionability': 0.20,
        'citation_quality': 0.20
    })
    
    # Output
    output_dir: str = "reports/evaluations"
    save_predictions: bool = True


@dataclass
class RoutingEvaluationResult:
    """Results from routing evaluation"""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    category_report: Dict[str, Any]
    priority_report: Dict[str, Any]
    category_confusion: List[List[int]]
    priority_confusion: List[List[int]]
    category_labels: List[str]
    priority_labels: List[str]
    num_samples: int
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class RetrievalEvaluationResult:
    """Results from retrieval evaluation"""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    num_queries: int
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class DraftEvaluationResult:
    """Results from draft quality evaluation"""
    overall_score: float
    rubric_scores: Dict[str, float]
    citation_rate: float
    avg_citations_per_draft: float
    hallucination_rate: float
    acceptance_rate: float
    num_samples: int
    latency_p50_ms: float
    latency_p95_ms: float


class RoutingEvaluator:
    """Evaluates ticket routing/classification performance"""
    
    def __init__(self, router):
        self.router = router
    
    def evaluate(
        self,
        subjects: List[str],
        bodies: List[str],
        true_categories: List[str],
        true_priorities: List[str]
    ) -> RoutingEvaluationResult:
        """Run routing evaluation"""
        latencies = []
        pred_categories = []
        pred_priorities = []
        
        for subject, body in zip(subjects, bodies):
            start = time.time()
            pred = self.router.predict(subject, body)
            latencies.append((time.time() - start) * 1000)
            pred_categories.append(pred.category)
            pred_priorities.append(pred.priority)
        
        # Category metrics
        cat_report = classification_report(
            true_categories, pred_categories,
            output_dict=True, zero_division=0
        )
        cat_confusion = confusion_matrix(
            true_categories, pred_categories
        ).tolist()
        cat_labels = list(set(true_categories) | set(pred_categories))
        
        # Priority metrics
        pri_report = classification_report(
            true_priorities, pred_priorities,
            output_dict=True, zero_division=0
        )
        pri_confusion = confusion_matrix(
            true_priorities, pred_priorities
        ).tolist()
        pri_labels = list(set(true_priorities) | set(pred_priorities))
        
        return RoutingEvaluationResult(
            accuracy=cat_report['accuracy'],
            f1_macro=f1_score(true_categories, pred_categories, average='macro', zero_division=0),
            f1_weighted=cat_report['weighted avg']['f1-score'],
            category_report=cat_report,
            priority_report=pri_report,
            category_confusion=cat_confusion,
            priority_confusion=pri_confusion,
            category_labels=sorted(cat_labels),
            priority_labels=sorted(pri_labels),
            num_samples=len(subjects),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95))
        )


class RetrievalEvaluator:
    """Evaluates retrieval performance"""
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / k
    
    def mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        import math
        relevant_set = set(relevant)
        
        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, doc_id in enumerate(retrieved[:k], 1)
            if doc_id in relevant_set
        )
        
        ideal_dcg = sum(
            1.0 / math.log2(i + 1)
            for i in range(1, min(k, len(relevant)) + 1)
        )
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def evaluate(
        self,
        queries: List[str],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalEvaluationResult:
        """Run retrieval evaluation"""
        metrics = {f'recall@{k}': [] for k in k_values}
        metrics.update({f'precision@{k}': [] for k in k_values})
        metrics.update({f'ndcg@{k}': [] for k in k_values})
        metrics['mrr'] = []
        latencies = []
        
        max_k = max(k_values)
        
        for query, relevant in zip(queries, relevant_docs):
            start = time.time()
            results = self.retriever.search_kb(query, top_k=max_k)
            latencies.append((time.time() - start) * 1000)
            
            retrieved_ids = [r.id for r in results]
            
            for k in k_values:
                metrics[f'recall@{k}'].append(self.recall_at_k(retrieved_ids, relevant, k))
                metrics[f'precision@{k}'].append(self.precision_at_k(retrieved_ids, relevant, k))
                metrics[f'ndcg@{k}'].append(self.ndcg_at_k(retrieved_ids, relevant, k))
            
            metrics['mrr'].append(self.mrr(retrieved_ids, relevant))
        
        return RetrievalEvaluationResult(
            recall_at_k={k: np.mean(metrics[f'recall@{k}']) for k in k_values},
            precision_at_k={k: np.mean(metrics[f'precision@{k}']) for k in k_values},
            mrr=np.mean(metrics['mrr']),
            ndcg_at_k={k: np.mean(metrics[f'ndcg@{k}']) for k in k_values},
            num_queries=len(queries),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95))
        )


class DraftEvaluator:
    """Evaluates draft response quality"""
    
    RUBRIC_CRITERIA = {
        'correctness': "Information accuracy relative to KB sources",
        'completeness': "Addresses all aspects of customer's issue",
        'tone': "Professional, empathetic, clear communication",
        'actionability': "Provides clear next steps for customer",
        'citation_quality': "Appropriate citations to sources"
    }
    
    def __init__(self, drafter, rubric_weights: Optional[Dict[str, float]] = None):
        self.drafter = drafter
        self.rubric_weights = rubric_weights or {
            'correctness': 0.25,
            'completeness': 0.20,
            'tone': 0.15,
            'actionability': 0.20,
            'citation_quality': 0.20
        }
    
    def evaluate_single(
        self,
        subject: str,
        body: str,
        ground_truth_response: Optional[str] = None,
        manual_scores: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single draft"""
        start = time.time()
        draft = self.drafter.generate_draft(
            ticket_id=str(uuid.uuid4()),
            subject=subject,
            body=body
        )
        latency_ms = (time.time() - start) * 1000
        
        # Automated checks
        has_citations = len(draft.citations) > 0
        citation_count = len(draft.citations)
        
        # Check for potential hallucination indicators
        hallucination_indicators = [
            "I believe", "I think", "probably", "might be",
            "it seems", "possibly"
        ]
        hallucination_score = sum(
            1 for ind in hallucination_indicators
            if ind.lower() in draft.draft_text.lower()
        )
        
        result = {
            'draft_id': draft.draft_id,
            'draft_text': draft.draft_text,
            'confidence': draft.confidence,
            'confidence_score': draft.confidence_score,
            'has_citations': has_citations,
            'citation_count': citation_count,
            'hallucination_indicators': hallucination_score,
            'latency_ms': latency_ms,
            'generation_time_ms': draft.generation_time_ms,
            'retrieval_time_ms': draft.retrieval_time_ms
        }
        
        if manual_scores:
            result['manual_scores'] = manual_scores
            result['weighted_score'] = sum(
                manual_scores.get(criterion, 3) * weight
                for criterion, weight in self.rubric_weights.items()
            )
        
        return result
    
    def evaluate_batch(
        self,
        tickets: List[Dict[str, Any]],
        include_manual: bool = False
    ) -> DraftEvaluationResult:
        """Evaluate multiple drafts"""
        results = []
        
        for ticket in tickets:
            manual_scores = ticket.get('manual_scores') if include_manual else None
            result = self.evaluate_single(
                subject=ticket['subject'],
                body=ticket['body'],
                manual_scores=manual_scores
            )
            results.append(result)
        
        # Aggregate metrics
        latencies = [r['latency_ms'] for r in results]
        citation_counts = [r['citation_count'] for r in results]
        hallucination_counts = [r['hallucination_indicators'] for r in results]
        
        # Calculate rubric scores if manual scores provided
        if include_manual and results[0].get('manual_scores'):
            rubric_scores = {}
            for criterion in self.rubric_weights.keys():
                scores = [
                    r['manual_scores'].get(criterion, 3) 
                    for r in results 
                    if r.get('manual_scores')
                ]
                rubric_scores[criterion] = np.mean(scores) if scores else 0.0
            
            overall_score = sum(
                rubric_scores[c] * w 
                for c, w in self.rubric_weights.items()
            )
        else:
            rubric_scores = {c: 0.0 for c in self.rubric_weights.keys()}
            overall_score = 0.0
        
        return DraftEvaluationResult(
            overall_score=overall_score,
            rubric_scores=rubric_scores,
            citation_rate=sum(1 for r in results if r['has_citations']) / len(results),
            avg_citations_per_draft=np.mean(citation_counts),
            hallucination_rate=sum(1 for r in results if r['hallucination_indicators'] > 0) / len(results),
            acceptance_rate=0.0,  # Requires feedback data
            num_samples=len(results),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95))
        )


class EvaluationRunner:
    """Orchestrates full evaluation runs"""
    
    def __init__(self, config: EvaluationConfig, db_config: Dict[str, Any]):
        self.config = config
        self.db_config = db_config
        self.results = {}
    
    def load_test_data(self) -> Dict[str, List[Dict]]:
        """Load test data from files or database"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        data = {
            'routing': [],
            'retrieval': [],
            'draft': []
        }
        
        # Load from database
        conn = psycopg2.connect(**self.db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Routing test data
            cur.execute("""
                SELECT subject, body, category, priority
                FROM tickets
                WHERE category IS NOT NULL AND priority IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
            """, (self.config.routing_test_size,))
            data['routing'] = [dict(r) for r in cur.fetchall()]
            
            # Draft test data (same as routing for now)
            cur.execute("""
                SELECT id, subject, body, category, priority
                FROM tickets
                ORDER BY RANDOM()
                LIMIT %s
            """, (self.config.draft_test_size,))
            data['draft'] = [dict(r) for r in cur.fetchall()]
        
        conn.close()
        
        # Load retrieval relevance data if provided
        if self.config.relevance_data_path and os.path.exists(self.config.relevance_data_path):
            with open(self.config.relevance_data_path, 'r') as f:
                data['retrieval'] = json.load(f)
        
        return data
    
    def run_routing_evaluation(self, test_data: List[Dict]) -> RoutingEvaluationResult:
        """Run routing evaluation"""
        from src.triage.baseline_classifier import BaselineRouter
        
        model_path = os.getenv('ROUTER_MODEL_PATH', 'models/baseline_router')
        if os.path.exists(model_path):
            router = BaselineRouter.load(model_path)
        else:
            raise ValueError("Router model not found. Train the model first.")
        
        evaluator = RoutingEvaluator(router)
        
        return evaluator.evaluate(
            subjects=[d['subject'] for d in test_data],
            bodies=[d['body'] for d in test_data],
            true_categories=[d['category'] for d in test_data],
            true_priorities=[d['priority'] for d in test_data]
        )
    
    def run_retrieval_evaluation(self, test_data: List[Dict]) -> Optional[RetrievalEvaluationResult]:
        """Run retrieval evaluation"""
        if not test_data:
            return None
        
        from src.retrieval.baseline_bm25 import BaselineBM25Retriever
        
        retriever = BaselineBM25Retriever(self.db_config)
        retriever.connect()
        
        evaluator = RetrievalEvaluator(retriever)
        
        result = evaluator.evaluate(
            queries=[d['query'] for d in test_data],
            relevant_docs=[d['relevant_docs'] for d in test_data],
            k_values=self.config.retrieval_k_values
        )
        
        retriever.close()
        return result
    
    def run_draft_evaluation(self, test_data: List[Dict]) -> DraftEvaluationResult:
        """Run draft quality evaluation"""
        from src.retrieval.baseline_bm25 import BaselineBM25Retriever
        from src.rag.drafter import LLMClient, RAGDrafter
        
        retriever = BaselineBM25Retriever(self.db_config)
        retriever.connect()
        
        llm_base_url = os.getenv('LLM_BASE_URL', 'http://localhost:8000/v1')
        llm_model = os.getenv('LLM_MODEL', 'Qwen/Qwen2.5-14B-Instruct')
        
        llm = LLMClient(base_url=llm_base_url, model_name=llm_model)
        drafter = RAGDrafter(llm, retriever)
        
        evaluator = DraftEvaluator(
            drafter,
            rubric_weights=self.config.draft_rubric_weights
        )
        
        result = evaluator.evaluate_batch(
            tickets=test_data,
            include_manual='manual_scores' in test_data[0] if test_data else False
        )
        
        retriever.close()
        return result
    
    def run(self) -> Dict[str, Any]:
        """Run full evaluation"""
        print(f"Starting evaluation run: {self.config.run_id}")
        start_time = time.time()
        
        # Load test data
        test_data = self.load_test_data()
        
        results = {
            'run_id': self.config.run_id,
            'run_name': self.config.run_name,
            'run_type': self.config.run_type,
            'timestamp': datetime.now().isoformat(),
            'routing': None,
            'retrieval': None,
            'draft': None
        }
        
        # Run evaluations based on run type
        if self.config.run_type in ['routing', 'full']:
            print(f"Running routing evaluation on {len(test_data['routing'])} samples...")
            results['routing'] = self.run_routing_evaluation(test_data['routing'])
            print(f"  Accuracy: {results['routing'].accuracy:.3f}")
            print(f"  F1 Macro: {results['routing'].f1_macro:.3f}")
        
        if self.config.run_type in ['retrieval', 'full'] and test_data['retrieval']:
            print(f"Running retrieval evaluation on {len(test_data['retrieval'])} queries...")
            results['retrieval'] = self.run_retrieval_evaluation(test_data['retrieval'])
            if results['retrieval']:
                print(f"  Recall@5: {results['retrieval'].recall_at_k[5]:.3f}")
                print(f"  MRR: {results['retrieval'].mrr:.3f}")
        
        if self.config.run_type in ['draft', 'full'] and test_data['draft']:
            print(f"Running draft evaluation on {len(test_data['draft'])} samples...")
            results['draft'] = self.run_draft_evaluation(test_data['draft'])
            print(f"  Citation Rate: {results['draft'].citation_rate:.3f}")
            print(f"  Latency p95: {results['draft'].latency_p95_ms:.0f}ms")
        
        results['total_time_seconds'] = time.time() - start_time
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{self.config.run_id}.json"
        
        # Convert dataclasses to dicts
        serializable = {}
        for key, value in results.items():
            if hasattr(value, '__dataclass_fields__'):
                serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.__dict__.items()
                }
            else:
                serializable[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def compare_evaluations(
    baseline_path: str,
    target_path: str
) -> Dict[str, Any]:
    """Compare two evaluation runs"""
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(target_path, 'r') as f:
        target = json.load(f)
    
    comparison = {
        'baseline_run': baseline.get('run_id'),
        'target_run': target.get('run_id'),
        'improvements': {}
    }
    
    # Compare routing
    if baseline.get('routing') and target.get('routing'):
        comparison['improvements']['routing'] = {
            'accuracy_delta': target['routing']['accuracy'] - baseline['routing']['accuracy'],
            'f1_macro_delta': target['routing']['f1_macro'] - baseline['routing']['f1_macro'],
            'latency_p95_delta': target['routing']['latency_p95_ms'] - baseline['routing']['latency_p95_ms']
        }
    
    # Compare retrieval
    if baseline.get('retrieval') and target.get('retrieval'):
        comparison['improvements']['retrieval'] = {
            'recall@5_delta': target['retrieval']['recall_at_k']['5'] - baseline['retrieval']['recall_at_k']['5'],
            'mrr_delta': target['retrieval']['mrr'] - baseline['retrieval']['mrr']
        }
    
    # Compare drafts
    if baseline.get('draft') and target.get('draft'):
        comparison['improvements']['draft'] = {
            'citation_rate_delta': target['draft']['citation_rate'] - baseline['draft']['citation_rate'],
            'overall_score_delta': target['draft']['overall_score'] - baseline['draft']['overall_score'],
            'latency_p95_delta': target['draft']['latency_p95_ms'] - baseline['draft']['latency_p95_ms']
        }
    
    return comparison


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LOCALTRIAGE Evaluation Harness')
    parser.add_argument('--run-type', choices=['routing', 'retrieval', 'draft', 'full'], default='full')
    parser.add_argument('--run-name', default='evaluation')
    parser.add_argument('--output-dir', default='reports/evaluations')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='localtriage')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-password', default='postgres')
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        run_name=args.run_name,
        run_type=args.run_type,
        output_dir=args.output_dir
    )
    
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    runner = EvaluationRunner(config, db_config)
    results = runner.run()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Total time: {results['total_time_seconds']:.1f}s")
