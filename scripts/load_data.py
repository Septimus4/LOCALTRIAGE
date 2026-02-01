#!/usr/bin/env python3
"""
Data loading and seeding script for the Local Triage system.
This script prepares training data, builds indexes, and sets up the system for use.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load JSON file and return contents."""
    try:
        with open(file_path) as f:
            data = json.load(f)
            logger.info(f"Loaded {file_path.name}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None


def validate_data() -> bool:
    """Validate all data files exist and have correct structure."""
    logger.info("Validating data files...")
    
    required_files = [
        DATA_DIR / "raw" / "sample_tickets.json",
        DATA_DIR / "raw" / "knowledge_base_articles.json",
        DATA_DIR / "processed" / "classifier_training_data.json",
        DATA_DIR / "processed" / "evaluation_dataset.json",
    ]
    
    all_valid = True
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Missing required file: {file_path}")
            all_valid = False
        else:
            data = load_json_file(file_path)
            if data is None:
                all_valid = False
    
    if all_valid:
        logger.info("✓ All data files validated successfully")
    else:
        logger.error("✗ Data validation failed")
    
    return all_valid


def train_classifier() -> bool:
    """Train the classification model using training data."""
    logger.info("Training classifier...")
    
    try:
        from src.triage.baseline_classifier import BaselineTriageClassifier
    except ImportError as e:
        logger.error(f"Failed to import classifier: {e}")
        return False
    
    # Load training data
    training_data = load_json_file(
        DATA_DIR / "processed" / "classifier_training_data.json"
    )
    
    if not training_data:
        return False
    
    samples = training_data.get("training_samples", [])
    if not samples:
        logger.error("No training samples found")
        return False
    
    texts = [s["text"] for s in samples]
    labels = [s["category"] for s in samples]
    
    logger.info(f"Training with {len(texts)} samples across {len(set(labels))} categories")
    
    try:
        classifier = BaselineTriageClassifier()
        classifier.train(texts, labels)
        
        # Save model
        model_path = DATA_DIR / "models" / "classifier_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(str(model_path))
        
        logger.info(f"✓ Classifier trained and saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to train classifier: {e}")
        return False


def build_vector_index() -> bool:
    """Build vector search index from knowledge base articles."""
    logger.info("Building vector search index...")
    
    try:
        from src.retrieval.vector_store import VectorStore
    except ImportError as e:
        logger.error(f"Failed to import VectorStore: {e}")
        return False
    
    # Load KB articles
    articles = load_json_file(
        DATA_DIR / "raw" / "knowledge_base_articles.json"
    )
    
    if not articles:
        return False
    
    logger.info(f"Indexing {len(articles)} knowledge base articles")
    
    try:
        vector_store = VectorStore()
        
        # Prepare documents for indexing
        documents = []
        for article in articles:
            doc = {
                "id": article["article_id"],
                "content": f"{article['title']}\n\n{article['content']}",
                "metadata": {
                    "category": article.get("category"),
                    "tags": article.get("tags", []),
                    "title": article["title"]
                }
            }
            documents.append(doc)
        
        # Add documents to index
        vector_store.add_documents(documents)
        
        # Save index
        index_path = DATA_DIR / "indices" / "kb_index"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save(str(index_path))
        
        logger.info(f"✓ Vector index built and saved to {index_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to build vector index: {e}")
        return False


def evaluate_models() -> bool:
    """Run evaluation on trained models using evaluation dataset."""
    logger.info("Evaluating models...")
    
    try:
        from src.triage.baseline_classifier import BaselineTriageClassifier
    except ImportError as e:
        logger.error(f"Failed to import classifier: {e}")
        return False
    
    # Load evaluation data
    eval_data = load_json_file(
        DATA_DIR / "processed" / "evaluation_dataset.json"
    )
    
    if not eval_data:
        return False
    
    test_samples = eval_data.get("test_samples", [])
    
    # Load trained model
    model_path = DATA_DIR / "models" / "classifier_model.pkl"
    if not model_path.exists():
        logger.error("No trained model found. Run training first.")
        return False
    
    try:
        classifier = BaselineTriageClassifier()
        classifier.load(str(model_path))
        
        # Evaluate
        correct = 0
        total = len(test_samples)
        results_by_category = {}
        
        for sample in test_samples:
            prediction = classifier.predict(sample["text"])
            expected = sample["expected_category"]
            
            is_correct = prediction == expected
            if is_correct:
                correct += 1
            
            # Track per-category accuracy
            if expected not in results_by_category:
                results_by_category[expected] = {"correct": 0, "total": 0}
            results_by_category[expected]["total"] += 1
            if is_correct:
                results_by_category[expected]["correct"] += 1
        
        overall_accuracy = correct / total if total > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATION RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
        logger.info(f"\nPer-Category Accuracy:")
        for category, results in sorted(results_by_category.items()):
            cat_accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
            logger.info(f"  {category}: {cat_accuracy:.2%} ({results['correct']}/{results['total']})")
        logger.info(f"{'='*50}\n")
        
        # Save evaluation results
        results = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "per_category": results_by_category
        }
        
        results_path = DATA_DIR / "processed" / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Evaluation results saved to {results_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to evaluate models: {e}")
        return False


def seed_database() -> bool:
    """Seed the database with sample tickets (if database is configured)."""
    logger.info("Seeding database...")
    
    # Load sample tickets
    tickets = load_json_file(DATA_DIR / "raw" / "sample_tickets.json")
    
    if not tickets:
        return False
    
    logger.info(f"Would seed {len(tickets)} tickets to database")
    logger.info("(Database seeding requires running database - skipping for now)")
    
    return True


def show_data_summary():
    """Show summary of available data."""
    logger.info("\n" + "="*50)
    logger.info("DATA SUMMARY")
    logger.info("="*50)
    
    # Sample tickets
    tickets = load_json_file(DATA_DIR / "raw" / "sample_tickets.json")
    if tickets:
        categories = set(t["category"] for t in tickets)
        logger.info(f"\nSample Tickets: {len(tickets)}")
        logger.info(f"  Categories: {', '.join(sorted(categories))}")
    
    # KB articles
    articles = load_json_file(DATA_DIR / "raw" / "knowledge_base_articles.json")
    if articles:
        categories = set(a["category"] for a in articles)
        logger.info(f"\nKB Articles: {len(articles)}")
        logger.info(f"  Categories: {', '.join(sorted(categories))}")
    
    # Training data
    training = load_json_file(DATA_DIR / "processed" / "classifier_training_data.json")
    if training:
        samples = training.get("training_samples", [])
        logger.info(f"\nTraining Samples: {len(samples)}")
        logger.info(f"  Categories: {', '.join(training.get('categories', []))}")
    
    # Evaluation data
    eval_data = load_json_file(DATA_DIR / "processed" / "evaluation_dataset.json")
    if eval_data:
        samples = eval_data.get("test_samples", [])
        logger.info(f"\nEvaluation Samples: {len(samples)}")
    
    # Check for trained models
    model_path = DATA_DIR / "models" / "classifier_model.pkl"
    if model_path.exists():
        logger.info(f"\n✓ Trained classifier model exists")
    else:
        logger.info(f"\n✗ No trained classifier model")
    
    # Check for indices
    index_path = DATA_DIR / "indices" / "kb_index"
    if index_path.exists():
        logger.info(f"✓ Vector search index exists")
    else:
        logger.info(f"✗ No vector search index")
    
    logger.info("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Data loading and seeding script for Local Triage system"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data files"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the classifier model"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build vector search index"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate trained models"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed database with sample data"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show data summary"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps (validate, train, index, evaluate)"
    )
    
    args = parser.parse_args()
    
    # If no args provided, show summary
    if not any(vars(args).values()):
        show_data_summary()
        parser.print_help()
        return
    
    if args.summary:
        show_data_summary()
        return
    
    success = True
    
    if args.all or args.validate:
        success = validate_data() and success
    
    if args.all or args.train:
        success = train_classifier() and success
    
    if args.all or args.index:
        success = build_vector_index() and success
    
    if args.all or args.evaluate:
        success = evaluate_models() and success
    
    if args.seed:
        success = seed_database() and success
    
    if success:
        logger.info("✓ All operations completed successfully")
    else:
        logger.error("✗ Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
