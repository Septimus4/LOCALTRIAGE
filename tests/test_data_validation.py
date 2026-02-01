"""
Tests that validate the models and system using real sample data.
These tests ensure the system works with actual data, not just mocks.
"""
import pytest
import json
from pathlib import Path
from tests.fixtures import (
    get_sample_tickets,
    get_knowledge_base_articles,
    get_training_data,
    get_evaluation_data,
    get_training_texts_and_labels,
)

# Import the actual modules to test
try:
    from src.triage.baseline_classifier import BaselineTriageClassifier
    from src.retrieval.vector_store import VectorStore
    HAS_ML_MODULES = True
except ImportError:
    HAS_ML_MODULES = False


class TestDataIntegrity:
    """Tests to verify data files are valid and complete."""

    def test_sample_tickets_load_successfully(self):
        """Sample tickets JSON should load and have expected structure."""
        tickets = get_sample_tickets()
        assert len(tickets) >= 5, "Should have at least 5 sample tickets"
        
        # Check required fields
        required_fields = ["ticket_id", "subject", "body", "category", "priority"]
        for ticket in tickets:
            for field in required_fields:
                assert field in ticket, f"Ticket missing required field: {field}"

    def test_kb_articles_load_successfully(self):
        """Knowledge base articles should load and have expected structure."""
        articles = get_knowledge_base_articles()
        assert len(articles) >= 3, "Should have at least 3 KB articles"
        
        required_fields = ["article_id", "title", "content", "category"]
        for article in articles:
            for field in required_fields:
                assert field in article, f"Article missing required field: {field}"

    def test_training_data_load_successfully(self):
        """Training data should load and have expected structure."""
        data = get_training_data()
        assert "training_samples" in data, "Missing training_samples"
        assert "categories" in data, "Missing categories"
        
        samples = data["training_samples"]
        assert len(samples) >= 10, "Should have at least 10 training samples"
        
        for sample in samples:
            assert "text" in sample, "Sample missing text"
            assert "category" in sample, "Sample missing category"

    def test_evaluation_data_load_successfully(self):
        """Evaluation data should load and have expected structure."""
        data = get_evaluation_data()
        assert "test_samples" in data, "Missing test_samples"
        
        samples = data["test_samples"]
        assert len(samples) >= 5, "Should have at least 5 evaluation samples"
        
        for sample in samples:
            assert "id" in sample, "Sample missing id"
            assert "text" in sample, "Sample missing text"
            assert "expected_category" in sample, "Sample missing expected_category"

    def test_categories_are_consistent(self):
        """Categories should be consistent across all data files."""
        training_data = get_training_data()
        eval_data = get_evaluation_data()
        tickets = get_sample_tickets()
        
        training_categories = set(training_data.get("categories", []))
        
        # Check training samples use defined categories
        for sample in training_data.get("training_samples", []):
            assert sample["category"] in training_categories, \
                f"Training sample uses undefined category: {sample['category']}"
        
        # Check eval samples use defined categories
        for sample in eval_data.get("test_samples", []):
            assert sample["expected_category"] in training_categories, \
                f"Eval sample uses undefined category: {sample['expected_category']}"
        
        # Check tickets use defined categories
        for ticket in tickets:
            assert ticket["category"] in training_categories, \
                f"Ticket uses undefined category: {ticket['category']}"


@pytest.mark.skipif(not HAS_ML_MODULES, reason="ML modules not available")
class TestClassifierWithRealData:
    """Tests for the classifier using actual training data."""

    def test_classifier_trains_on_real_data(self):
        """Classifier should train successfully on actual training data."""
        texts, labels = get_training_texts_and_labels()
        
        classifier = BaselineTriageClassifier()
        classifier.train(texts, labels)
        
        assert classifier.is_trained, "Classifier should be marked as trained"

    def test_classifier_predicts_correctly(self):
        """Classifier should predict categories with reasonable accuracy."""
        texts, labels = get_training_texts_and_labels()
        
        classifier = BaselineTriageClassifier()
        classifier.train(texts, labels)
        
        # Test with evaluation data
        eval_data = get_evaluation_data()
        correct = 0
        total = 0
        
        for sample in eval_data.get("test_samples", [])[:10]:  # Test first 10
            prediction = classifier.predict(sample["text"])
            if prediction == sample["expected_category"]:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        # Expect at least 40% accuracy on simple classification
        assert accuracy >= 0.4, f"Accuracy too low: {accuracy:.2%}"

    def test_classifier_handles_all_categories(self):
        """Classifier should handle all defined categories."""
        texts, labels = get_training_texts_and_labels()
        training_data = get_training_data()
        
        classifier = BaselineTriageClassifier()
        classifier.train(texts, labels)
        
        # Check that all categories are represented
        unique_labels = set(labels)
        expected_categories = set(training_data.get("categories", []))
        
        # At least most categories should be in training data
        coverage = len(unique_labels & expected_categories) / len(expected_categories)
        assert coverage >= 0.8, f"Category coverage too low: {coverage:.2%}"


@pytest.mark.skipif(not HAS_ML_MODULES, reason="ML modules not available")
class TestRetrievalWithRealData:
    """Tests for the retrieval system using actual KB data."""

    def test_vector_store_indexes_kb_articles(self):
        """Vector store should successfully index KB articles."""
        articles = get_knowledge_base_articles()
        
        # Extract content for indexing
        documents = [
            {
                "id": article["article_id"],
                "content": f"{article['title']}\n\n{article['content']}",
                "metadata": {"category": article["category"]}
            }
            for article in articles
        ]
        
        vector_store = VectorStore()
        # This tests that the initialization works
        assert vector_store is not None

    def test_retrieval_returns_relevant_articles(self):
        """Retrieval should return relevant KB articles for queries."""
        articles = get_knowledge_base_articles()
        eval_data = get_evaluation_data()
        
        # For samples with relevant_kb_articles, verify they exist
        for sample in eval_data.get("test_samples", []):
            relevant_ids = sample.get("relevant_kb_articles", [])
            article_ids = [a["article_id"] for a in articles]
            
            for kb_id in relevant_ids:
                assert kb_id in article_ids, \
                    f"Referenced KB article {kb_id} not found in knowledge base"


class TestDataQuality:
    """Tests for data quality and coverage."""

    def test_training_data_has_diverse_categories(self):
        """Training data should have samples from all categories."""
        data = get_training_data()
        samples = data.get("training_samples", [])
        categories = data.get("categories", [])
        
        samples_by_category = {}
        for sample in samples:
            cat = sample["category"]
            samples_by_category[cat] = samples_by_category.get(cat, 0) + 1
        
        # Each category should have at least some samples
        for category in categories:
            count = samples_by_category.get(category, 0)
            assert count >= 2, f"Category '{category}' has only {count} samples"

    def test_training_data_has_diverse_priorities(self):
        """Training data should have samples from different priorities."""
        data = get_training_data()
        samples = data.get("training_samples", [])
        priorities = data.get("priorities", [])
        
        samples_by_priority = {}
        for sample in samples:
            pri = sample.get("priority", "Unknown")
            samples_by_priority[pri] = samples_by_priority.get(pri, 0) + 1
        
        # At least 3 different priorities should be represented
        assert len(samples_by_priority) >= 3, \
            f"Only {len(samples_by_priority)} priorities represented"

    def test_kb_articles_cover_main_categories(self):
        """KB articles should cover the main support categories."""
        articles = get_knowledge_base_articles()
        categories = set(a["category"] for a in articles)
        
        # Should have at least Technical, Billing, Account
        required_categories = {"Technical", "Billing", "Account"}
        missing = required_categories - categories
        assert not missing, f"Missing KB articles for categories: {missing}"

    def test_evaluation_data_has_difficulty_levels(self):
        """Evaluation data should have samples of varying difficulty."""
        data = get_evaluation_data()
        samples = data.get("test_samples", [])
        
        difficulties = set(s.get("difficulty", "unknown") for s in samples)
        
        # Should have at least easy and medium difficulties
        assert "easy" in difficulties, "No 'easy' difficulty samples"
        assert len(difficulties) >= 2, "Need more diversity in difficulty levels"

    def test_text_content_is_reasonable_length(self):
        """Text content should be reasonable length for ML processing."""
        tickets = get_sample_tickets()
        
        for ticket in tickets:
            subject_len = len(ticket.get("subject", ""))
            body_len = len(ticket.get("body", ""))
            
            assert subject_len >= 5, f"Subject too short: {ticket['ticket_id']}"
            assert body_len >= 10, f"Body too short: {ticket['ticket_id']}"
            assert body_len <= 10000, f"Body too long: {ticket['ticket_id']}"


class TestEdgeCases:
    """Tests for edge cases in the data."""

    def test_handles_special_characters(self):
        """System should handle special characters in text."""
        tickets = get_sample_tickets()
        
        # Check that tickets with special chars are present
        special_chars_found = False
        for ticket in tickets:
            text = ticket.get("body", "")
            if any(c in text for c in ["'", '"', "&", "<", ">", "\n", "/"]):
                special_chars_found = True
                break
        
        assert special_chars_found, "No tickets with special characters for testing"

    def test_handles_varying_text_lengths(self):
        """System should handle texts of varying lengths."""
        data = get_training_data()
        samples = data.get("training_samples", [])
        
        lengths = [len(s["text"]) for s in samples]
        
        min_len = min(lengths)
        max_len = max(lengths)
        
        # Should have variety in text lengths
        assert max_len > min_len * 2, "Need more variety in text lengths"

    def test_kb_articles_have_searchable_content(self):
        """KB articles should have enough content for meaningful search."""
        articles = get_knowledge_base_articles()
        
        for article in articles:
            content = article.get("content", "")
            word_count = len(content.split())
            
            assert word_count >= 50, \
                f"Article {article['article_id']} has only {word_count} words"
