"""
Integration tests for LOCALTRIAGE
Tests that verify components work together correctly
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Embedding + Vector Store Integration
# =============================================================================

class TestVectorSearchIntegration:
    """Tests for embedding model + vector store integration"""
    
    def test_faiss_add_and_search(self):
        """Test adding and searching vectors in FAISS"""
        from retrieval.vector_search import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=128)
        
        # Create test embeddings
        embeddings = np.random.rand(5, 128).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        documents = [
            {"id": f"doc_{i}", "content": f"Document {i}", "metadata": {"index": i}}
            for i in range(5)
        ]
        
        # Add to store
        ids = store.add(embeddings, documents)
        assert len(ids) == 5
        
        # Search with one of the embeddings
        query = embeddings[0:1]
        results = store.search(query, top_k=3)
        
        assert len(results) <= 3
        # Top result should be the same document (highest similarity)
        assert results[0].score > 0.9  # High similarity to itself


class TestClassifierIntegration:
    """Tests for classifier training and prediction"""
    
    def test_train_and_predict(self):
        """Test training classifier and making predictions"""
        from triage.baseline_classifier import BaselineRouter
        
        router = BaselineRouter(max_features=100, min_df=1)
        
        # Training data
        subjects = [
            "Cannot login",
            "Password reset",
            "Billing error",
            "Charged twice",
            "App crashes",
            "Feature not working"
        ]
        bodies = [
            "I cannot access my account",
            "Need to reset my password",
            "There's an error in my invoice",
            "I was charged twice for the same item",
            "The application crashes on startup",
            "The export feature is not working"
        ]
        categories = ["Account", "Account", "Billing", "Billing", "Technical", "Technical"]
        priorities = ["P2", "P2", "P2", "P1", "P2", "P3"]
        
        # Train
        metrics = router.fit(subjects, bodies, categories, priorities)
        assert router.is_fitted
        
        # Predict
        prediction = router.predict("Login problem", "Can't access my account")
        assert prediction.category in ["Account", "Billing", "Technical"]
        assert prediction.priority in ["P1", "P2", "P3"]
        assert 0 <= prediction.category_confidence <= 1


# =============================================================================
# Retrieval Evaluator Integration
# =============================================================================

class TestRetrievalEvaluatorIntegration:
    """Tests for retrieval evaluation metrics"""
    
    def test_batch_evaluation(self):
        """Test evaluating multiple queries"""
        from retrieval.baseline_bm25 import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator()
        
        # Simulate multiple query results
        queries_results = [
            (["doc1", "doc2", "doc3"], ["doc1", "doc4"]),  # 1/2 recall@3
            (["doc5", "doc6", "doc7"], ["doc5", "doc6"]),  # 2/2 recall@3
            (["doc8", "doc9", "doc10"], ["doc11"]),         # 0/1 recall@3
        ]
        
        total_recall = 0
        for retrieved, relevant in queries_results:
            total_recall += evaluator.recall_at_k(retrieved, relevant, k=3)
        
        avg_recall = total_recall / len(queries_results)
        assert 0 <= avg_recall <= 1
    
    def test_ndcg_calculation(self):
        """Test nDCG calculation for ranking"""
        from retrieval.baseline_bm25 import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator()
        
        # Perfect ranking
        retrieved_perfect = ["rel1", "rel2", "irr1"]
        relevant = {"rel1": 3, "rel2": 2}  # relevance scores
        
        # Calculate nDCG
        ndcg = evaluator.ndcg_at_k(retrieved_perfect, relevant, k=3)
        # nDCG is between 0 and 1
        assert 0 <= ndcg <= 1


# =============================================================================
# Template Responder Integration
# =============================================================================

class TestTemplateResponderIntegration:
    """Tests for template-based response generation"""
    
    def test_response_for_different_categories(self):
        """Test generating responses for various categories"""
        from rag.drafter import BaselineTemplateResponder
        
        responder = BaselineTemplateResponder()
        
        test_cases = [
            ("Billing", "Invoice error", "My invoice is incorrect"),
            ("Technical", "App crash", "Application crashes"),
            ("Account", "Password", "Forgot password"),
        ]
        
        for category, subject, body in test_cases:
            response = responder.generate_response(
                subject=subject,
                body=body,
                category=category
            )
            assert response is not None
            assert len(response) > 0
            assert isinstance(response, str)


# =============================================================================
# Data Flow Integration
# =============================================================================

class TestDataFlowIntegration:
    """Tests for end-to-end data flow"""
    
    def test_ticket_classification_flow(self):
        """Test full flow: ticket -> classification -> response template"""
        from ingestion.ingest import Ticket
        from triage.baseline_classifier import BaselineRouter
        from rag.drafter import BaselineTemplateResponder
        
        # Create ticket
        ticket = Ticket(
            subject="Billing Error",
            body="I was charged twice for my subscription",
            customer_email="customer@example.com"
        )
        
        # Mock classifier (since training requires more data)
        router = BaselineRouter(max_features=100, min_df=1)
        
        # Quick training with minimal data (need at least 2 classes for each)
        router.fit(
            subjects=["billing", "technical", "account", "billing2", "technical2", "account2"],
            bodies=["invoice issue", "app crash", "password reset", "payment error", "bug report", "login issue"],
            categories=["Billing", "Technical", "Account", "Billing", "Technical", "Account"],
            priorities=["P2", "P2", "P2", "P1", "P1", "P3"]
        )
        
        # Classify
        prediction = router.predict(ticket.subject, ticket.body)
        
        # Generate template response
        responder = BaselineTemplateResponder()
        response = responder.generate_response(
            subject=ticket.subject,
            body=ticket.body,
            category=prediction.category
        )
        
        assert response is not None
        assert len(response) > 0
    
    def test_vector_search_flow(self):
        """Test flow: documents -> embeddings -> search"""
        from retrieval.vector_search import FAISSVectorStore
        
        # Create store
        store = FAISSVectorStore(dimension=64)
        
        # Create simple embeddings for documents
        num_docs = 10
        embeddings = np.random.rand(num_docs, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        documents = [
            {
                "id": f"kb_{i}",
                "content": f"Knowledge base article {i}",
                "metadata": {"category": "test"}
            }
            for i in range(num_docs)
        ]
        
        # Add documents
        store.add(embeddings, documents)
        
        # Search
        query_embedding = np.random.rand(1, 64).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = store.search(query_embedding, top_k=5)
        
        assert len(results) == 5
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'content') for r in results)


# =============================================================================
# Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Tests for configuration and initialization"""
    
    def test_database_connection_params(self):
        """Test database connection parameter handling"""
        from ingestion.ingest import DatabaseConnection
        
        conn = DatabaseConnection(
            host="localhost",
            port=5432,
            database="localtriage",
            user="test_user",
            password="test_pass"
        )
        
        assert conn.conn_params['host'] == "localhost"
        assert conn.conn_params['port'] == 5432
        assert conn.conn_params['database'] == "localtriage"
    
    def test_llm_client_params(self):
        """Test LLM client parameter handling"""
        from rag.drafter import LLMClient
        
        client = LLMClient(
            base_url="http://localhost:8000/v1",
            model_name="test-model",
            max_tokens=2048,
            temperature=0.5
        )
        
        assert client.base_url == "http://localhost:8000/v1"
        assert client.model_name == "test-model"
        assert client.max_tokens == 2048
        assert client.temperature == 0.5
    
    def test_router_params(self):
        """Test router parameter handling"""
        from triage.baseline_classifier import BaselineRouter
        
        router = BaselineRouter(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.9
        )
        
        assert router.vectorizer.max_features == 5000
        assert router.vectorizer.ngram_range == (1, 3)
