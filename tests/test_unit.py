"""
Unit tests for LOCALTRIAGE core functionality
Simplified tests that don't require external dependencies
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import csv
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Test Data Models (Ingestion)
# =============================================================================

class TestDataModels:
    """Tests for data classes"""
    
    def test_ticket_creation(self):
        """Test Ticket dataclass"""
        from ingestion.ingest import Ticket
        
        ticket = Ticket(
            subject="Cannot login",
            body="Password reset not working",
            category="Account",
            priority="P2"
        )
        assert ticket.subject == "Cannot login"
        assert ticket.category == "Account"
        assert ticket.priority == "P2"
    
    def test_ticket_to_dict(self):
        """Test Ticket to_dict method"""
        from ingestion.ingest import Ticket
        
        ticket = Ticket(
            subject="Test",
            body="Test body",
            customer_email="test@example.com"
        )
        d = ticket.to_dict()
        assert isinstance(d, dict)
        assert d['subject'] == "Test"
        assert d['customer_email'] == "test@example.com"
    
    def test_kb_article_creation(self):
        """Test KBArticle dataclass"""
        from ingestion.ingest import KBArticle
        
        article = KBArticle(
            title="Password Reset Guide",
            content="To reset your password...",
            category="Account"
        )
        assert article.title == "Password Reset Guide"
    
    def test_kb_chunk_creation(self):
        """Test KBChunk dataclass"""
        from ingestion.ingest import KBChunk
        
        chunk = KBChunk(
            article_id="art_001",
            content="Test content",
            chunk_index=0,
            token_count=25,
            start_char=0,
            end_char=100
        )
        assert chunk.article_id == "art_001"
        assert chunk.chunk_index == 0


# =============================================================================
# Test Routing Prediction
# =============================================================================

class TestRoutingPrediction:
    """Tests for RoutingPrediction dataclass"""
    
    def test_creation(self):
        """Test basic creation"""
        from triage.baseline_classifier import RoutingPrediction
        
        pred = RoutingPrediction(
            category="Technical",
            category_confidence=0.85,
            category_probabilities={"Technical": 0.85, "Billing": 0.15},
            priority="P2",
            priority_confidence=0.75,
            priority_probabilities={"P1": 0.1, "P2": 0.75, "P3": 0.15}
        )
        assert pred.category == "Technical"
        assert pred.category_confidence == 0.85
        assert pred.priority == "P2"


# =============================================================================
# Test Vector Search Result
# =============================================================================

class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass"""
    
    def test_creation(self):
        """Test basic creation"""
        from retrieval.vector_search import VectorSearchResult
        
        result = VectorSearchResult(
            id="chunk_001",
            content="Test content",
            score=0.85,
            rank=1,
            metadata={"source": "test"}
        )
        assert result.id == "chunk_001"
        assert result.score == 0.85
        assert result.rank == 1


# =============================================================================
# Test RAG Components
# =============================================================================

class TestCitation:
    """Tests for Citation dataclass"""
    
    def test_creation(self):
        """Test Citation creation"""
        from rag.drafter import Citation
        
        citation = Citation(
            id="cit_001",
            source_type="kb_article",
            source_id="art_123",
            title="Password Reset",
            excerpt="To reset your password..."
        )
        assert citation.id == "cit_001"
        assert citation.source_type == "kb_article"


class TestDraftResponse:
    """Tests for DraftResponse dataclass"""
    
    def test_creation(self):
        """Test DraftResponse creation"""
        from rag.drafter import DraftResponse, Citation
        
        citation = Citation(
            id="cit_001",
            source_type="kb_article",
            source_id="art_123",
            title="Test",
            excerpt="Test excerpt"
        )
        
        response = DraftResponse(
            draft_id="draft_001",
            ticket_id="ticket_123",
            draft_text="Thank you for contacting us...",
            rationale="Based on the ticket content...",
            confidence="high",
            confidence_score=0.9,
            citations=[citation],
            follow_up_questions=["Do you need more help?"],
            model_name="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            generation_time_ms=500,
            retrieval_time_ms=100,
            sources_used=3
        )
        assert response.draft_id == "draft_001"
        assert response.confidence == "high"
        assert len(response.citations) == 1
    
    def test_to_dict(self):
        """Test DraftResponse to_dict method"""
        from rag.drafter import DraftResponse, Citation
        
        response = DraftResponse(
            draft_id="draft_001",
            ticket_id="ticket_123",
            draft_text="Test draft",
            rationale="Test rationale",
            confidence="medium",
            confidence_score=0.7,
            citations=[],
            follow_up_questions=[],
            model_name="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            generation_time_ms=500,
            retrieval_time_ms=100,
            sources_used=3
        )
        d = response.to_dict()
        assert d['draft_id'] == "draft_001"
        assert d['confidence_score'] == 0.7


# =============================================================================
# Test Template Responder (No LLM required)
# =============================================================================

class TestBaselineTemplateResponder:
    """Tests for baseline template-based responder"""
    
    def test_init(self):
        """Test initialization"""
        from rag.drafter import BaselineTemplateResponder
        
        responder = BaselineTemplateResponder()
        assert responder is not None
    
    def test_generate_billing_response(self):
        """Test billing response generation"""
        from rag.drafter import BaselineTemplateResponder
        
        responder = BaselineTemplateResponder()
        response = responder.generate_response(
            subject="Billing issue",
            body="I was charged twice",
            category="Billing"
        )
        assert "billing" in response.lower() or "charge" in response.lower()
    
    def test_generate_technical_response(self):
        """Test technical response generation"""
        from rag.drafter import BaselineTemplateResponder
        
        responder = BaselineTemplateResponder()
        response = responder.generate_response(
            subject="App not working",
            body="The app crashes on startup",
            category="Technical"
        )
        assert response is not None
        assert len(response) > 0


# =============================================================================
# Test BM25 Retriever
# =============================================================================

class TestBM25Components:
    """Tests for BM25 retrieval components"""
    
    def test_retrieval_evaluator_recall(self):
        """Test recall@k calculation"""
        from retrieval.baseline_bm25 import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator()
        
        # 3 relevant items, retrieved 2 of them in top-5
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6"]
        
        recall = evaluator.recall_at_k(retrieved, relevant, k=5)
        assert recall == 2/3  # Retrieved 2 out of 3 relevant
    
    def test_retrieval_evaluator_precision(self):
        """Test precision@k calculation"""
        from retrieval.baseline_bm25 import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator()
        
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]
        
        precision = evaluator.precision_at_k(retrieved, relevant, k=5)
        assert precision == 2/5  # 2 relevant in top 5
    
    def test_retrieval_evaluator_mrr(self):
        """Test MRR calculation"""
        from retrieval.baseline_bm25 import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator()
        
        # First relevant at position 2 (0-indexed: 1)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]
        
        mrr = evaluator.mrr(retrieved, relevant)
        assert mrr == 0.5  # 1/2


# =============================================================================
# Test FAISS Vector Store
# =============================================================================

class TestFAISSVectorStore:
    """Tests for FAISS vector store"""
    
    def test_init(self):
        """Test FAISS store initialization"""
        from retrieval.vector_search import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=768)
        assert store.dimension == 768
        assert store.index is not None


# =============================================================================
# Test File Operations
# =============================================================================

class TestFileOperations:
    """Tests for file reading/writing"""
    
    def test_read_csv(self, tmp_path):
        """Test reading CSV file"""
        csv_file = tmp_path / "tickets.csv"
        data = [
            {"subject": "Cannot login", "body": "Password reset not working"},
            {"subject": "Billing error", "body": "Charged twice"},
        ]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            tickets = list(reader)
        
        assert len(tickets) == 2
        assert tickets[0]['subject'] == "Cannot login"
    
    def test_read_json(self, tmp_path):
        """Test reading JSON file"""
        json_file = tmp_path / "tickets.json"
        data = [
            {"subject": "Cannot login", "body": "Password reset not working"},
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        with open(json_file, 'r') as f:
            tickets = json.load(f)
        
        assert len(tickets) == 1
        assert tickets[0]['subject'] == "Cannot login"


# =============================================================================
# Test Database Connection (Mocked)
# =============================================================================

class TestDatabaseConnection:
    """Tests for database connection"""
    
    def test_init(self):
        """Test DatabaseConnection initialization"""
        from ingestion.ingest import DatabaseConnection
        
        conn = DatabaseConnection(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass"
        )
        assert conn.conn_params['host'] == "localhost"
        assert conn.conn_params['database'] == "testdb"
    
    def test_context_manager(self):
        """Test context manager pattern"""
        from ingestion.ingest import DatabaseConnection
        
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            db = DatabaseConnection()
            with db as conn:
                assert conn is not None
            
            mock_conn.close.assert_called_once()


# =============================================================================
# Test LLM Client (Mocked)
# =============================================================================

class TestLLMClient:
    """Tests for LLM client"""
    
    def test_init(self):
        """Test LLMClient initialization"""
        from rag.drafter import LLMClient
        
        client = LLMClient(
            base_url="http://localhost:8000/v1",
            model_name="test-model"
        )
        assert client.base_url == "http://localhost:8000/v1"
        assert client.model_name == "test-model"
    
    def test_generate_mocked(self):
        """Test generate with mocked response"""
        from rag.drafter import LLMClient
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Test response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }
            mock_post.return_value = mock_response
            
            client = LLMClient(base_url="http://localhost:8000/v1")
            messages = [{"role": "user", "content": "Hello"}]
            result = client.generate(messages)
            
            # The LLMClient.generate returns a processed dict, not raw response
            assert result is not None
            assert "content" in result


# =============================================================================
# Test Classifier (Mocked)
# =============================================================================

class TestBaselineRouter:
    """Tests for baseline routing classifier"""
    
    def test_init(self):
        """Test BaselineRouter initialization"""
        from triage.baseline_classifier import BaselineRouter
        
        router = BaselineRouter(
            max_features=5000,
            ngram_range=(1, 2)
        )
        assert router.vectorizer.max_features == 5000
        assert router.is_fitted == False
    
    def test_combine_text(self):
        """Test text combination"""
        from triage.baseline_classifier import BaselineRouter
        
        router = BaselineRouter()
        combined = router._combine_text("Subject here", "Body content")
        
        assert "Subject here" in combined
        assert "Body content" in combined
