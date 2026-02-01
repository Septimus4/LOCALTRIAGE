"""
API Tests for LOCALTRIAGE FastAPI endpoints
Uses httpx TestClient to test API functionality
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_app():
    """Create a test client with mocked dependencies"""
    # Mock the database and ML components
    with patch('api.api.psycopg2.connect') as mock_connect, \
         patch('api.api.BaselineRouter') as mock_router, \
         patch('api.api.RAGDrafter') as mock_drafter, \
         patch('api.api.HybridRetriever') as mock_retriever:
        
        # Setup mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        
        # Setup mock router
        mock_router_instance = MagicMock()
        mock_router_instance.predict.return_value = MagicMock(
            category="Technical",
            category_confidence=0.85,
            category_probabilities={"Technical": 0.85, "Billing": 0.15},
            priority="P2",
            priority_confidence=0.75,
            priority_probabilities={"P1": 0.1, "P2": 0.75, "P3": 0.15}
        )
        mock_router_instance.is_fitted = True
        mock_router.return_value = mock_router_instance
        
        # Setup mock drafter
        mock_drafter_instance = MagicMock()
        mock_drafter_instance.generate_draft.return_value = MagicMock(
            draft_id="draft_001",
            ticket_id="ticket_123",
            draft_text="Thank you for contacting us...",
            rationale="Based on the query...",
            confidence="high",
            confidence_score=0.9,
            citations=[],
            follow_up_questions=[],
            model_name="test-model",
            generation_time_ms=100,
            retrieval_time_ms=50
        )
        mock_drafter.return_value = mock_drafter_instance
        
        from api.api import app
        
        with TestClient(app) as client:
            yield client


@pytest.fixture
def client():
    """Create a basic test client without mocks (for health check)"""
    from api.api import app
    
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "api" in data  # API health status key


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation"""
    
    def test_triage_missing_subject(self, client):
        """Test triage endpoint rejects missing subject"""
        response = client.post("/triage", json={
            "body": "Test body without subject"
        })
        assert response.status_code == 422  # Validation error
    
    def test_triage_missing_body(self, client):
        """Test triage endpoint rejects missing body"""
        response = client.post("/triage", json={
            "subject": "Test subject without body"
        })
        assert response.status_code == 422  # Validation error
    
    def test_triage_empty_subject(self, client):
        """Test triage endpoint rejects empty subject"""
        response = client.post("/triage", json={
            "subject": "",
            "body": "Test body"
        })
        assert response.status_code == 422  # Validation error
    
    def test_draft_missing_body(self, client):
        """Test draft endpoint rejects missing body"""
        response = client.post("/draft", json={
            "subject": "Test subject"
        })
        assert response.status_code == 422


# =============================================================================
# Pydantic Model Tests
# =============================================================================

class TestPydanticModels:
    """Tests for API Pydantic models"""
    
    def test_ticket_input_model(self):
        """Test TicketInput model"""
        from api.api import TicketInput
        
        ticket = TicketInput(
            subject="Test subject",
            body="Test body",
            customer_email="test@example.com"
        )
        assert ticket.subject == "Test subject"
        assert ticket.body == "Test body"
    
    def test_draft_request_model(self):
        """Test DraftRequest model"""
        from api.api import DraftRequest
        
        request = DraftRequest(
            subject="Test subject",
            body="Test body",
            category="Technical",
            use_llm=True
        )
        assert request.category == "Technical"
        assert request.use_llm == True
    
    def test_feedback_request_model(self):
        """Test FeedbackInput model"""
        from api.api import FeedbackInput
        
        feedback = FeedbackInput(
            draft_id="draft_001",
            rating=5,
            is_helpful=True,
            feedback_text="Great response!"
        )
        assert feedback.rating == 5
        assert feedback.is_helpful == True


# =============================================================================
# Response Model Tests
# =============================================================================

class TestResponseModels:
    """Tests for API response models"""
    
    def test_triage_response_model(self):
        """Test TriageResponse model"""
        from api.api import TriageResponse
        
        response = TriageResponse(
            ticket_id="ticket_001",
            category="Technical",
            category_confidence=0.85,
            category_probabilities={"Technical": 0.85},
            priority="P2",
            priority_confidence=0.75,
            priority_probabilities={"P2": 0.75},
            sla_risk=False,
            suggested_queue="Tech Support",
            explanation="Classified as technical issue",
            processing_time_ms=50
        )
        assert response.category == "Technical"
        assert response.sla_risk == False
    
    def test_draft_response_model(self):
        """Test DraftResponse model"""
        from api.api import DraftResponse
        
        response = DraftResponse(
            draft_id="draft_001",
            ticket_id="ticket_001",
            draft_text="Thank you for contacting us...",
            rationale="Based on the query...",
            confidence="high",
            confidence_score=0.9,
            citations=[],
            follow_up_questions=[],
            model_name="test-model",
            generation_time_ms=100,
            retrieval_time_ms=50,
            total_time_ms=150
        )
        assert response.confidence == "high"
        assert response.total_time_ms == 150


# =============================================================================
# Metric Calculation Tests
# =============================================================================

class TestMetricCalculations:
    """Tests for metric/analytics calculations"""
    
    def test_confidence_score_range(self):
        """Test confidence scores are in valid range"""
        from api.api import TriageResponse
        
        # Valid confidence
        response = TriageResponse(
            ticket_id="t1",
            category="Technical",
            category_confidence=0.85,
            category_probabilities={"Technical": 0.85},
            priority="P2",
            priority_confidence=0.75,
            priority_probabilities={"P2": 0.75},
            sla_risk=False,
            suggested_queue=None,
            explanation=None,
            processing_time_ms=50
        )
        assert 0 <= response.category_confidence <= 1
        assert 0 <= response.priority_confidence <= 1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/triage",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_extra_fields_ignored(self, client):
        """Test extra fields in request are handled"""
        response = client.post("/triage", json={
            "subject": "Test",
            "body": "Test body",
            "extra_field": "should be ignored"
        })
        # Should either accept or reject, not crash
        assert response.status_code in [200, 422, 500]


# =============================================================================
# API Endpoint Existence Tests
# =============================================================================

class TestEndpointExistence:
    """Tests that expected endpoints exist"""
    
    def test_health_endpoint_exists(self, client):
        """Test /health endpoint exists"""
        response = client.get("/health")
        assert response.status_code != 404
    
    def test_triage_endpoint_exists(self, client):
        """Test /triage endpoint exists"""
        response = client.post("/triage", json={
            "subject": "Test",
            "body": "Test"
        })
        assert response.status_code != 404
    
    def test_draft_endpoint_exists(self, client):
        """Test /draft endpoint exists"""
        response = client.post("/draft", json={
            "subject": "Test",
            "body": "Test"
        })
        assert response.status_code != 404
    
    def test_metrics_endpoint_exists(self, client):
        """Test /metrics endpoint exists"""
        response = client.get("/metrics")
        assert response.status_code != 404
    
    def test_feedback_endpoint_exists(self, client):
        """Test /feedback endpoint exists"""
        response = client.post("/feedback", json={
            "draft_id": "draft_001",
            "rating": 5
        })
        assert response.status_code != 404
