"""
End-to-End API Tests for LOCALTRIAGE
Python-based alternative to Postman/Newman tests
Can be run with pytest for simpler CI integration
"""
import pytest
import httpx
import os
import time
from typing import Optional

# Configuration
BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0


@pytest.fixture(scope="module")
def client():
    """HTTP client for E2E tests"""
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
def async_client():
    """Async HTTP client for E2E tests"""
    return httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)


class TestHealthEndpoint:
    """E2E tests for health endpoint"""
    
    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_has_api_status(self, client):
        """Health response should have API status"""
        response = client.get("/health")
        data = response.json()
        assert "api" in data
        assert data["api"] == "healthy"
    
    def test_health_response_time(self, client):
        """Health check should respond quickly"""
        start = time.time()
        client.get("/health")
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Health check took {elapsed:.2f}s"


class TestTriageEndpoint:
    """E2E tests for ticket triage"""
    
    def test_triage_technical_issue(self, client):
        """Triage should classify technical issues"""
        response = client.post("/triage", json={
            "subject": "Application crashes on startup",
            "body": "The app crashes when I try to open it. Error message shows null pointer exception.",
            "customer_email": "test@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "ticket_id" in data
        assert "category" in data
        assert "priority" in data
        assert 0 <= data["category_confidence"] <= 1
        assert data["priority"] in ["P1", "P2", "P3", "P4"]
    
    def test_triage_billing_issue(self, client):
        """Triage should classify billing issues"""
        response = client.post("/triage", json={
            "subject": "Charged twice for subscription",
            "body": "I see two charges on my credit card for the same subscription."
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "category_probabilities" in data
        assert isinstance(data["category_probabilities"], dict)
    
    def test_triage_account_issue(self, client):
        """Triage should classify account issues"""
        response = client.post("/triage", json={
            "subject": "Cannot reset password",
            "body": "Password reset email not arriving."
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "sla_risk" in data
        assert isinstance(data["sla_risk"], bool)
    
    def test_triage_missing_subject_returns_422(self, client):
        """Triage should reject requests without subject"""
        response = client.post("/triage", json={
            "body": "Missing the subject field"
        })
        assert response.status_code == 422
    
    def test_triage_missing_body_returns_422(self, client):
        """Triage should reject requests without body"""
        response = client.post("/triage", json={
            "subject": "Missing the body field"
        })
        assert response.status_code == 422
    
    def test_triage_empty_subject_returns_422(self, client):
        """Triage should reject empty subject"""
        response = client.post("/triage", json={
            "subject": "",
            "body": "Valid body"
        })
        assert response.status_code == 422
    
    def test_triage_has_processing_time(self, client):
        """Triage response should include processing time"""
        response = client.post("/triage", json={
            "subject": "Test ticket",
            "body": "Test body content"
        })
        data = response.json()
        
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], int)


class TestDraftEndpoint:
    """E2E tests for draft generation"""
    
    def test_draft_with_llm(self, client):
        """Draft endpoint should generate LLM response"""
        response = client.post("/draft", json={
            "subject": "How do I export data?",
            "body": "I need to export my project data to CSV format.",
            "use_llm": True
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "draft_id" in data
        assert "draft_text" in data
        assert len(data["draft_text"]) > 0
        assert "confidence" in data
        assert data["confidence"] in ["high", "medium", "low"]
    
    def test_draft_without_llm(self, client):
        """Draft endpoint should generate template response"""
        response = client.post("/draft", json={
            "subject": "Billing question",
            "body": "What payment methods do you accept?",
            "category": "Billing",
            "use_llm": False
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "draft_text" in data
        assert len(data["draft_text"]) > 0
    
    def test_draft_with_category_hint(self, client):
        """Draft should accept category hint"""
        response = client.post("/draft", json={
            "subject": "Integration not working",
            "body": "API integration stopped working",
            "category": "Technical",
            "priority": "P1"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "citations" in data
        assert isinstance(data["citations"], list)
        assert "follow_up_questions" in data
    
    def test_draft_has_timing_metrics(self, client):
        """Draft response should include timing metrics"""
        response = client.post("/draft", json={
            "subject": "Test",
            "body": "Test body"
        })
        data = response.json()
        
        assert "generation_time_ms" in data
        assert "retrieval_time_ms" in data


class TestSimilarEndpoint:
    """E2E tests for similar tickets search"""
    
    def test_similar_tickets_search(self, client):
        """Similar endpoint should return results"""
        response = client.post("/similar", json={
            "query": "password reset not working",
            "limit": 5
        })
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            assert "ticket_id" in data[0]
            assert "similarity_score" in data[0]


@pytest.mark.skip(reason="Draft persistence not implemented - drafts not stored in DB")
class TestFeedbackEndpoint:
    """E2E tests for feedback submission
    
    NOTE: These tests are skipped because the /draft endpoint doesn't persist
    drafts to the database. The /feedback endpoint requires draft_id to exist
    in response_drafts table (foreign key constraint). This is a known 
    limitation that should be addressed in a future iteration.
    """
    
    @pytest.fixture
    def draft_id(self, client):
        """Get a draft_id to use for feedback"""
        response = client.post("/draft", json={
            "subject": "Test",
            "body": "Test body",
            "use_llm": False
        })
        return response.json()["draft_id"]
    
    def test_positive_feedback(self, client, draft_id):
        """Should accept positive feedback"""
        response = client.post("/feedback", json={
            "draft_id": draft_id,
            "rating": 5,
            "is_helpful": True,
            "feedback_text": "Great response!"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
    
    def test_negative_feedback_with_correction(self, client, draft_id):
        """Should accept negative feedback with correction"""
        response = client.post("/feedback", json={
            "draft_id": draft_id,
            "rating": 2,
            "is_helpful": False,
            "feedback_text": "Missed the main issue",
            "correction_text": "The customer was asking about X, not Y"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "feedback_id" in data


class TestMetricsEndpoint:
    """E2E tests for metrics endpoint"""
    
    def test_metrics_default_period(self, client):
        """Metrics should return daily stats by default"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        
        assert "period" in data
        assert "total_tickets" in data
    
    def test_metrics_weekly(self, client):
        """Metrics should support weekly period"""
        response = client.get("/metrics?period=week")
        assert response.status_code == 200
        data = response.json()
        
        assert data["period"] == "week"
    
    def test_metrics_monthly(self, client):
        """Metrics should support monthly period"""
        response = client.get("/metrics?period=month")
        assert response.status_code == 200
        data = response.json()
        
        assert data["period"] == "month"
    
    def test_metrics_invalid_period(self, client):
        """Metrics should reject invalid period"""
        response = client.get("/metrics?period=invalid")
        assert response.status_code == 422


class TestTicketsEndpoint:
    """E2E tests for tickets listing"""
    
    def test_list_tickets_default(self, client):
        """Should list tickets with default pagination"""
        response = client.get("/tickets")
        assert response.status_code == 200
        data = response.json()
        
        assert "tickets" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
    
    def test_list_tickets_with_category(self, client):
        """Should filter tickets by category"""
        response = client.get("/tickets?category=Technical")
        assert response.status_code == 200
        data = response.json()
        
        for ticket in data["tickets"]:
            assert ticket["category"] == "Technical"
    
    def test_list_tickets_pagination(self, client):
        """Should support pagination"""
        response = client.get("/tickets?page=2&page_size=5")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["tickets"]) <= 5
        assert data["page"] == 2


class TestErrorHandling:
    """E2E tests for error handling"""
    
    def test_invalid_json(self, client):
        """Should handle invalid JSON gracefully"""
        response = client.post(
            "/triage",
            content="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_nonexistent_endpoint(self, client):
        """Should return 404 for non-existent endpoints"""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestPerformance:
    """E2E performance tests"""
    
    def test_triage_response_time(self, client):
        """Triage should respond within 5 seconds"""
        start = time.time()
        response = client.post("/triage", json={
            "subject": "Performance test",
            "body": "Testing response time"
        })
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 5.0, f"Triage took {elapsed:.2f}s"
    
    def test_draft_response_time(self, client):
        """Draft generation should complete within 30 seconds"""
        start = time.time()
        response = client.post("/draft", json={
            "subject": "Performance test",
            "body": "Testing response time",
            "use_llm": True
        })
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 30.0, f"Draft took {elapsed:.2f}s"


# Run with: pytest tests/e2e/test_e2e.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
