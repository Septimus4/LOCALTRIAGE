"""
Test fixtures providing sample data for unit and integration tests.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_sample_tickets() -> List[Dict[str, Any]]:
    """Load sample tickets from JSON file."""
    file_path = DATA_DIR / "raw" / "sample_tickets.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return get_default_sample_tickets()


def get_knowledge_base_articles() -> List[Dict[str, Any]]:
    """Load knowledge base articles for RAG testing."""
    file_path = DATA_DIR / "raw" / "knowledge_base_articles.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return get_default_kb_articles()


def get_training_data() -> Dict[str, Any]:
    """Load classifier training data."""
    file_path = DATA_DIR / "processed" / "classifier_training_data.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return get_default_training_data()


def get_evaluation_data() -> Dict[str, Any]:
    """Load evaluation dataset with ground truth labels."""
    file_path = DATA_DIR / "processed" / "evaluation_dataset.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return get_default_evaluation_data()


# Default fallback data for when files don't exist
def get_default_sample_tickets() -> List[Dict[str, Any]]:
    """Default sample tickets for testing."""
    return [
        {
            "ticket_id": "TKT-001",
            "subject": "Cannot login to my account",
            "body": "I am trying to login but getting an error message.",
            "category": "Account",
            "priority": "P2-High",
            "customer_email": "test@example.com"
        },
        {
            "ticket_id": "TKT-002",
            "subject": "Application crashes on startup",
            "body": "The app crashes every time I try to open it.",
            "category": "Technical",
            "priority": "P1-Critical",
            "customer_email": "user@example.com"
        },
        {
            "ticket_id": "TKT-003",
            "subject": "Billing question about charges",
            "body": "I see an unexpected charge on my invoice.",
            "category": "Billing",
            "priority": "P3-Medium",
            "customer_email": "billing@example.com"
        },
        {
            "ticket_id": "TKT-004",
            "subject": "Feature request: Dark mode",
            "body": "Please add dark mode to the application.",
            "category": "Feature Request",
            "priority": "P4-Low",
            "customer_email": "feature@example.com"
        },
        {
            "ticket_id": "TKT-005",
            "subject": "Data sync not working",
            "body": "Changes on mobile not syncing to desktop.",
            "category": "Technical",
            "priority": "P2-High",
            "customer_email": "sync@example.com"
        }
    ]


def get_default_kb_articles() -> List[Dict[str, Any]]:
    """Default KB articles for testing."""
    return [
        {
            "article_id": "KB-001",
            "title": "How to Reset Your Password",
            "category": "Account",
            "content": "To reset your password, go to login page and click Forgot Password.",
            "tags": ["password", "reset", "login"]
        },
        {
            "article_id": "KB-002",
            "title": "Understanding Your Bill",
            "category": "Billing",
            "content": "Your invoice includes subscription fees, add-ons, and taxes.",
            "tags": ["billing", "invoice", "charges"]
        },
        {
            "article_id": "KB-003",
            "title": "Troubleshooting Application Crashes",
            "category": "Technical",
            "content": "If the app crashes, try clearing cache and restarting.",
            "tags": ["crash", "troubleshooting", "error"]
        }
    ]


def get_default_training_data() -> Dict[str, Any]:
    """Default training data for classifier."""
    return {
        "categories": ["Technical", "Billing", "Account", "Feature Request", "General Inquiry"],
        "priorities": ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"],
        "training_samples": [
            {"text": "App crashes on startup", "category": "Technical", "priority": "P1-Critical"},
            {"text": "Charged twice this month", "category": "Billing", "priority": "P2-High"},
            {"text": "Reset my password", "category": "Account", "priority": "P3-Medium"},
            {"text": "Add dark mode", "category": "Feature Request", "priority": "P4-Low"},
            {"text": "What are your hours", "category": "General Inquiry", "priority": "P4-Low"},
        ]
    }


def get_default_evaluation_data() -> Dict[str, Any]:
    """Default evaluation data for testing."""
    return {
        "test_samples": [
            {
                "id": "eval-001",
                "text": "App keeps freezing when uploading",
                "expected_category": "Technical",
                "expected_priority": "P2-High"
            },
            {
                "id": "eval-002",
                "text": "Invoice shows wrong amount",
                "expected_category": "Billing",
                "expected_priority": "P2-High"
            },
            {
                "id": "eval-003",
                "text": "Can't login to my account",
                "expected_category": "Account",
                "expected_priority": "P2-High"
            }
        ]
    }


# Convenience functions for specific test scenarios
def get_technical_tickets() -> List[Dict[str, Any]]:
    """Get tickets categorized as Technical."""
    tickets = get_sample_tickets()
    return [t for t in tickets if t.get("category") == "Technical"]


def get_billing_tickets() -> List[Dict[str, Any]]:
    """Get tickets categorized as Billing."""
    tickets = get_sample_tickets()
    return [t for t in tickets if t.get("category") == "Billing"]


def get_high_priority_tickets() -> List[Dict[str, Any]]:
    """Get P1 and P2 priority tickets."""
    tickets = get_sample_tickets()
    return [t for t in tickets if t.get("priority") in ["P1-Critical", "P2-High"]]


def get_training_texts_and_labels() -> tuple:
    """Get training texts and category labels for classifier training."""
    data = get_training_data()
    samples = data.get("training_samples", [])
    texts = [s["text"] for s in samples]
    labels = [s["category"] for s in samples]
    return texts, labels


def get_kb_articles_for_category(category: str) -> List[Dict[str, Any]]:
    """Get KB articles for a specific category."""
    articles = get_knowledge_base_articles()
    return [a for a in articles if a.get("category") == category]


def create_test_ticket(
    ticket_id: str = "TEST-001",
    subject: str = "Test Subject",
    body: str = "Test body content",
    category: str = "Technical",
    priority: str = "P3-Medium"
) -> Dict[str, Any]:
    """Create a test ticket with specified or default values."""
    return {
        "ticket_id": ticket_id,
        "subject": subject,
        "body": body,
        "category": category,
        "priority": priority,
        "customer_email": "test@example.com",
        "created_at": "2025-01-15T10:00:00Z",
        "status": "new"
    }


def create_test_kb_article(
    article_id: str = "KB-TEST",
    title: str = "Test Article",
    content: str = "Test content",
    category: str = "Technical"
) -> Dict[str, Any]:
    """Create a test KB article with specified or default values."""
    return {
        "article_id": article_id,
        "title": title,
        "content": content,
        "category": category,
        "tags": ["test"],
        "last_updated": "2025-01-15T10:00:00Z"
    }
