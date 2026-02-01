"""
Baseline Routing Classifier for LOCALTRIAGE
TF-IDF + Logistic Regression approach
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib


@dataclass
class RoutingPrediction:
    """Represents a routing prediction result"""
    category: str
    category_confidence: float
    category_probabilities: Dict[str, float]
    priority: str
    priority_confidence: float
    priority_probabilities: Dict[str, float]
    

class BaselineRouter:
    """
    Baseline routing classifier using TF-IDF + Logistic Regression
    
    This serves as the baseline to compare against LLM-enhanced routing.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )
        
        # Classifiers
        self.category_classifier = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            class_weight='balanced'
        )
        
        self.priority_classifier = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            class_weight='balanced'
        )
        
        # Label encoders
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        # State
        self.is_fitted = False
        self.categories: List[str] = []
        self.priorities: List[str] = []
    
    def _combine_text(self, subject: str, body: str) -> str:
        """Combine subject and body for classification"""
        # Weight subject higher by repeating
        return f"{subject} {subject} {body}"
    
    def fit(
        self,
        subjects: List[str],
        bodies: List[str],
        categories: List[str],
        priorities: List[str]
    ) -> Dict[str, Any]:
        """
        Train the classifier
        
        Args:
            subjects: List of ticket subjects
            bodies: List of ticket bodies
            categories: List of category labels
            priorities: List of priority labels
            
        Returns:
            Training metrics
        """
        # Combine text
        texts = [
            self._combine_text(s, b) 
            for s, b in zip(subjects, bodies)
        ]
        
        # Fit vectorizer and transform
        X = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        y_category = self.category_encoder.fit_transform(categories)
        y_priority = self.priority_encoder.fit_transform(priorities)
        
        # Store class names
        self.categories = list(self.category_encoder.classes_)
        self.priorities = list(self.priority_encoder.classes_)
        
        # Train classifiers
        self.category_classifier.fit(X, y_category)
        self.priority_classifier.fit(X, y_priority)
        
        self.is_fitted = True
        
        # Calculate training metrics
        category_pred = self.category_classifier.predict(X)
        priority_pred = self.priority_classifier.predict(X)
        
        return {
            'category_accuracy': (category_pred == y_category).mean(),
            'priority_accuracy': (priority_pred == y_priority).mean(),
            'num_samples': len(texts),
            'num_features': X.shape[1],
            'categories': self.categories,
            'priorities': self.priorities
        }
    
    def predict(self, subject: str, body: str) -> RoutingPrediction:
        """
        Predict category and priority for a single ticket
        
        Args:
            subject: Ticket subject
            body: Ticket body
            
        Returns:
            RoutingPrediction with predictions and confidences
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        text = self._combine_text(subject, body)
        X = self.vectorizer.transform([text])
        
        # Category prediction
        category_probs = self.category_classifier.predict_proba(X)[0]
        category_idx = np.argmax(category_probs)
        category = self.categories[category_idx]
        category_confidence = float(category_probs[category_idx])
        
        # Priority prediction
        priority_probs = self.priority_classifier.predict_proba(X)[0]
        priority_idx = np.argmax(priority_probs)
        priority = self.priorities[priority_idx]
        priority_confidence = float(priority_probs[priority_idx])
        
        return RoutingPrediction(
            category=category,
            category_confidence=category_confidence,
            category_probabilities={
                cat: float(prob) 
                for cat, prob in zip(self.categories, category_probs)
            },
            priority=priority,
            priority_confidence=priority_confidence,
            priority_probabilities={
                pri: float(prob) 
                for pri, prob in zip(self.priorities, priority_probs)
            }
        )
    
    def predict_batch(
        self, 
        subjects: List[str], 
        bodies: List[str]
    ) -> List[RoutingPrediction]:
        """Predict for multiple tickets"""
        return [
            self.predict(s, b) 
            for s, b in zip(subjects, bodies)
        ]
    
    def evaluate(
        self,
        subjects: List[str],
        bodies: List[str],
        true_categories: List[str],
        true_priorities: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test set
        
        Returns:
            Evaluation metrics including accuracy, F1, and confusion matrix
        """
        predictions = self.predict_batch(subjects, bodies)
        
        pred_categories = [p.category for p in predictions]
        pred_priorities = [p.priority for p in predictions]
        
        # Category metrics
        category_report = classification_report(
            true_categories, pred_categories,
            output_dict=True, zero_division=0
        )
        category_confusion = confusion_matrix(
            true_categories, pred_categories,
            labels=self.categories
        )
        category_f1_macro = f1_score(
            true_categories, pred_categories,
            average='macro', zero_division=0
        )
        
        # Priority metrics
        priority_report = classification_report(
            true_priorities, pred_priorities,
            output_dict=True, zero_division=0
        )
        priority_confusion = confusion_matrix(
            true_priorities, pred_priorities,
            labels=self.priorities
        )
        priority_f1_macro = f1_score(
            true_priorities, pred_priorities,
            average='macro', zero_division=0
        )
        
        return {
            'category': {
                'accuracy': category_report['accuracy'],
                'f1_macro': category_f1_macro,
                'f1_weighted': category_report['weighted avg']['f1-score'],
                'report': category_report,
                'confusion_matrix': category_confusion.tolist(),
                'labels': self.categories
            },
            'priority': {
                'accuracy': priority_report['accuracy'],
                'f1_macro': priority_f1_macro,
                'f1_weighted': priority_report['weighted avg']['f1-score'],
                'report': priority_report,
                'confusion_matrix': priority_confusion.tolist(),
                'labels': self.priorities
            },
            'num_samples': len(subjects)
        }
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, path / 'vectorizer.joblib')
        
        # Save classifiers
        joblib.dump(self.category_classifier, path / 'category_classifier.joblib')
        joblib.dump(self.priority_classifier, path / 'priority_classifier.joblib')
        
        # Save encoders
        joblib.dump(self.category_encoder, path / 'category_encoder.joblib')
        joblib.dump(self.priority_encoder, path / 'priority_encoder.joblib')
        
        # Save metadata
        metadata = {
            'categories': self.categories,
            'priorities': self.priorities,
            'is_fitted': self.is_fitted
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaselineRouter':
        """Load model from disk"""
        path = Path(path)
        
        router = cls()
        
        # Load vectorizer
        router.vectorizer = joblib.load(path / 'vectorizer.joblib')
        
        # Load classifiers
        router.category_classifier = joblib.load(path / 'category_classifier.joblib')
        router.priority_classifier = joblib.load(path / 'priority_classifier.joblib')
        
        # Load encoders
        router.category_encoder = joblib.load(path / 'category_encoder.joblib')
        router.priority_encoder = joblib.load(path / 'priority_encoder.joblib')
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        router.categories = metadata['categories']
        router.priorities = metadata['priorities']
        router.is_fitted = metadata['is_fitted']
        
        return router


def train_from_database(db_config: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """
    Train baseline router from database
    
    Args:
        db_config: Database connection parameters
        model_path: Path to save trained model
        
    Returns:
        Training and evaluation metrics
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    # Load data from database
    conn = psycopg2.connect(**db_config)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT subject, body, category, priority
            FROM tickets
            WHERE category IS NOT NULL 
              AND priority IS NOT NULL
              AND subject IS NOT NULL
              AND body IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()
    
    if len(rows) < 100:
        raise ValueError(f"Insufficient labeled data: {len(rows)} samples")
    
    # Prepare data
    subjects = [r['subject'] for r in rows]
    bodies = [r['body'] for r in rows]
    categories = [r['category'] for r in rows]
    priorities = [r['priority'] for r in rows]
    
    # Split data
    (train_subjects, test_subjects, 
     train_bodies, test_bodies,
     train_categories, test_categories,
     train_priorities, test_priorities) = train_test_split(
        subjects, bodies, categories, priorities,
        test_size=0.2, random_state=42, stratify=categories
    )
    
    # Train
    router = BaselineRouter()
    train_metrics = router.fit(
        train_subjects, train_bodies,
        train_categories, train_priorities
    )
    
    # Evaluate
    eval_metrics = router.evaluate(
        test_subjects, test_bodies,
        test_categories, test_priorities
    )
    
    # Save model
    router.save(model_path)
    
    return {
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'model_path': model_path
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Router Training')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='localtriage')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-password', default='postgres')
    parser.add_argument('--model-path', default='models/baseline_router')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    results = train_from_database(db_config, args.model_path)
    print(json.dumps(results, indent=2, default=str))
