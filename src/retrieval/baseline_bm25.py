"""
Baseline BM25 Retrieval for LOCALTRIAGE
Sparse retrieval using PostgreSQL full-text search
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class RetrievalResult:
    """Represents a single retrieval result"""
    id: str
    source_type: str  # 'kb_chunk' or 'ticket'
    content: str
    title: Optional[str]
    score: float
    rank: int
    metadata: Dict[str, Any]


class BaselineBM25Retriever:
    """
    Baseline retrieval using PostgreSQL full-text search (BM25-like)
    
    PostgreSQL's ts_rank_cd provides BM25-like ranking with document length normalization.
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        return self
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for full-text search"""
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        # Remove extra whitespace
        query = ' '.join(query.split())
        # Convert to tsquery format with OR between terms
        terms = query.split()
        if not terms:
            return ''
        # Use & (AND) for better precision or | (OR) for better recall
        return ' | '.join(terms)
    
    def search_kb(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Search KB chunks using full-text search
        
        Args:
            query: Search query
            top_k: Number of results to return
            category_filter: Optional category to filter by
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected to database")
        
        tsquery = self._preprocess_query(query)
        if not tsquery:
            return []
        
        sql = """
            WITH ranked_chunks AS (
                SELECT 
                    c.id,
                    c.content,
                    c.chunk_index,
                    a.id as article_id,
                    a.title,
                    a.category,
                    ts_rank_cd(
                        setweight(to_tsvector('english', a.title), 'A') ||
                        setweight(to_tsvector('english', c.content), 'B'),
                        to_tsquery('english', %s),
                        32  -- Normalization: divide by document length
                    ) as score
                FROM kb_chunks c
                JOIN kb_articles a ON c.article_id = a.id
                WHERE a.status = 'published'
                  AND (
                    to_tsvector('english', a.title) @@ to_tsquery('english', %s)
                    OR to_tsvector('english', c.content) @@ to_tsquery('english', %s)
                  )
        """
        
        params = [tsquery, tsquery, tsquery]
        
        if category_filter:
            sql += " AND a.category = %s"
            params.append(category_filter)
        
        sql += """
            )
            SELECT * FROM ranked_chunks
            WHERE score > %s
            ORDER BY score DESC
            LIMIT %s
        """
        params.extend([min_score, top_k])
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        
        results = []
        for rank, row in enumerate(rows, 1):
            results.append(RetrievalResult(
                id=str(row['id']),
                source_type='kb_chunk',
                content=row['content'],
                title=row['title'],
                score=float(row['score']),
                rank=rank,
                metadata={
                    'article_id': str(row['article_id']),
                    'chunk_index': row['chunk_index'],
                    'category': row['category']
                }
            ))
        
        return results
    
    def search_similar_tickets(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        exclude_ticket_id: Optional[str] = None,
        resolved_only: bool = True,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Search for similar past tickets
        
        Args:
            query: Search query (usually the current ticket text)
            top_k: Number of results to return
            category_filter: Optional category to filter by
            exclude_ticket_id: Ticket ID to exclude (current ticket)
            resolved_only: Only return resolved tickets
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected to database")
        
        tsquery = self._preprocess_query(query)
        if not tsquery:
            return []
        
        sql = """
            SELECT 
                id,
                subject,
                body,
                category,
                priority,
                status,
                ts_rank_cd(
                    search_vector,
                    to_tsquery('english', %s),
                    32
                ) as score
            FROM tickets
            WHERE search_vector @@ to_tsquery('english', %s)
        """
        
        params = [tsquery, tsquery]
        
        if resolved_only:
            sql += " AND status IN ('resolved', 'closed')"
        
        if category_filter:
            sql += " AND category = %s"
            params.append(category_filter)
        
        if exclude_ticket_id:
            sql += " AND id != %s"
            params.append(exclude_ticket_id)
        
        sql += """
            AND ts_rank_cd(search_vector, to_tsquery('english', %s), 32) > %s
            ORDER BY score DESC
            LIMIT %s
        """
        params.extend([tsquery, min_score, top_k])
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        
        results = []
        for rank, row in enumerate(rows, 1):
            results.append(RetrievalResult(
                id=str(row['id']),
                source_type='ticket',
                content=f"Subject: {row['subject']}\n\n{row['body']}",
                title=row['subject'],
                score=float(row['score']),
                rank=rank,
                metadata={
                    'category': row['category'],
                    'priority': row['priority'],
                    'status': row['status']
                }
            ))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        kb_top_k: int = 3,
        ticket_top_k: int = 2,
        **kwargs
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Combined search across KB and tickets
        
        Args:
            query: Search query
            kb_top_k: Number of KB results
            ticket_top_k: Number of ticket results
            **kwargs: Additional filters
            
        Returns:
            Dict with 'kb' and 'tickets' result lists
        """
        kb_results = self.search_kb(query, top_k=kb_top_k, **kwargs)
        ticket_results = self.search_similar_tickets(
            query, 
            top_k=ticket_top_k,
            category_filter=kwargs.get('category_filter'),
            exclude_ticket_id=kwargs.get('exclude_ticket_id')
        )
        
        return {
            'kb': kb_results,
            'tickets': ticket_results
        }


class RetrievalEvaluator:
    """Evaluates retrieval performance"""
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """Calculate Recall@k"""
        if not relevant_ids:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / k
    
    @staticmethod
    def mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """Calculate nDCG@k (binary relevance)"""
        import math
        
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 1)
        
        # Calculate ideal DCG
        ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant_ids)) + 1))
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def evaluate_queries(
        self,
        retriever: BaselineBM25Retriever,
        queries: List[str],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retriever on a set of queries
        
        Args:
            retriever: Retriever instance
            queries: List of queries
            relevant_docs: List of relevant doc ID lists per query
            k_values: k values to evaluate
            
        Returns:
            Evaluation metrics
        """
        metrics = {f'recall@{k}': [] for k in k_values}
        metrics.update({f'precision@{k}': [] for k in k_values})
        metrics['mrr'] = []
        metrics.update({f'ndcg@{k}': [] for k in k_values})
        
        for query, relevant in zip(queries, relevant_docs):
            results = retriever.search_kb(query, top_k=max(k_values))
            retrieved_ids = [r.id for r in results]
            
            for k in k_values:
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(retrieved_ids, relevant, k)
                )
                metrics[f'precision@{k}'].append(
                    self.precision_at_k(retrieved_ids, relevant, k)
                )
                metrics[f'ndcg@{k}'].append(
                    self.ndcg_at_k(retrieved_ids, relevant, k)
                )
            
            metrics['mrr'].append(self.mrr(retrieved_ids, relevant))
        
        # Average metrics
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in metrics.items()
        }


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Baseline BM25 Retrieval')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='localtriage')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-password', default='postgres')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--search-type', choices=['kb', 'tickets', 'hybrid'], default='hybrid')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    with BaselineBM25Retriever(db_config) as retriever:
        if args.search_type == 'kb':
            results = retriever.search_kb(args.query, top_k=args.top_k)
        elif args.search_type == 'tickets':
            results = retriever.search_similar_tickets(args.query, top_k=args.top_k)
        else:
            results = retriever.hybrid_search(args.query, kb_top_k=args.top_k, ticket_top_k=args.top_k)
        
        if isinstance(results, dict):
            output = {
                'kb': [{'rank': r.rank, 'score': r.score, 'title': r.title, 'content': r.content[:200]} for r in results['kb']],
                'tickets': [{'rank': r.rank, 'score': r.score, 'title': r.title, 'content': r.content[:200]} for r in results['tickets']]
            }
        else:
            output = [{'rank': r.rank, 'score': r.score, 'title': r.title, 'content': r.content[:200]} for r in results]
        
        print(json.dumps(output, indent=2))
