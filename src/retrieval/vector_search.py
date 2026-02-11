"""
Vector Search Module for LOCALTRIAGE
Dense retrieval using embeddings and Qdrant/FAISS
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os
import json
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class VectorSearchResult:
    """Represents a vector search result"""
    id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class EmbeddingModel:
    """Wrapper for embedding model"""
    
    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cpu",
        batch_size: int = 32
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._dimension = None
    
    def load(self):
        """Load the model.
        
        Uses CPU with float32 by default. For GPU with half-precision,
        ensure your CUDA driver supports BF16/FP16 CUBLAS operations.
        """
        import torch
        # Determine effective device: try half-precision GPU, fallback to CPU
        effective_device = self.device
        dtype = torch.bfloat16
        if effective_device != "cpu":
            try:
                # Quick CUBLAS sanity check for half-precision
                a = torch.randn(2, 128, dtype=torch.bfloat16, device=effective_device)
                b = torch.randn(128, 128, dtype=torch.bfloat16, device=effective_device)
                _ = a @ b
                del a, b
                torch.cuda.empty_cache()
            except RuntimeError:
                import logging
                logging.getLogger(__name__).warning(
                    "GPU half-precision CUBLAS unavailable, falling back to CPU for embeddings"
                )
                effective_device = "cpu"
                dtype = torch.float32
        else:
            dtype = torch.float32

        self.model = SentenceTransformer(
            self.model_name,
            device=effective_device,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self._dimension = self.model.get_sentence_embedding_dimension()
        return self
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            if self.model is None:
                self.load()
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search
    
    Simple in-memory store suitable for smaller datasets (<100k vectors)
    """
    
    def __init__(self, dimension: int):
        import faiss
        
        self.dimension = dimension
        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add documents with their embeddings
        
        Args:
            embeddings: Numpy array of embeddings (n_docs x dimension)
            documents: List of document dicts with 'id', 'content', 'metadata'
            
        Returns:
            List of internal IDs assigned
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
        ids = []
        for doc in documents:
            doc_id = self._next_id
            self.documents[doc_id] = doc
            ids.append(doc_id)
            self._next_id += 1
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding (1 x dimension or dimension,)
            top_k: Number of results
            filter_fn: Optional function to filter results
            
        Returns:
            List of VectorSearchResult
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        
        # Search more if filtering
        search_k = top_k * 3 if filter_fn else top_k
        
        scores, indices = self.index.search(query_embedding, min(search_k, len(self.documents)))
        
        results = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            doc = self.documents.get(int(idx))
            if doc is None:
                continue
            
            if filter_fn and not filter_fn(doc):
                continue
            
            results.append(VectorSearchResult(
                id=doc.get('id', str(idx)),
                content=doc.get('content', ''),
                score=float(score),
                rank=rank,
                metadata=doc.get('metadata', {})
            ))
            rank += 1
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: str):
        """Save index and documents to disk"""
        import faiss
        
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        with open(os.path.join(path, 'documents.json'), 'w') as f:
            json.dump({
                'documents': {str(k): v for k, v in self.documents.items()},
                'next_id': self._next_id,
                'dimension': self.dimension
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'FAISSVectorStore':
        """Load index and documents from disk"""
        import faiss
        
        with open(os.path.join(path, 'documents.json'), 'r') as f:
            data = json.load(f)
        
        store = cls(data['dimension'])
        store.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        store.documents = {int(k): v for k, v in data['documents'].items()}
        store._next_id = data['next_id']
        
        return store


class QdrantVectorStore:
    """
    Qdrant-based vector store for production use
    
    Supports filtering, persistence, and horizontal scaling
    """
    
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        self.client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=prefer_grpc
        )
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents with embeddings"""
        from qdrant_client.models import PointStruct
        
        points = []
        ids = []
        
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            original_id = doc.get('id', str(uuid.uuid4()))
            # Convert string IDs to UUIDs (Qdrant requires UUID or int)
            if isinstance(original_id, str) and not self._is_valid_uuid(original_id):
                # Generate deterministic UUID from string ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, original_id))
            else:
                point_id = original_id
            ids.append(original_id)  # Return original ID
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    'content': doc.get('content', ''),
                    'original_id': original_id,  # Store original ID in payload
                    **doc.get('metadata', {})
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    def _is_valid_uuid(self, val: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)
        
        # Ensure query is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Use query_points for newer qdrant-client versions (1.7+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        return [
            VectorSearchResult(
                id=str(hit.id),
                content=hit.payload.get('content', ''),
                score=hit.score,
                rank=i + 1,
                metadata={k: v for k, v in hit.payload.items() if k != 'content'}
            )
            for i, hit in enumerate(results.points)
        ]
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval
    
    Uses Reciprocal Rank Fusion (RRF) to combine results
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: Union[FAISSVectorStore, QdrantVectorStore],
        bm25_retriever,  # BaselineBM25Retriever
        alpha: float = 0.6  # Weight for dense retrieval
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # Higher = more weight on dense
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,  # RRF constant
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            rrf_k: RRF constant (higher = more weight on lower ranks)
            **kwargs: Additional arguments for retrievers
            
        Returns:
            Combined and re-ranked results
        """
        # Get dense results
        query_embedding = self.embedding_model.encode(query)
        dense_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2  # Retrieve more for fusion
        )
        
        # Get sparse results
        sparse_results = self.bm25_retriever.search_kb(
            query,
            top_k=top_k * 2
        )
        
        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, VectorSearchResult] = {}
        
        # Dense contribution
        for result in dense_results:
            rrf_score = self.alpha / (rrf_k + result.rank)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + rrf_score
            doc_map[result.id] = result
        
        # Sparse contribution
        for result in sparse_results:
            rrf_score = (1 - self.alpha) / (rrf_k + result.rank)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + rrf_score
            if result.id not in doc_map:
                doc_map[result.id] = VectorSearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score,
                    rank=0,
                    metadata=result.metadata
                )
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        results = []
        for rank, doc_id in enumerate(sorted_ids[:top_k], 1):
            result = doc_map[doc_id]
            results.append(VectorSearchResult(
                id=result.id,
                content=result.content,
                score=rrf_scores[doc_id],  # Use RRF score
                rank=rank,
                metadata=result.metadata
            ))
        
        return results


def index_kb_chunks(
    db_config: Dict[str, Any],
    embedding_model: EmbeddingModel,
    vector_store: Union[FAISSVectorStore, QdrantVectorStore],
    batch_size: int = 100
) -> int:
    """
    Index all KB chunks from database into vector store
    
    Args:
        db_config: Database connection config
        embedding_model: Embedding model
        vector_store: Vector store to index into
        batch_size: Batch size for processing
        
    Returns:
        Number of chunks indexed
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    conn = psycopg2.connect(**db_config)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                c.id, c.content, c.chunk_index,
                a.id as article_id, a.title, a.category
            FROM kb_chunks c
            JOIN kb_articles a ON c.article_id = a.id
            WHERE a.status = 'published'
        """)
        rows = cur.fetchall()
    
    conn.close()
    
    total_indexed = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        
        # Prepare texts and documents
        texts = [f"{row['title']}\n\n{row['content']}" for row in batch]
        documents = [
            {
                'id': str(row['id']),
                'content': row['content'],
                'metadata': {
                    'article_id': str(row['article_id']),
                    'title': row['title'],
                    'category': row['category'],
                    'chunk_index': row['chunk_index']
                }
            }
            for row in batch
        ]
        
        # Encode and add
        embeddings = embedding_model.encode(texts)
        vector_store.add(embeddings, documents)
        
        total_indexed += len(batch)
        print(f"Indexed {total_indexed}/{len(rows)} chunks")
    
    return total_indexed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Search')
    parser.add_argument('command', choices=['index', 'search'])
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='localtriage')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-password', default='postgres')
    parser.add_argument('--index-path', default='data/vector_index')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--use-qdrant', action='store_true')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    embedding_model = EmbeddingModel()
    embedding_model.load()
    
    if args.command == 'index':
        if args.use_qdrant:
            vector_store = QdrantVectorStore('kb_chunks', embedding_model.dimension)
        else:
            vector_store = FAISSVectorStore(embedding_model.dimension)
        
        count = index_kb_chunks(db_config, embedding_model, vector_store)
        print(f"Indexed {count} chunks")
        
        if not args.use_qdrant:
            vector_store.save(args.index_path)
            print(f"Saved index to {args.index_path}")
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search")
            exit(1)
        
        if args.use_qdrant:
            vector_store = QdrantVectorStore('kb_chunks', embedding_model.dimension)
        else:
            vector_store = FAISSVectorStore.load(args.index_path)
        
        query_embedding = embedding_model.encode(args.query)
        results = vector_store.search(query_embedding, top_k=args.top_k)
        
        for result in results:
            print(f"\n[{result.rank}] Score: {result.score:.4f}")
            print(f"Content: {result.content[:200]}...")
            print(f"Metadata: {result.metadata}")
