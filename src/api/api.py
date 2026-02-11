"""
FastAPI Application for LOCALTRIAGE
REST API endpoints for ticket triage, drafting, and analytics
"""

import os
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class TicketInput(BaseModel):
    """Input model for ticket triage and drafting"""
    ticket_id: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=10000)
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None


class TriageResponse(BaseModel):
    """Response model for ticket triage"""
    ticket_id: str
    category: str
    category_confidence: float
    category_probabilities: Dict[str, float]
    priority: str
    priority_confidence: float
    priority_probabilities: Dict[str, float]
    sla_risk: bool
    suggested_queue: Optional[str]
    explanation: Optional[str]
    processing_time_ms: int


class DraftRequest(BaseModel):
    """Request model for draft generation"""
    ticket_id: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=10000)
    category: Optional[str] = None
    priority: Optional[str] = None
    use_llm: bool = True


class DraftResponse(BaseModel):
    """Response model for draft generation"""
    draft_id: str
    ticket_id: str
    draft_text: str
    rationale: str
    confidence: str
    confidence_score: float
    citations: List[Dict[str, Any]]
    follow_up_questions: List[str]
    model_name: str
    generation_time_ms: int
    retrieval_time_ms: int
    total_time_ms: int


class SimilarTicketsRequest(BaseModel):
    """Request model for similar tickets search"""
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    category_filter: Optional[str] = None


class SimilarTicketResult(BaseModel):
    """Single similar ticket result"""
    ticket_id: str
    subject: str
    body_preview: str
    category: Optional[str]
    priority: Optional[str]
    similarity_score: float
    status: str


class FeedbackInput(BaseModel):
    """Input model for draft feedback"""
    draft_id: str
    rating: int = Field(..., ge=1, le=5)
    is_helpful: bool
    feedback_text: Optional[str] = None
    correction_text: Optional[str] = None
    agent_id: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response model for system metrics"""
    period: str
    total_tickets: int
    total_drafts: int
    avg_routing_accuracy: Optional[float]
    avg_draft_rating: Optional[float]
    avg_latency_ms: Optional[float]
    tickets_by_category: Dict[str, int]
    tickets_by_priority: Dict[str, int]
    sla_compliance_rate: Optional[float]


# =============================================================================
# Application State & Dependencies
# =============================================================================

class AppState:
    """Application state container"""
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'localtriage'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        self.llm_base_url = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
        self.llm_model = os.getenv('LLM_MODEL', 'qwen3:32b')
        self.use_llm = os.getenv('USE_LLM', 'true').lower() == 'true'
        
        # Lazy-loaded components
        self._router = None
        self._retriever = None
        self._drafter = None
        self._embedding_model = None
        self._vector_store = None
    
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    @property
    def router(self):
        """Lazy-load router"""
        if self._router is None:
            from src.triage.baseline_classifier import BaselineRouter
            model_path = os.getenv('ROUTER_MODEL_PATH', 'models/baseline_router')
            if os.path.exists(model_path):
                self._router = BaselineRouter.load(model_path)
            else:
                self._router = BaselineRouter()
        return self._router
    
    @property
    def retriever(self):
        """Lazy-load retriever"""
        if self._retriever is None:
            from src.retrieval.baseline_bm25 import BaselineBM25Retriever
            self._retriever = BaselineBM25Retriever(self.db_config)
            self._retriever.connect()
        return self._retriever
    
    @property
    def drafter(self):
        """Lazy-load drafter"""
        if self._drafter is None:
            if self.use_llm:
                from src.rag.drafter import LLMClient, RAGDrafter
                llm = LLMClient(
                    base_url=self.llm_base_url,
                    model_name=self.llm_model
                )
                self._drafter = RAGDrafter(llm, self.retriever)
            else:
                from src.rag.drafter import BaselineTemplateResponder
                self._drafter = BaselineTemplateResponder()
        return self._drafter


state = AppState()


# =============================================================================
# Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting LOCALTRIAGE API...")
    logger.info(f"LLM Mode: {'Enabled' if state.use_llm else 'Disabled (Baseline)'}")
    yield
    logger.info("Shutting down LOCALTRIAGE API...")
    if state._retriever:
        state._retriever.close()


app = FastAPI(
    title="LOCALTRIAGE API",
    description="Local LLM Customer Support Triage + Response Drafting Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Background Tasks
# =============================================================================

def log_request(
    endpoint: str,
    request_body: dict,
    response_body: dict,
    latency_ms: int,
    model_name: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
):
    """Log request to database"""
    try:
        conn = state.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO request_logs (
                    endpoint, request_body, response_body, latency_ms,
                    model_name, prompt_tokens, completion_tokens
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                endpoint,
                psycopg2.extras.Json(request_body),
                psycopg2.extras.Json(response_body),
                latency_ms,
                model_name,
                prompt_tokens,
                completion_tokens
            ))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log request: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    checks = {
        'api': 'healthy',
        'database': 'unknown',
        'llm': 'unknown'
    }
    
    # Check database
    try:
        conn = state.get_db_connection()
        conn.close()
        checks['database'] = 'healthy'
    except Exception as e:
        checks['database'] = f'unhealthy: {str(e)}'
    
    # Check LLM (if enabled)
    if state.use_llm:
        try:
            from src.rag.drafter import LLMClient
            llm = LLMClient(base_url=state.llm_base_url)
            checks['llm'] = 'healthy' if llm.health_check() else 'unhealthy'
        except:
            checks['llm'] = 'unhealthy'
    else:
        checks['llm'] = 'disabled'
    
    return checks


@app.post("/triage", response_model=TriageResponse)
async def triage_ticket(
    ticket: TicketInput,
    background_tasks: BackgroundTasks
):
    """
    Triage a ticket: predict category, priority, and SLA risk
    """
    start_time = time.time()
    ticket_id = ticket.ticket_id or str(uuid.uuid4())
    
    try:
        # Get routing prediction
        if state.router.is_fitted:
            prediction = state.router.predict(ticket.subject, ticket.body)
        else:
            # Fallback if model not trained
            prediction = type('obj', (object,), {
                'category': 'General',
                'category_confidence': 0.5,
                'category_probabilities': {'General': 0.5},
                'priority': 'P3',
                'priority_confidence': 0.5,
                'priority_probabilities': {'P3': 0.5}
            })()
        
        # Determine SLA risk (P1/P2 are higher risk)
        sla_risk = prediction.priority in ['P1', 'P2'] and prediction.priority_confidence > 0.6
        
        # Suggest queue based on category
        queue_mapping = {
            'Billing': 'billing-team',
            'Technical': 'tech-support',
            'Account': 'account-team',
            'Shipping': 'logistics',
            'Returns': 'returns-team',
            'Product': 'product-support',
            'Escalation': 'senior-support',
        }
        suggested_queue = queue_mapping.get(prediction.category, 'general-support')
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response = TriageResponse(
            ticket_id=ticket_id,
            category=prediction.category,
            category_confidence=prediction.category_confidence,
            category_probabilities=prediction.category_probabilities,
            priority=prediction.priority,
            priority_confidence=prediction.priority_confidence,
            priority_probabilities=prediction.priority_probabilities,
            sla_risk=sla_risk,
            suggested_queue=suggested_queue,
            explanation=None,  # TODO: Add LLM explanation
            processing_time_ms=processing_time_ms
        )
        
        # Persist ticket to database
        try:
            conn = state.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tickets (
                        id, subject, body, predicted_category, predicted_priority,
                        predicted_confidence, assignee_queue, status, customer_email
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s)
                    ON CONFLICT (id) DO UPDATE SET
                        predicted_category = EXCLUDED.predicted_category,
                        predicted_priority = EXCLUDED.predicted_priority,
                        predicted_confidence = EXCLUDED.predicted_confidence,
                        assignee_queue = EXCLUDED.assignee_queue,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    ticket_id,
                    ticket.subject,
                    ticket.body,
                    prediction.category,
                    prediction.priority,
                    prediction.category_confidence,
                    suggested_queue,
                    ticket.customer_email
                ))
                # Update search vector
                cur.execute("""
                    UPDATE tickets SET search_vector =
                        setweight(to_tsvector('english', COALESCE(subject, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(body, '')), 'B')
                    WHERE id = %s
                """, (ticket_id,))
                conn.commit()
            conn.close()
        except Exception as db_err:
            logger.error(f"Failed to persist ticket: {db_err}")

        # Log request
        background_tasks.add_task(
            log_request,
            '/triage',
            ticket.model_dump(),
            response.model_dump(),
            processing_time_ms
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Triage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/draft", response_model=DraftResponse)
async def generate_draft(
    request: DraftRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a draft response for a ticket using RAG
    """
    start_time = time.time()
    ticket_id = request.ticket_id or str(uuid.uuid4())
    
    try:
        if request.use_llm and state.use_llm:
            # Use LLM-based drafter
            draft = state.drafter.generate_draft(
                ticket_id=ticket_id,
                subject=request.subject,
                body=request.body,
                category=request.category,
                priority=request.priority
            )
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            response = DraftResponse(
                draft_id=draft.draft_id,
                ticket_id=ticket_id,
                draft_text=draft.draft_text,
                rationale=draft.rationale,
                confidence=draft.confidence,
                confidence_score=draft.confidence_score,
                citations=[c.__dict__ for c in draft.citations],
                follow_up_questions=draft.follow_up_questions,
                model_name=draft.model_name,
                generation_time_ms=draft.generation_time_ms,
                retrieval_time_ms=draft.retrieval_time_ms,
                total_time_ms=total_time_ms
            )
            
            # Persist draft to database
            try:
                conn = state.get_db_connection()
                with conn.cursor() as cur:
                    # Ensure the ticket exists first
                    cur.execute("SELECT id FROM tickets WHERE id = %s", (ticket_id,))
                    if cur.fetchone():
                        cur.execute("""
                            INSERT INTO response_drafts (
                                ticket_id, draft_text, rationale, confidence,
                                confidence_score, citations, follow_up_questions,
                                model_name, generation_time_ms, prompt_tokens,
                                completion_tokens
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            ticket_id,
                            draft.draft_text,
                            draft.rationale,
                            draft.confidence,
                            draft.confidence_score,
                            psycopg2.extras.Json([c.__dict__ for c in draft.citations]),
                            psycopg2.extras.Json(draft.follow_up_questions),
                            draft.model_name,
                            draft.generation_time_ms,
                            draft.prompt_tokens,
                            draft.completion_tokens
                        ))
                        conn.commit()
                conn.close()
            except Exception as db_err:
                logger.error(f"Failed to persist draft: {db_err}")

            # Log request
            background_tasks.add_task(
                log_request,
                '/draft',
                request.model_dump(),
                response.model_dump(),
                total_time_ms,
                draft.model_name,
                draft.prompt_tokens,
                draft.completion_tokens
            )
        else:
            # Use baseline template responder
            from src.rag.drafter import BaselineTemplateResponder
            template_responder = BaselineTemplateResponder()
            draft_text = template_responder.generate_response(category=request.category)
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            response = DraftResponse(
                draft_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                draft_text=draft_text,
                rationale="Generated using template-based baseline",
                confidence='medium',
                confidence_score=0.5,
                citations=[],
                follow_up_questions=[],
                model_name='baseline-template',
                generation_time_ms=total_time_ms,
                retrieval_time_ms=0,
                total_time_ms=total_time_ms
            )
        
        return response
    
    except Exception as e:
        logger.error(f"Draft error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=List[SimilarTicketResult])
async def find_similar_tickets(request: SimilarTicketsRequest):
    """
    Find similar past tickets for context
    """
    try:
        results = state.retriever.search_similar_tickets(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
            resolved_only=False
        )
        
        return [
            SimilarTicketResult(
                ticket_id=r.id,
                subject=r.title or 'Untitled',
                body_preview=r.content[:300] + '...' if len(r.content) > 300 else r.content,
                category=r.metadata.get('category'),
                priority=r.metadata.get('priority'),
                similarity_score=r.score,
                status=r.metadata.get('status', 'unknown')
            )
            for r in results
        ]
    
    except Exception as e:
        logger.error(f"Similar tickets error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """
    Submit feedback on a draft response
    """
    try:
        conn = state.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO draft_feedback (
                    draft_id, rating, is_helpful, feedback_text,
                    correction_text, agent_id
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                feedback.draft_id,
                feedback.rating,
                feedback.is_helpful,
                feedback.feedback_text,
                feedback.correction_text,
                feedback.agent_id
            ))
            feedback_id = cur.fetchone()[0]
            conn.commit()
        conn.close()
        
        return {"status": "success", "feedback_id": str(feedback_id)}
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    period: str = Query(default="day", pattern="^(day|week|month)$")
):
    """
    Get system metrics and analytics
    """
    try:
        # Calculate date range
        now = datetime.now()
        if period == 'day':
            start_date = now - timedelta(days=1)
        elif period == 'week':
            start_date = now - timedelta(weeks=1)
        else:
            start_date = now - timedelta(days=30)
        
        conn = state.get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total tickets (from DB + triage requests in period)
            cur.execute("""
                SELECT COUNT(*) as count
                FROM tickets
                WHERE created_at >= %s
            """, (start_date,))
            total_tickets = cur.fetchone()['count']
            
            # If no tickets in period, count from request_logs as fallback
            if total_tickets == 0:
                cur.execute("""
                    SELECT COUNT(*) as count
                    FROM request_logs
                    WHERE created_at >= %s AND endpoint = '/triage'
                """, (start_date,))
                total_tickets = cur.fetchone()['count']
            
            # Total drafts (from DB + draft requests in period)
            cur.execute("""
                SELECT COUNT(*) as count
                FROM response_drafts
                WHERE created_at >= %s
            """, (start_date,))
            total_drafts = cur.fetchone()['count']
            
            # If no drafts in period, count from request_logs as fallback
            if total_drafts == 0:
                cur.execute("""
                    SELECT COUNT(*) as count
                    FROM request_logs
                    WHERE created_at >= %s AND endpoint = '/draft'
                """, (start_date,))
                total_drafts = cur.fetchone()['count']
            
            # Average draft rating
            cur.execute("""
                SELECT AVG(rating) as avg_rating
                FROM draft_feedback
                WHERE created_at >= %s
            """, (start_date,))
            avg_rating = cur.fetchone()['avg_rating']
            
            # Average latency
            cur.execute("""
                SELECT AVG(latency_ms) as avg_latency
                FROM request_logs
                WHERE created_at >= %s AND endpoint = '/draft'
            """, (start_date,))
            avg_latency = cur.fetchone()['avg_latency']
            
            # Tickets by category - use predicted_category if category is null
            cur.execute("""
                SELECT COALESCE(category, predicted_category) as cat, COUNT(*) as count
                FROM tickets
                WHERE created_at >= %s AND COALESCE(category, predicted_category) IS NOT NULL
                GROUP BY cat
            """, (start_date,))
            by_category = {row['cat']: row['count'] for row in cur.fetchall()}
            
            # If no tickets in DB for period, extract from request_logs
            if not by_category:
                cur.execute("""
                    SELECT
                        response_body->>'category' as cat,
                        COUNT(*) as count
                    FROM request_logs
                    WHERE created_at >= %s AND endpoint = '/triage'
                        AND response_body->>'category' IS NOT NULL
                    GROUP BY cat
                """, (start_date,))
                by_category = {row['cat']: row['count'] for row in cur.fetchall()}
            
            # Tickets by priority - use predicted_priority if priority is null
            cur.execute("""
                SELECT COALESCE(priority, predicted_priority) as prio, COUNT(*) as count
                FROM tickets
                WHERE created_at >= %s AND COALESCE(priority, predicted_priority) IS NOT NULL
                GROUP BY prio
            """, (start_date,))
            by_priority = {row['prio']: row['count'] for row in cur.fetchall()}
            
            # If no tickets in DB for period, extract from request_logs
            if not by_priority:
                cur.execute("""
                    SELECT
                        response_body->>'priority' as prio,
                        COUNT(*) as count
                    FROM request_logs
                    WHERE created_at >= %s AND endpoint = '/triage'
                        AND response_body->>'priority' IS NOT NULL
                    GROUP BY prio
                """, (start_date,))
                by_priority = {row['prio']: row['count'] for row in cur.fetchall()}
        
        conn.close()
        
        return MetricsResponse(
            period=period,
            total_tickets=total_tickets,
            total_drafts=total_drafts,
            avg_routing_accuracy=None,  # TODO: Calculate from evaluations
            avg_draft_rating=float(avg_rating) if avg_rating else None,
            avg_latency_ms=float(avg_latency) if avg_latency else None,
            tickets_by_category=by_category,
            tickets_by_priority=by_priority,
            sla_compliance_rate=None  # TODO: Calculate
        )
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets")
async def list_tickets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    category: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List tickets with pagination and filtering
    """
    try:
        offset = (page - 1) * page_size
        
        conn = state.get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build query
            query = "SELECT * FROM tickets WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            if priority:
                query += " AND priority = %s"
                params.append(priority)
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([page_size, offset])
            
            cur.execute(query, params)
            tickets = cur.fetchall()
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM tickets WHERE 1=1"
            count_params = []
            if category:
                count_query += " AND category = %s"
                count_params.append(category)
            if priority:
                count_query += " AND priority = %s"
                count_params.append(priority)
            if status:
                count_query += " AND status = %s"
                count_params.append(status)
            
            cur.execute(count_query, count_params)
            total = cur.fetchone()['count']
        
        conn.close()
        
        return {
            'tickets': [dict(t) for t in tickets],
            'page': page,
            'page_size': page_size,
            'total': total,
            'total_pages': (total + page_size - 1) // page_size
        }
    
    except Exception as e:
        logger.error(f"List tickets error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """
    Get a single ticket with its drafts
    """
    try:
        conn = state.get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get ticket
            cur.execute("SELECT * FROM tickets WHERE id = %s", (ticket_id,))
            ticket = cur.fetchone()
            
            if not ticket:
                raise HTTPException(status_code=404, detail="Ticket not found")
            
            # Get drafts
            cur.execute("""
                SELECT * FROM response_drafts
                WHERE ticket_id = %s
                ORDER BY created_at DESC
            """, (ticket_id,))
            drafts = cur.fetchall()
        
        conn.close()
        
        return {
            'ticket': dict(ticket),
            'drafts': [dict(d) for d in drafts]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get ticket error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run Application
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
