-- ============================================================
-- LOCALTRIAGE Database Schema
-- Customer Support Triage Platform
-- Version: 1.0
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================
-- CORE TABLES
-- ============================================================

-- Tickets table: stores customer support tickets
CREATE TABLE tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE,
    
    -- Core fields
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    
    -- Classification (ground truth if labeled)
    category VARCHAR(100),
    priority VARCHAR(20) CHECK (priority IN ('P1', 'P2', 'P3', 'P4')),
    
    -- Predicted values
    predicted_category VARCHAR(100),
    predicted_priority VARCHAR(20),
    predicted_confidence FLOAT,
    
    -- Assignment
    assignee_queue VARCHAR(100),
    assigned_agent VARCHAR(255),
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'open' CHECK (status IN ('open', 'pending', 'resolved', 'closed')),
    sla_deadline TIMESTAMP,
    sla_at_risk BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    customer_id VARCHAR(255),
    customer_email VARCHAR(255),
    channel VARCHAR(50),
    language VARCHAR(10) DEFAULT 'en',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    
    -- Full-text search
    search_vector TSVECTOR
);

-- Create indexes for tickets
CREATE INDEX idx_tickets_category ON tickets(category);
CREATE INDEX idx_tickets_priority ON tickets(priority);
CREATE INDEX idx_tickets_status ON tickets(status);
CREATE INDEX idx_tickets_created_at ON tickets(created_at);
CREATE INDEX idx_tickets_search ON tickets USING GIN(search_vector);

-- Update search vector trigger
CREATE OR REPLACE FUNCTION tickets_search_vector_update() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.subject, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.body, '')), 'B');
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tickets_search_update
    BEFORE INSERT OR UPDATE ON tickets
    FOR EACH ROW EXECUTE FUNCTION tickets_search_vector_update();

-- ============================================================
-- KNOWLEDGE BASE TABLES
-- ============================================================

-- KB Articles table: stores knowledge base articles
CREATE TABLE kb_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE,
    
    -- Content
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    
    -- Categorization
    category VARCHAR(100),
    tags TEXT[],
    
    -- Versioning
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,
    parent_id UUID REFERENCES kb_articles(id),
    
    -- Metadata
    author VARCHAR(255),
    source_url TEXT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'published' CHECK (status IN ('draft', 'published', 'archived')),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP,
    
    -- Full-text search
    search_vector TSVECTOR
);

-- Create indexes for kb_articles
CREATE INDEX idx_kb_articles_category ON kb_articles(category);
CREATE INDEX idx_kb_articles_tags ON kb_articles USING GIN(tags);
CREATE INDEX idx_kb_articles_status ON kb_articles(status);
CREATE INDEX idx_kb_articles_search ON kb_articles USING GIN(search_vector);

-- Update search vector trigger for KB
CREATE OR REPLACE FUNCTION kb_articles_search_vector_update() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.summary, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'C');
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER kb_articles_search_update
    BEFORE INSERT OR UPDATE ON kb_articles
    FOR EACH ROW EXECUTE FUNCTION kb_articles_search_vector_update();

-- KB Chunks table: stores chunked KB content for retrieval
CREATE TABLE kb_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID NOT NULL REFERENCES kb_articles(id) ON DELETE CASCADE,
    
    -- Content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    
    -- Metadata
    token_count INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    
    -- Embedding reference (actual vector stored in vector DB)
    embedding_id VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kb_chunks_article ON kb_chunks(article_id);
CREATE INDEX idx_kb_chunks_embedding ON kb_chunks(embedding_id);

-- ============================================================
-- DRAFT AND RESPONSE TABLES
-- ============================================================

-- Response drafts table
CREATE TABLE response_drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    
    -- Draft content
    draft_text TEXT NOT NULL,
    rationale TEXT,
    
    -- Confidence and quality
    confidence VARCHAR(20) CHECK (confidence IN ('high', 'medium', 'low')),
    confidence_score FLOAT,
    
    -- Citations
    citations JSONB DEFAULT '[]',
    follow_up_questions JSONB DEFAULT '[]',
    
    -- Status
    status VARCHAR(50) DEFAULT 'generated' CHECK (status IN ('generated', 'accepted', 'modified', 'rejected')),
    
    -- Model info
    model_name VARCHAR(255),
    model_config JSONB,
    
    -- Performance
    generation_time_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drafts_ticket ON response_drafts(ticket_id);
CREATE INDEX idx_drafts_status ON response_drafts(status);
CREATE INDEX idx_drafts_created ON response_drafts(created_at);

-- Retrieved documents for each draft
CREATE TABLE draft_retrievals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    draft_id UUID NOT NULL REFERENCES response_drafts(id) ON DELETE CASCADE,
    
    -- Source reference
    source_type VARCHAR(50) CHECK (source_type IN ('kb_chunk', 'ticket')),
    source_id UUID NOT NULL,
    
    -- Retrieval info
    similarity_score FLOAT,
    rank INTEGER,
    retrieval_method VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retrievals_draft ON draft_retrievals(draft_id);

-- ============================================================
-- FEEDBACK AND EVALUATION TABLES
-- ============================================================

-- Agent feedback on drafts
CREATE TABLE draft_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    draft_id UUID NOT NULL REFERENCES response_drafts(id) ON DELETE CASCADE,
    
    -- Rating
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    is_helpful BOOLEAN,
    
    -- Qualitative feedback
    feedback_text TEXT,
    correction_text TEXT,
    
    -- Detailed scores
    correctness_score INTEGER CHECK (correctness_score BETWEEN 1 AND 5),
    completeness_score INTEGER CHECK (completeness_score BETWEEN 1 AND 5),
    tone_score INTEGER CHECK (tone_score BETWEEN 1 AND 5),
    
    -- Modification tracking
    edit_distance INTEGER,
    modification_percentage FLOAT,
    
    -- Metadata
    agent_id VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_draft ON draft_feedback(draft_id);
CREATE INDEX idx_feedback_rating ON draft_feedback(rating);

-- Routing evaluation results
CREATE TABLE routing_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    
    -- Evaluation
    predicted_category VARCHAR(100),
    actual_category VARCHAR(100),
    is_correct BOOLEAN,
    
    predicted_priority VARCHAR(20),
    actual_priority VARCHAR(20),
    priority_correct BOOLEAN,
    
    -- Confidence
    confidence_score FLOAT,
    
    -- Model info
    model_name VARCHAR(255),
    evaluation_run_id UUID,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_routing_eval_ticket ON routing_evaluations(ticket_id);
CREATE INDEX idx_routing_eval_correct ON routing_evaluations(is_correct);

-- ============================================================
-- LOGGING AND MONITORING TABLES
-- ============================================================

-- Request logs for all API calls
CREATE TABLE request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Request info
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10),
    request_body JSONB,
    
    -- Response info
    response_status INTEGER,
    response_body JSONB,
    
    -- Performance
    latency_ms INTEGER,
    
    -- Breakdown
    embedding_time_ms INTEGER,
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    
    -- Model info
    model_name VARCHAR(255),
    model_config JSONB,
    
    -- Tokens
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Metadata
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_logs_endpoint ON request_logs(endpoint);
CREATE INDEX idx_logs_created ON request_logs(created_at);
CREATE INDEX idx_logs_latency ON request_logs(latency_ms);

-- Evaluation runs table
CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Run info
    run_name VARCHAR(255),
    run_type VARCHAR(50) CHECK (run_type IN ('routing', 'retrieval', 'draft', 'full')),
    
    -- Configuration
    config JSONB,
    model_name VARCHAR(255),
    
    -- Results
    metrics JSONB,
    
    -- Status
    status VARCHAR(50) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    error_message TEXT,
    
    -- Timestamps
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_eval_runs_type ON evaluation_runs(run_type);
CREATE INDEX idx_eval_runs_status ON evaluation_runs(status);

-- ============================================================
-- ANALYTICS TABLES
-- ============================================================

-- Daily metrics aggregation
CREATE TABLE daily_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    
    -- Volume
    total_tickets INTEGER DEFAULT 0,
    tickets_by_category JSONB DEFAULT '{}',
    tickets_by_priority JSONB DEFAULT '{}',
    
    -- Performance
    avg_routing_accuracy FLOAT,
    avg_draft_acceptance FLOAT,
    avg_latency_ms FLOAT,
    
    -- SLA
    sla_breaches INTEGER DEFAULT 0,
    sla_at_risk INTEGER DEFAULT 0,
    
    -- Quality
    avg_draft_rating FLOAT,
    hallucination_count INTEGER DEFAULT 0,
    
    -- Cost
    total_tokens INTEGER DEFAULT 0,
    estimated_cost FLOAT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_daily_metrics_date ON daily_metrics(date);

-- Emerging topics/clusters
CREATE TABLE topic_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Cluster info
    cluster_name VARCHAR(255),
    description TEXT,
    
    -- Statistics
    ticket_count INTEGER DEFAULT 0,
    exemplar_tickets UUID[],
    
    -- Keywords
    keywords TEXT[],
    
    -- Trend
    trend VARCHAR(20) CHECK (trend IN ('emerging', 'stable', 'declining')),
    first_seen DATE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_clusters_trend ON topic_clusters(trend);
CREATE INDEX idx_clusters_active ON topic_clusters(is_active);

-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- View: Ticket with latest draft
CREATE VIEW tickets_with_drafts AS
SELECT 
    t.*,
    d.id as draft_id,
    d.draft_text,
    d.confidence,
    d.citations,
    d.status as draft_status
FROM tickets t
LEFT JOIN LATERAL (
    SELECT * FROM response_drafts rd 
    WHERE rd.ticket_id = t.id 
    ORDER BY rd.created_at DESC 
    LIMIT 1
) d ON true;

-- View: Daily volume by category
CREATE VIEW daily_volume_by_category AS
SELECT 
    DATE(created_at) as date,
    category,
    COUNT(*) as ticket_count
FROM tickets
GROUP BY DATE(created_at), category
ORDER BY date DESC, ticket_count DESC;

-- View: Routing accuracy by category
CREATE VIEW routing_accuracy_by_category AS
SELECT 
    actual_category as category,
    COUNT(*) as total,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
    ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct
FROM routing_evaluations
GROUP BY actual_category;

-- ============================================================
-- SAMPLE DATA INSERTION FUNCTIONS
-- ============================================================

-- Function to insert sample ticket
CREATE OR REPLACE FUNCTION insert_sample_ticket(
    p_subject TEXT,
    p_body TEXT,
    p_category VARCHAR(100) DEFAULT NULL,
    p_priority VARCHAR(20) DEFAULT 'P3'
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO tickets (subject, body, category, priority)
    VALUES (p_subject, p_body, p_category, p_priority)
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to insert sample KB article
CREATE OR REPLACE FUNCTION insert_sample_kb_article(
    p_title TEXT,
    p_content TEXT,
    p_category VARCHAR(100) DEFAULT NULL,
    p_tags TEXT[] DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO kb_articles (title, content, category, tags, status, published_at)
    VALUES (p_title, p_content, p_category, p_tags, 'published', CURRENT_TIMESTAMP)
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- GRANTS (adjust as needed for your setup)
-- ============================================================

-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO localtriage_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO localtriage_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO localtriage_app;
