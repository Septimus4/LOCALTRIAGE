"""
Streamlit Dashboard for LOCALTRIAGE
Agent-facing UI for ticket triage, drafting, and analytics
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8080')

st.set_page_config(
    page_title="LOCALTRIAGE - Support Triage Platform",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .citation-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .priority-p1 { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; }
    .priority-p2 { background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; }
    .priority-p3 { background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; }
    .priority-p4 { background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client Functions
# =============================================================================

def api_request(endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == 'GET':
            response = requests.get(url, timeout=60)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=120)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}


def get_health_status() -> Dict:
    """Get API health status"""
    return api_request('/health')


def triage_ticket(subject: str, body: str) -> Dict:
    """Triage a ticket"""
    return api_request('/triage', 'POST', {
        'subject': subject,
        'body': body
    })


def generate_draft(subject: str, body: str, category: str = None, priority: str = None) -> Dict:
    """Generate draft response"""
    return api_request('/draft', 'POST', {
        'subject': subject,
        'body': body,
        'category': category,
        'priority': priority,
        'use_llm': True
    })


def find_similar_tickets(query: str, top_k: int = 5) -> List[Dict]:
    """Find similar tickets"""
    return api_request('/similar', 'POST', {
        'query': query,
        'top_k': top_k
    })


def submit_feedback(draft_id: str, rating: int, is_helpful: bool, feedback_text: str = None) -> Dict:
    """Submit feedback on draft"""
    return api_request('/feedback', 'POST', {
        'draft_id': draft_id,
        'rating': rating,
        'is_helpful': is_helpful,
        'feedback_text': feedback_text
    })


def get_metrics(period: str = 'day') -> Dict:
    """Get system metrics"""
    return api_request(f'/metrics?period={period}')


def list_tickets(page: int = 1, page_size: int = 20, **filters) -> Dict:
    """List tickets with pagination"""
    params = f'?page={page}&page_size={page_size}'
    for key, value in filters.items():
        if value:
            params += f'&{key}={value}'
    return api_request(f'/tickets{params}')


# =============================================================================
# Sidebar Navigation
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and status"""
    with st.sidebar:
        st.markdown("# 🎫 LOCALTRIAGE")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🆕 New Ticket", "📋 Ticket List", "📊 Analytics", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### System Status")
        health = get_health_status()
        
        if health:
            status_colors = {'healthy': '🟢', 'unhealthy': '🔴', 'disabled': '⚪', 'unknown': '🟡'}
            
            for component, status in health.items():
                color = status_colors.get(status.split(':')[0] if ':' in str(status) else status, '🟡')
                st.markdown(f"{color} **{component.title()}**: {status}")
        else:
            st.error("Unable to connect to API")
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        metrics = get_metrics('day')
        if metrics:
            st.metric("Today's Tickets", metrics.get('total_tickets', 0))
            st.metric("Drafts Generated", metrics.get('total_drafts', 0))
            if metrics.get('avg_draft_rating'):
                st.metric("Avg Rating", f"{metrics['avg_draft_rating']:.1f}/5")
        
        return page


# =============================================================================
# New Ticket Page
# =============================================================================

def render_new_ticket_page():
    """Render new ticket triage and drafting page"""
    st.markdown('<div class="main-header">🆕 New Ticket Triage & Draft</div>', unsafe_allow_html=True)
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Ticket Details")
        
        subject = st.text_input(
            "Subject",
            placeholder="Enter ticket subject...",
            key="ticket_subject"
        )
        
        body = st.text_area(
            "Description",
            placeholder="Enter ticket description...",
            height=200,
            key="ticket_body"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            triage_btn = st.button("🏷️ Triage Only", use_container_width=True)
        
        with col_btn2:
            draft_btn = st.button("📝 Generate Draft", type="primary", use_container_width=True)
        
        with col_btn3:
            similar_btn = st.button("🔍 Find Similar", use_container_width=True)
    
    with col2:
        st.markdown("### Quick Templates")
        templates = {
            "Billing Issue": ("Billing question", "I was charged incorrectly on my last bill. The amount shows $99 but my plan is $49/month. Please help."),
            "Technical Problem": ("App not working", "The application crashes whenever I try to upload a file. I've tried reinstalling but the issue persists."),
            "Account Access": ("Cannot login", "I forgot my password and the reset email never arrived. I've checked spam folder too."),
        }
        
        for name, (subj, desc) in templates.items():
            if st.button(name, key=f"template_{name}", use_container_width=True):
                st.session_state.ticket_subject = subj
                st.session_state.ticket_body = desc
                st.rerun()
    
    st.markdown("---")
    
    # Process Actions
    if triage_btn and subject and body:
        with st.spinner("Triaging ticket..."):
            result = triage_ticket(subject, body)
        
        if result:
            st.session_state.triage_result = result
    
    if draft_btn and subject and body:
        with st.spinner("Generating draft response..."):
            # First triage
            triage_result = triage_ticket(subject, body)
            if triage_result:
                st.session_state.triage_result = triage_result
                
                # Then generate draft
                draft_result = generate_draft(
                    subject, body,
                    category=triage_result.get('category'),
                    priority=triage_result.get('priority')
                )
                if draft_result:
                    st.session_state.draft_result = draft_result
    
    if similar_btn and (subject or body):
        with st.spinner("Finding similar tickets..."):
            query = f"{subject}\n{body}"
            similar = find_similar_tickets(query, top_k=5)
            if similar:
                st.session_state.similar_tickets = similar
    
    # Display Results
    if 'triage_result' in st.session_state:
        render_triage_results(st.session_state.triage_result)
    
    if 'draft_result' in st.session_state:
        render_draft_results(st.session_state.draft_result)
    
    if 'similar_tickets' in st.session_state:
        render_similar_tickets(st.session_state.similar_tickets)


def render_triage_results(result: Dict):
    """Render triage results"""
    st.markdown("### 🏷️ Triage Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Category**")
        st.markdown(f"### {result.get('category', 'Unknown')}")
        confidence = result.get('category_confidence', 0)
        st.progress(confidence)
        st.caption(f"{confidence:.0%} confidence")
    
    with col2:
        priority = result.get('priority', 'P3')
        st.markdown("**Priority**")
        priority_class = f"priority-{priority.lower()}"
        st.markdown(f'<span class="{priority_class}">{priority}</span>', unsafe_allow_html=True)
        st.progress(result.get('priority_confidence', 0))
    
    with col3:
        st.markdown("**Suggested Queue**")
        st.markdown(f"### {result.get('suggested_queue', 'N/A')}")
    
    with col4:
        st.markdown("**SLA Risk**")
        if result.get('sla_risk'):
            st.markdown("### ⚠️ At Risk")
        else:
            st.markdown("### ✅ On Track")
    
    # Show probability distribution
    with st.expander("View Category Probabilities"):
        probs = result.get('category_probabilities', {})
        if probs:
            df = pd.DataFrame([
                {'Category': k, 'Probability': v}
                for k, v in sorted(probs.items(), key=lambda x: -x[1])
            ])
            fig = px.bar(df, x='Category', y='Probability', 
                        title='Category Confidence Distribution')
            st.plotly_chart(fig, use_container_width=True)


def render_draft_results(result: Dict):
    """Render draft response results"""
    st.markdown("### 📝 Draft Response")
    
    # Confidence indicator
    confidence = result.get('confidence', 'medium')
    confidence_class = f"confidence-{confidence}"
    st.markdown(
        f'<span class="{confidence_class}">Confidence: {confidence.upper()}</span> '
        f'({result.get("confidence_score", 0):.0%})',
        unsafe_allow_html=True
    )
    
    # Draft text
    st.markdown("#### Generated Draft")
    st.text_area(
        "Draft",
        value=result.get('draft_text', ''),
        height=200,
        key="draft_text_display",
        label_visibility="collapsed"
    )
    
    # Copy button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📋 Copy to Clipboard"):
            st.write("Draft copied!")
    with col2:
        if st.button("✏️ Edit Draft"):
            st.session_state.editing_draft = True
    
    # Citations
    citations = result.get('citations', [])
    if citations:
        st.markdown("#### 📚 Sources")
        for citation in citations:
            st.markdown(
                f'<div class="citation-box">'
                f'<strong>[{citation.get("id", "?")}]</strong> '
                f'{citation.get("title", "Untitled")}<br>'
                f'<em>{citation.get("excerpt", "")[:200]}...</em>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Follow-up questions
    follow_ups = result.get('follow_up_questions', [])
    if follow_ups:
        st.markdown("#### ❓ Verification Questions")
        for q in follow_ups:
            st.markdown(f"- {q}")
    
    # Performance metrics
    with st.expander("View Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{result.get('total_time_ms', 0)}ms")
        with col2:
            st.metric("Retrieval Time", f"{result.get('retrieval_time_ms', 0)}ms")
        with col3:
            st.metric("Generation Time", f"{result.get('generation_time_ms', 0)}ms")
    
    # Feedback section
    st.markdown("---")
    st.markdown("#### 📊 Feedback")
    
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider("Rate this draft", 1, 5, 3, key="draft_rating")
    with col2:
        is_helpful = st.radio("Was this helpful?", ["Yes", "No"], horizontal=True) == "Yes"
    
    feedback_text = st.text_input("Additional feedback (optional)")
    
    if st.button("Submit Feedback", type="primary"):
        feedback_result = submit_feedback(
            result.get('draft_id', ''),
            rating,
            is_helpful,
            feedback_text
        )
        if feedback_result:
            st.success("Feedback submitted! Thank you.")


def render_similar_tickets(tickets: List[Dict]):
    """Render similar tickets"""
    st.markdown("### 🔍 Similar Tickets")
    
    if not tickets:
        st.info("No similar tickets found.")
        return
    
    for i, ticket in enumerate(tickets, 1):
        with st.expander(f"**{i}. {ticket.get('subject', 'Untitled')}** (Score: {ticket.get('similarity_score', 0):.2f})"):
            st.markdown(f"**Category:** {ticket.get('category', 'N/A')} | "
                       f"**Priority:** {ticket.get('priority', 'N/A')} | "
                       f"**Status:** {ticket.get('status', 'N/A')}")
            st.markdown(ticket.get('body_preview', ''))


# =============================================================================
# Ticket List Page
# =============================================================================

def render_ticket_list_page():
    """Render ticket list page"""
    st.markdown('<div class="main-header">📋 Ticket List</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category_filter = st.selectbox(
            "Category",
            ["All", "Billing", "Technical", "Account", "Shipping", "Returns", "Product", "General"]
        )
    
    with col2:
        priority_filter = st.selectbox(
            "Priority",
            ["All", "P1", "P2", "P3", "P4"]
        )
    
    with col3:
        status_filter = st.selectbox(
            "Status",
            ["All", "open", "pending", "resolved", "closed"]
        )
    
    with col4:
        page_size = st.selectbox("Page Size", [10, 20, 50], index=1)
    
    # Pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Build filters
    filters = {}
    if category_filter != "All":
        filters['category'] = category_filter
    if priority_filter != "All":
        filters['priority'] = priority_filter
    if status_filter != "All":
        filters['status'] = status_filter
    
    # Fetch tickets
    result = list_tickets(
        page=st.session_state.current_page,
        page_size=page_size,
        **filters
    )
    
    if result and result.get('tickets'):
        tickets = result['tickets']
        total = result.get('total', 0)
        total_pages = result.get('total_pages', 1)
        
        st.markdown(f"Showing {len(tickets)} of {total} tickets (Page {st.session_state.current_page}/{total_pages})")
        
        # Display tickets as table
        df = pd.DataFrame(tickets)
        if not df.empty:
            display_cols = ['subject', 'category', 'priority', 'status', 'created_at']
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Previous", disabled=st.session_state.current_page <= 1):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col3:
            if st.button("Next →", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page += 1
                st.rerun()
    else:
        st.info("No tickets found.")


# =============================================================================
# Analytics Page
# =============================================================================

def render_analytics_page():
    """Render analytics dashboard"""
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Time period selector
    period = st.selectbox("Time Period", ["day", "week", "month"], index=1)
    
    metrics = get_metrics(period)
    
    if not metrics:
        st.warning("Unable to load metrics.")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tickets",
            metrics.get('total_tickets', 0),
            help="Total tickets in selected period"
        )
    
    with col2:
        st.metric(
            "Drafts Generated",
            metrics.get('total_drafts', 0),
            help="AI drafts generated"
        )
    
    with col3:
        avg_rating = metrics.get('avg_draft_rating')
        st.metric(
            "Avg Draft Rating",
            f"{avg_rating:.1f}/5" if avg_rating else "N/A",
            help="Average agent rating of drafts"
        )
    
    with col4:
        avg_latency = metrics.get('avg_latency_ms')
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f}ms" if avg_latency else "N/A",
            help="Average draft generation time"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Tickets by Category")
        by_category = metrics.get('tickets_by_category', {})
        if by_category:
            df = pd.DataFrame([
                {'Category': k, 'Count': v}
                for k, v in by_category.items()
            ])
            fig = px.pie(df, values='Count', names='Category', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")
    
    with col2:
        st.markdown("### Tickets by Priority")
        by_priority = metrics.get('tickets_by_priority', {})
        if by_priority:
            # Sort priorities
            priority_order = ['P1', 'P2', 'P3', 'P4']
            df = pd.DataFrame([
                {'Priority': k, 'Count': by_priority.get(k, 0)}
                for k in priority_order if k in by_priority
            ])
            colors = {'P1': '#dc3545', 'P2': '#fd7e14', 'P3': '#ffc107', 'P4': '#28a745'}
            fig = px.bar(df, x='Priority', y='Count',
                        color='Priority', color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No priority data available")
    
    # Performance Section
    st.markdown("---")
    st.markdown("### System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Response Time Distribution")
        # Placeholder for actual latency distribution data
        st.info("Latency distribution chart would appear here with real data")
    
    with col2:
        st.markdown("#### Draft Quality Metrics")
        # Placeholder for quality metrics
        quality_metrics = {
            'Citation Rate': 94,
            'Acceptance Rate': 72,
            'Low Confidence Rate': 8
        }
        for metric, value in quality_metrics.items():
            st.progress(value / 100, text=f"{metric}: {value}%")


# =============================================================================
# Settings Page
# =============================================================================

def render_settings_page():
    """Render settings page"""
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    st.markdown("### API Configuration")
    st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
    
    st.markdown("---")
    
    st.markdown("### Model Settings")
    st.info("Model configuration is managed via environment variables.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**LLM Settings**")
        st.code("""
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
USE_LLM=true
        """)
    
    with col2:
        st.markdown("**Database Settings**")
        st.code("""
DB_HOST=localhost
DB_PORT=5432
DB_NAME=localtriage
        """)
    
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    **LOCALTRIAGE** is a self-hosted, privacy-preserving customer support triage platform.
    
    Features:
    - 🏷️ Automatic ticket categorization and prioritization
    - 📝 RAG-powered response drafting with citations
    - 📊 Analytics and insights
    - 🔒 Fully local LLM inference
    
    Version: 1.0.0
    """)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point"""
    page = render_sidebar()
    
    if page == "🆕 New Ticket":
        render_new_ticket_page()
    elif page == "📋 Ticket List":
        render_ticket_list_page()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "⚙️ Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
