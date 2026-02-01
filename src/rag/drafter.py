"""
RAG Response Drafter for LOCALTRIAGE
LLM-based response generation with citation support
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

import requests


@dataclass
class Citation:
    """Represents a citation to a source document"""
    id: str
    source_type: str  # 'kb' or 'ticket'
    source_id: str
    title: str
    excerpt: str
    confidence: float = 1.0


@dataclass
class DraftResponse:
    """Represents a generated draft response"""
    draft_id: str
    ticket_id: str
    draft_text: str
    rationale: str
    confidence: str  # 'high', 'medium', 'low'
    confidence_score: float
    citations: List[Citation]
    follow_up_questions: List[str]
    
    # Generation metadata
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    generation_time_ms: int
    
    # Retrieval metadata
    retrieval_time_ms: int
    sources_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'draft_id': self.draft_id,
            'ticket_id': self.ticket_id,
            'draft_text': self.draft_text,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'confidence_score': self.confidence_score,
            'citations': [
                {
                    'id': c.id,
                    'source_type': c.source_type,
                    'source_id': c.source_id,
                    'title': c.title,
                    'excerpt': c.excerpt[:200] if c.excerpt else ''
                }
                for c in self.citations
            ],
            'follow_up_questions': self.follow_up_questions,
            'model_name': self.model_name,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'generation_time_ms': self.generation_time_ms,
            'retrieval_time_ms': self.retrieval_time_ms,
            'sources_used': self.sources_used
        }


class LLMClient:
    """
    Client for local LLM inference via OpenAI-compatible API
    
    Supports vLLM, llama.cpp server, Ollama, and other compatible endpoints
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        api_key: str = "not-needed",
        timeout: int = 120,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate completion from LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Optional JSON schema for structured output
            
        Returns:
            Dict with 'content', 'prompt_tokens', 'completion_tokens', 'latency_ms'
        """
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                'content': data['choices'][0]['message']['content'],
                'prompt_tokens': data.get('usage', {}).get('prompt_tokens', 0),
                'completion_tokens': data.get('usage', {}).get('completion_tokens', 0),
                'latency_ms': latency_ms,
                'finish_reason': data['choices'][0].get('finish_reason', 'unknown')
            }
        
        except requests.exceptions.RequestException as e:
            return {
                'content': '',
                'error': str(e),
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'latency_ms': int((time.time() - start_time) * 1000)
            }
    
    def health_check(self) -> bool:
        """Check if LLM service is available"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


class RAGDrafter:
    """
    RAG-based response drafter with citation support
    
    Combines retrieval results with LLM generation to create
    grounded, citable response drafts.
    """
    
    SYSTEM_PROMPT = """You are an expert customer support assistant. Your task is to draft helpful, accurate responses to customer support tickets.

CRITICAL RULES:
1. ONLY use information from the provided context (Knowledge Base articles and similar tickets)
2. ALWAYS cite your sources using [KB-X] for knowledge base or [TICKET-X] for similar tickets
3. If information is not in the context, say "I don't have information about that in our knowledge base"
4. Be professional, empathetic, and solution-oriented
5. Keep responses concise but complete

OUTPUT FORMAT:
You must respond in valid JSON format with these fields:
{
  "draft": "Your response to the customer, with citations like [KB-1] inline",
  "rationale": "Brief explanation of your approach",
  "confidence": "high|medium|low",
  "confidence_reason": "Why this confidence level",
  "follow_up_questions": ["Questions the agent should verify", ...]
}"""

    DRAFT_PROMPT_TEMPLATE = """## Customer Ticket
**Subject:** {subject}
**Body:** {body}
{category_info}

## Knowledge Base Context
{kb_context}

## Similar Resolved Tickets
{ticket_context}

## Instructions
Draft a response to the customer ticket above. Remember to:
- Cite sources with [KB-X] or [TICKET-X] format
- Address all aspects of the customer's issue
- Provide clear next steps
- Maintain a professional and helpful tone

Respond with valid JSON only."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever,  # HybridRetriever or similar
        min_confidence_threshold: float = 0.3,
        max_context_chunks: int = 5,
        max_similar_tickets: int = 3
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.min_confidence_threshold = min_confidence_threshold
        self.max_context_chunks = max_context_chunks
        self.max_similar_tickets = max_similar_tickets
    
    def _format_kb_context(self, kb_results: List[Any]) -> Tuple[str, List[Citation]]:
        """Format KB results for prompt and create citations"""
        if not kb_results:
            return "No relevant knowledge base articles found.", []
        
        context_parts = []
        citations = []
        
        for i, result in enumerate(kb_results[:self.max_context_chunks], 1):
            kb_id = f"KB-{i}"
            title = result.metadata.get('title', 'Untitled')
            
            context_parts.append(f"[{kb_id}] **{title}**\n{result.content}\n")
            
            citations.append(Citation(
                id=kb_id,
                source_type='kb',
                source_id=result.id,
                title=title,
                excerpt=result.content[:500],
                confidence=result.score
            ))
        
        return "\n".join(context_parts), citations
    
    def _format_ticket_context(self, ticket_results: List[Any]) -> Tuple[str, List[Citation]]:
        """Format similar ticket results for prompt and create citations"""
        if not ticket_results:
            return "No similar resolved tickets found.", []
        
        context_parts = []
        citations = []
        
        for i, result in enumerate(ticket_results[:self.max_similar_tickets], 1):
            ticket_id = f"TICKET-{i}"
            title = result.title or 'Ticket'
            
            context_parts.append(f"[{ticket_id}] **{title}**\n{result.content}\n")
            
            citations.append(Citation(
                id=ticket_id,
                source_type='ticket',
                source_id=result.id,
                title=title,
                excerpt=result.content[:500],
                confidence=result.score
            ))
        
        return "\n".join(context_parts), citations
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response, handling JSON extraction"""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object anywhere in content
        json_match = re.search(r'\{[^{}]*"draft"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: treat entire content as draft
        return {
            'draft': content,
            'rationale': 'Unable to parse structured response',
            'confidence': 'low',
            'confidence_reason': 'Response format issue',
            'follow_up_questions': ['Please verify the response']
        }
    
    def _extract_used_citations(
        self,
        draft_text: str,
        all_citations: List[Citation]
    ) -> List[Citation]:
        """Extract only citations that were actually used in the draft"""
        used = []
        for citation in all_citations:
            if citation.id in draft_text:
                used.append(citation)
        return used
    
    def _calculate_confidence_score(
        self,
        confidence_level: str,
        used_citations: List[Citation],
        kb_results: List[Any]
    ) -> float:
        """Calculate numeric confidence score"""
        base_scores = {'high': 0.85, 'medium': 0.6, 'low': 0.3}
        base = base_scores.get(confidence_level.lower(), 0.5)
        
        # Adjust based on retrieval quality
        if kb_results:
            avg_retrieval_score = sum(r.score for r in kb_results) / len(kb_results)
            retrieval_factor = min(avg_retrieval_score, 1.0) * 0.1
        else:
            retrieval_factor = -0.1
        
        # Adjust based on citation usage
        citation_factor = min(len(used_citations) * 0.05, 0.15)
        
        return min(max(base + retrieval_factor + citation_factor, 0.0), 1.0)
    
    def generate_draft(
        self,
        ticket_id: str,
        subject: str,
        body: str,
        category: Optional[str] = None,
        priority: Optional[str] = None
    ) -> DraftResponse:
        """
        Generate a draft response for a ticket
        
        Args:
            ticket_id: Unique ticket identifier
            subject: Ticket subject
            body: Ticket body text
            category: Optional predicted category
            priority: Optional predicted priority
            
        Returns:
            DraftResponse with generated draft and metadata
        """
        draft_id = str(uuid.uuid4())
        
        # Retrieve relevant context
        retrieval_start = time.time()
        
        query = f"{subject}\n{body}"
        
        # Get KB context
        kb_results = []
        if hasattr(self.retriever, 'search'):
            kb_results = self.retriever.search(query, top_k=self.max_context_chunks)
        elif hasattr(self.retriever, 'search_kb'):
            kb_results = self.retriever.search_kb(query, top_k=self.max_context_chunks)
        
        # Get similar tickets
        ticket_results = []
        if hasattr(self.retriever, 'search_similar_tickets'):
            ticket_results = self.retriever.search_similar_tickets(
                query,
                top_k=self.max_similar_tickets,
                exclude_ticket_id=ticket_id
            )
        
        retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
        
        # Format context
        kb_context, kb_citations = self._format_kb_context(kb_results)
        ticket_context, ticket_citations = self._format_ticket_context(ticket_results)
        all_citations = kb_citations + ticket_citations
        
        # Check if we have sufficient context
        if not kb_results and not ticket_results:
            # Return low-confidence response
            return DraftResponse(
                draft_id=draft_id,
                ticket_id=ticket_id,
                draft_text="I apologize, but I couldn't find relevant information in our knowledge base to address your specific question. Please allow me to escalate this to a specialist who can assist you better.",
                rationale="No relevant context found in knowledge base or similar tickets",
                confidence='low',
                confidence_score=0.1,
                citations=[],
                follow_up_questions=[
                    "Escalate to appropriate specialist",
                    "Check if KB needs updating for this topic"
                ],
                model_name=self.llm.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                generation_time_ms=0,
                retrieval_time_ms=retrieval_time_ms,
                sources_used=0
            )
        
        # Build category info
        category_info = ""
        if category or priority:
            parts = []
            if category:
                parts.append(f"**Category:** {category}")
            if priority:
                parts.append(f"**Priority:** {priority}")
            category_info = "\n" + "\n".join(parts)
        
        # Build prompt
        user_prompt = self.DRAFT_PROMPT_TEMPLATE.format(
            subject=subject,
            body=body,
            category_info=category_info,
            kb_context=kb_context,
            ticket_context=ticket_context
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response
        response = self.llm.generate(messages)
        
        if response.get('error'):
            return DraftResponse(
                draft_id=draft_id,
                ticket_id=ticket_id,
                draft_text=f"Error generating response: {response['error']}",
                rationale="LLM generation failed",
                confidence='low',
                confidence_score=0.0,
                citations=[],
                follow_up_questions=["Retry generation or respond manually"],
                model_name=self.llm.model_name,
                prompt_tokens=response['prompt_tokens'],
                completion_tokens=response['completion_tokens'],
                generation_time_ms=response['latency_ms'],
                retrieval_time_ms=retrieval_time_ms,
                sources_used=len(kb_results) + len(ticket_results)
            )
        
        # Parse response
        parsed = self._parse_llm_response(response['content'])
        
        draft_text = parsed.get('draft', response['content'])
        confidence = parsed.get('confidence', 'medium')
        
        # Extract used citations
        used_citations = self._extract_used_citations(draft_text, all_citations)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            confidence, used_citations, kb_results
        )
        
        return DraftResponse(
            draft_id=draft_id,
            ticket_id=ticket_id,
            draft_text=draft_text,
            rationale=parsed.get('rationale', ''),
            confidence=confidence,
            confidence_score=confidence_score,
            citations=used_citations,
            follow_up_questions=parsed.get('follow_up_questions', []),
            model_name=self.llm.model_name,
            prompt_tokens=response['prompt_tokens'],
            completion_tokens=response['completion_tokens'],
            generation_time_ms=response['latency_ms'],
            retrieval_time_ms=retrieval_time_ms,
            sources_used=len(kb_results) + len(ticket_results)
        )


class BaselineTemplateResponder:
    """
    Baseline template-based responder (no LLM)
    
    Used for comparison and as fallback
    """
    
    TEMPLATES = {
        'Billing': """Dear Customer,

Thank you for contacting us about your billing inquiry.

We understand that billing matters are important, and we want to help resolve your concern as quickly as possible.

Our billing team has been notified and will review your account. You can expect a detailed response within 24-48 hours.

In the meantime, you can view your billing history and invoices in your account dashboard.

Best regards,
Support Team""",

        'Technical': """Dear Customer,

Thank you for reporting this technical issue.

We apologize for any inconvenience this may have caused. Our technical team is looking into this matter.

In the meantime, please try the following troubleshooting steps:
1. Clear your browser cache and cookies
2. Try using a different browser or device
3. Check your internet connection

If the issue persists, please reply with any error messages you're seeing.

Best regards,
Technical Support Team""",

        'Account': """Dear Customer,

Thank you for contacting us about your account.

We take account security and access very seriously. To assist you better, we may need to verify your identity.

Please reply with the email address associated with your account, and we'll help you from there.

Best regards,
Account Support Team""",

        'default': """Dear Customer,

Thank you for contacting our support team.

We have received your inquiry and our team is reviewing it. You can expect a response within 24-48 hours.

If your matter is urgent, please reply to this message with additional details.

Best regards,
Support Team"""
    }
    
    def generate_response(
        self,
        category: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate template response based on category"""
        template = self.TEMPLATES.get(category, self.TEMPLATES['default'])
        return template


if __name__ == '__main__':
    # Example usage
    llm = LLMClient(
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-14B-Instruct"
    )
    
    if llm.health_check():
        print("LLM service is available")
        
        # Test generation
        response = llm.generate([
            {"role": "user", "content": "Hello, how can you help me?"}
        ])
        print(f"Response: {response['content'][:200]}...")
        print(f"Tokens: {response['prompt_tokens']} prompt, {response['completion_tokens']} completion")
        print(f"Latency: {response['latency_ms']}ms")
    else:
        print("LLM service is not available")
