"""
Data Ingestion Module for LOCALTRIAGE
Handles CSV, JSON ticket ingestion and KB article processing
"""

import os
import json
import csv
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
import re
import uuid

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import markdown
from bs4 import BeautifulSoup


@dataclass
class Ticket:
    """Represents a customer support ticket"""
    subject: str
    body: str
    external_id: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    channel: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'external_id': self.external_id,
            'subject': self.subject,
            'body': self.body,
            'category': self.category,
            'priority': self.priority,
            'customer_id': self.customer_id,
            'customer_email': self.customer_email,
            'channel': self.channel,
            'created_at': self.created_at
        }


@dataclass
class KBArticle:
    """Represents a knowledge base article"""
    title: str
    content: str
    external_id: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    source_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'external_id': self.external_id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'category': self.category,
            'tags': self.tags,
            'author': self.author,
            'source_url': self.source_url
        }


@dataclass
class KBChunk:
    """Represents a chunk of a KB article for retrieval"""
    article_id: str
    content: str
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int
    embedding_id: Optional[str] = None


class DatabaseConnection:
    """Manages PostgreSQL database connections"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "localtriage",
        user: str = "postgres",
        password: str = "postgres"
    ):
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.conn_params)
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TicketIngester:
    """Handles ingestion of support tickets from various formats"""
    
    CATEGORY_MAPPING = {
        'billing': 'Billing',
        'payment': 'Billing',
        'invoice': 'Billing',
        'charge': 'Billing',
        'technical': 'Technical',
        'bug': 'Technical',
        'error': 'Technical',
        'issue': 'Technical',
        'account': 'Account',
        'login': 'Account',
        'password': 'Account',
        'shipping': 'Shipping',
        'delivery': 'Shipping',
        'return': 'Returns',
        'refund': 'Returns',
        'product': 'Product',
        'feature': 'Product',
    }
    
    PRIORITY_MAPPING = {
        'critical': 'P1',
        'urgent': 'P1',
        'high': 'P2',
        'medium': 'P3',
        'normal': 'P3',
        'low': 'P4',
        '1': 'P1',
        '2': 'P2',
        '3': 'P3',
        '4': 'P4',
    }
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def normalize_category(self, category: Optional[str]) -> Optional[str]:
        """Normalize category to standard values"""
        if not category:
            return None
        category_lower = category.lower().strip()
        return self.CATEGORY_MAPPING.get(category_lower, category.title())
    
    def normalize_priority(self, priority: Optional[str]) -> Optional[str]:
        """Normalize priority to P1-P4 scale"""
        if not priority:
            return None
        priority_lower = str(priority).lower().strip()
        return self.PRIORITY_MAPPING.get(priority_lower, 'P3')
    
    def parse_csv(self, filepath: str) -> Generator[Ticket, None, None]:
        """Parse tickets from CSV file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle various column naming conventions
                subject = row.get('subject') or row.get('title') or row.get('Subject') or ''
                body = row.get('body') or row.get('description') or row.get('content') or row.get('Body') or ''
                
                ticket = Ticket(
                    external_id=row.get('id') or row.get('ticket_id') or row.get('ID'),
                    subject=subject.strip(),
                    body=body.strip(),
                    category=self.normalize_category(
                        row.get('category') or row.get('type') or row.get('Category')
                    ),
                    priority=self.normalize_priority(
                        row.get('priority') or row.get('Priority') or row.get('severity')
                    ),
                    customer_id=row.get('customer_id') or row.get('user_id'),
                    customer_email=row.get('email') or row.get('customer_email'),
                    channel=row.get('channel') or row.get('source'),
                )
                
                if ticket.subject or ticket.body:
                    yield ticket
    
    def parse_json(self, filepath: str) -> Generator[Ticket, None, None]:
        """Parse tickets from JSON file (array or newline-delimited)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        except json.JSONDecodeError:
            # Try newline-delimited JSON
            items = []
            for line in content.split('\n'):
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        for item in items:
            subject = item.get('subject') or item.get('title') or ''
            body = item.get('body') or item.get('description') or item.get('content') or ''
            
            ticket = Ticket(
                external_id=item.get('id') or item.get('ticket_id'),
                subject=subject.strip() if isinstance(subject, str) else str(subject),
                body=body.strip() if isinstance(body, str) else str(body),
                category=self.normalize_category(item.get('category') or item.get('type')),
                priority=self.normalize_priority(item.get('priority') or item.get('severity')),
                customer_id=item.get('customer_id') or item.get('user_id'),
                customer_email=item.get('email') or item.get('customer_email'),
                channel=item.get('channel') or item.get('source'),
            )
            
            if ticket.subject or ticket.body:
                yield ticket
    
    def ingest_file(self, filepath: str) -> Dict[str, int]:
        """Ingest tickets from a file (CSV or JSON)"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            tickets = list(self.parse_csv(str(filepath)))
        elif filepath.suffix.lower() == '.json':
            tickets = list(self.parse_json(str(filepath)))
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self.insert_tickets(tickets)
    
    def insert_tickets(self, tickets: List[Ticket]) -> Dict[str, int]:
        """Insert tickets into database"""
        inserted = 0
        skipped = 0
        errors = 0
        
        with self.db.connect() as conn:
            with conn.cursor() as cur:
                for ticket in tickets:
                    try:
                        cur.execute("""
                            INSERT INTO tickets (
                                external_id, subject, body, category, priority,
                                customer_id, customer_email, channel, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (external_id) DO NOTHING
                            RETURNING id
                        """, (
                            ticket.external_id,
                            ticket.subject,
                            ticket.body,
                            ticket.category,
                            ticket.priority,
                            ticket.customer_id,
                            ticket.customer_email,
                            ticket.channel,
                            ticket.created_at or datetime.now()
                        ))
                        
                        result = cur.fetchone()
                        if result:
                            inserted += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        errors += 1
                        print(f"Error inserting ticket: {e}")
                
                conn.commit()
        
        return {
            'inserted': inserted,
            'skipped': skipped,
            'errors': errors,
            'total': len(tickets)
        }


class KBIngester:
    """Handles ingestion of knowledge base articles"""
    
    def __init__(
        self,
        db: DatabaseConnection,
        chunk_size: int = 1024,
        chunk_overlap: int = 256
    ):
        self.db = db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def parse_markdown(self, filepath: str) -> KBArticle:
        """Parse a markdown file into a KB article"""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from first heading or filename
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = filepath.stem.replace('-', ' ').replace('_', ' ').title()
        
        # Extract frontmatter if present
        category = None
        tags = []
        author = None
        
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            content = content[frontmatter_match.end():]
            
            # Parse YAML-like frontmatter
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'category':
                        category = value
                    elif key == 'tags':
                        tags = [t.strip() for t in value.strip('[]').split(',')]
                    elif key == 'author':
                        author = value
        
        # Convert markdown to plain text for storage
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        plain_text = soup.get_text(separator='\n')
        
        # Generate summary (first paragraph or first N chars)
        paragraphs = [p.strip() for p in plain_text.split('\n\n') if p.strip()]
        summary = paragraphs[0][:500] if paragraphs else None
        
        return KBArticle(
            external_id=hashlib.md5(filepath.name.encode()).hexdigest()[:16],
            title=title,
            content=plain_text,
            summary=summary,
            category=category,
            tags=tags,
            author=author,
            source_url=str(filepath)
        )
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Simple token estimation (4 chars per token)
        chars_per_token = 4
        char_chunk_size = chunk_size * chars_per_token
        char_overlap = overlap * chars_per_token
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + char_chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within last 20% of chunk
                search_start = int(end - char_chunk_size * 0.2)
                last_period = text.rfind('. ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': chunk_index,
                    'token_count': len(chunk_text) // chars_per_token,
                    'start_char': start,
                    'end_char': end
                })
                chunk_index += 1
            
            start = end - char_overlap
            if start <= chunks[-1]['start_char'] if chunks else 0:
                start = end
        
        return chunks
    
    def ingest_directory(self, dirpath: str) -> Dict[str, int]:
        """Ingest all markdown files from a directory"""
        dirpath = Path(dirpath)
        articles = []
        
        for filepath in dirpath.rglob('*.md'):
            try:
                article = self.parse_markdown(str(filepath))
                articles.append(article)
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
        
        return self.insert_articles(articles)
    
    def insert_articles(self, articles: List[KBArticle]) -> Dict[str, int]:
        """Insert KB articles and their chunks into database"""
        inserted = 0
        chunks_inserted = 0
        skipped = 0
        errors = 0
        
        with self.db.connect() as conn:
            with conn.cursor() as cur:
                for article in articles:
                    try:
                        # Insert article
                        cur.execute("""
                            INSERT INTO kb_articles (
                                external_id, title, content, summary,
                                category, tags, author, source_url,
                                status, published_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'published', NOW())
                            ON CONFLICT (external_id) DO UPDATE SET
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                summary = EXCLUDED.summary,
                                updated_at = NOW()
                            RETURNING id
                        """, (
                            article.external_id,
                            article.title,
                            article.content,
                            article.summary,
                            article.category,
                            article.tags,
                            article.author,
                            article.source_url
                        ))
                        
                        result = cur.fetchone()
                        if result:
                            article_id = result[0]
                            inserted += 1
                            
                            # Delete existing chunks for this article
                            cur.execute(
                                "DELETE FROM kb_chunks WHERE article_id = %s",
                                (article_id,)
                            )
                            
                            # Create and insert chunks
                            chunks = self.chunk_text(article.content)
                            for chunk in chunks:
                                cur.execute("""
                                    INSERT INTO kb_chunks (
                                        article_id, content, chunk_index,
                                        token_count, start_char, end_char
                                    ) VALUES (%s, %s, %s, %s, %s, %s)
                                """, (
                                    article_id,
                                    chunk['content'],
                                    chunk['chunk_index'],
                                    chunk['token_count'],
                                    chunk['start_char'],
                                    chunk['end_char']
                                ))
                                chunks_inserted += 1
                        else:
                            skipped += 1
                    
                    except Exception as e:
                        errors += 1
                        print(f"Error inserting article {article.title}: {e}")
                
                conn.commit()
        
        return {
            'articles_inserted': inserted,
            'chunks_inserted': chunks_inserted,
            'skipped': skipped,
            'errors': errors,
            'total': len(articles)
        }


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LOCALTRIAGE Data Ingestion')
    parser.add_argument('command', choices=['tickets', 'kb', 'schema'],
                        help='Ingestion command')
    parser.add_argument('--file', '-f', help='Input file path')
    parser.add_argument('--dir', '-d', help='Input directory path')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--db-name', default='localtriage', help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--db-password', default='postgres', help='Database password')
    
    args = parser.parse_args()
    
    db = DatabaseConnection(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password
    )
    
    if args.command == 'schema':
        # Apply database schema
        schema_path = Path(__file__).parent / 'schema.sql'
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(schema)
                conn.commit()
        print("Schema applied successfully")
    
    elif args.command == 'tickets':
        if not args.file:
            print("Error: --file required for ticket ingestion")
            return
        
        ingester = TicketIngester(db)
        result = ingester.ingest_file(args.file)
        print(f"Ticket ingestion complete: {result}")
    
    elif args.command == 'kb':
        if args.dir:
            ingester = KBIngester(db)
            result = ingester.ingest_directory(args.dir)
            print(f"KB ingestion complete: {result}")
        elif args.file:
            ingester = KBIngester(db)
            article = ingester.parse_markdown(args.file)
            result = ingester.insert_articles([article])
            print(f"KB ingestion complete: {result}")
        else:
            print("Error: --file or --dir required for KB ingestion")


if __name__ == '__main__':
    main()
