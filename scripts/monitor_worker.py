#!/usr/bin/env python3
"""
Unified Monitor Worker for GitHub Duplicate Issue Finder.
This worker consolidates all monitoring functionality into a single, configurable system.
"""

import requests
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys
from enum import Enum

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import Session
from sqlalchemy import text
import httpx

from src.db import SessionLocal
from src.models import Issue, DiscoursePost, MetabaseDoc, Question, KeywordDefinition, Synonym
from src.llm_client import llm_client
from src.text_utils import combine_discourse_posts, get_topic_creator_username, combine_all_discourse_posts, calculate_token_count
from src.settings import (
    GITHUB_REPO_OWNER,
    GITHUB_REPO_NAME,
    GITHUB_WORKER_TOKEN,
    DISCOURSE_BASE_URL,
    DISCOURSE_API_USERNAME,
    DISCOURSE_API_KEY,
    DISCOURSE_MAX_PAGES,
    API_KEY,
    LITELLM_MODEL_NAME,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    RERANKER_BATCH_SIZE,
    WORKER_POLL_INTERVAL_SECONDS,
    WORKER_BACKOFF_SECONDS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitorType(Enum):
    """Types of monitoring operations."""
    GITHUB_ISSUES = "github_issues"
    DISCOURSE_POSTS = "discourse_posts"
    LLM_SUMMARIES = "llm_summaries"
    EMBEDDINGS = "embeddings"

class UnifiedMonitorWorker:
    """Unified worker that handles all types of monitoring operations."""
    
    def __init__(self, monitor_type: MonitorType):
        """Initialize the worker with a specific monitoring type."""
        self.monitor_type = monitor_type
        self.api_url = "http://localhost:8000/v1/similar-github-issues"
        
        # Initialize analyzers if needed
        if monitor_type in [MonitorType.LLM_SUMMARIES]:
            self.llm_client = llm_client
        
        # Note: We don't initialize embedding_client or reranker_client here
        # as they would load ML models. Instead, we use the API endpoints
        # for embedding and reranking operations.
    
    def get_latest_entity_from_db(self, db: Session) -> int:
        """Get the latest entity ID from the database based on monitor type."""
        if self.monitor_type == MonitorType.GITHUB_ISSUES:
            latest_issue = db.query(Issue).order_by(Issue.number.desc()).first()
            return getattr(latest_issue, "number", 0) if latest_issue else 0
        elif self.monitor_type == MonitorType.DISCOURSE_POSTS:
            latest_post = db.query(DiscoursePost).order_by(DiscoursePost.topic_id.desc()).first()
            return getattr(latest_post, "topic_id", 0) if latest_post else 0
        else:
            return 0  # For LLM and embedding monitors, we don't need latest ID
    
    def get_new_github_issues(self, latest_issue_in_db: int) -> List[Dict[str, Any]]:
        """Fetches all new issues from the GitHub repository since the last known issue."""
        url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues"
        headers = {}
        if GITHUB_WORKER_TOKEN:
            headers["Authorization"] = f"token {GITHUB_WORKER_TOKEN}"
        else:
            logger.warning("GITHUB_WORKER_TOKEN not set. Making unauthenticated requests.")
        
        params = {"state": "open", "sort": "created", "direction": "desc", "per_page": 100}
        new_issues = []
        
        page = 1
        while True:
            logger.info(f"Fetching page {page} of new issues...")
            params["page"] = str(page)
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            page_of_issues = response.json()
            
            if not page_of_issues:
                break
                
            found_existing_issue = False
            for issue in page_of_issues:
                if issue['number'] <= latest_issue_in_db:
                    found_existing_issue = True
                    break
                if "pull_request" not in issue:
                    new_issues.append(issue)
            
            if found_existing_issue:
                break
                
            page += 1
            
        return new_issues
    
    def get_discourse_headers(self) -> Dict[str, str]:
        """Returns headers for Discourse API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'MetabaseDuplicateFinder/1.0'
        }
        
        if DISCOURSE_API_KEY and DISCOURSE_API_USERNAME:
            headers['Api-Key'] = str(DISCOURSE_API_KEY)
            headers['Api-Username'] = str(DISCOURSE_API_USERNAME)
        
        return headers
    
    def get_new_discourse_topics(self, latest_topic_in_db: int) -> List[Dict[str, Any]]:
        """Fetches all new topics from Discourse since the last known topic."""
        url = f"{DISCOURSE_BASE_URL}/latest.json"
        headers = self.get_discourse_headers()
        params = {"order": "desc"}
        new_topics = []
        
        page = 0
        topics_per_page = 30
        
        while True:
            logger.info(f"Fetching page {page + 1} of new topics...")
            if page > 0:
                params["page"] = str(page)
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                topic_list = data.get('topic_list', {})
                topics = topic_list.get('topics', [])
                
                if not topics:
                    logger.info("No topics found on this page.")
                    break
                    
                found_existing_topic = False
                new_topics_on_this_page = 0
                
                for topic in topics:
                    topic_id = topic.get('id')
                    if topic_id <= latest_topic_in_db:
                        found_existing_topic = True
                        logger.info(f"Found existing topic #{topic_id}, stopping pagination.")
                        break
                    new_topics.append(topic)
                    new_topics_on_this_page += 1
                
                logger.info(f"Found {new_topics_on_this_page} new topics on page {page + 1}")
                
                if found_existing_topic:
                    break
                
                more_topics_url = topic_list.get('more_topics_url')
                per_page = topic_list.get('per_page', topics_per_page)
                
                should_continue = False
                
                if more_topics_url:
                    should_continue = True
                elif len(topics) >= per_page:
                    should_continue = True
                else:
                    should_continue = False
                
                if not should_continue:
                    break
                    
                page += 1
                
                if page >= DISCOURSE_MAX_PAGES:
                    logger.warning(f"Reached maximum pages limit ({DISCOURSE_MAX_PAGES})")
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Error fetching Discourse topics: {e}")
                break
        
        return new_topics
    
    def get_entities_lacking_summaries(self, db: Session) -> Dict[str, List]:
        """Get entities from all tables that lack LLM summaries."""
        entities = {
            'issues': [],
            'discourse_posts': [],
            'metabase_docs': []
        }
        
        # Check issues
        issues_without_summary = db.query(Issue).filter(
            Issue.llm_summary.is_(None)
        ).limit(10).all()
        entities['issues'] = issues_without_summary
        
        # Check discourse posts
        try:
            discourse_without_summary = db.query(DiscoursePost).filter(
                DiscoursePost.llm_summary.is_(None)
            ).limit(10).all()
            entities['discourse_posts'] = discourse_without_summary
        except Exception as e:
            logger.error(f"Error querying discourse posts: {e}")
            entities['discourse_posts'] = []
        
        # Check metabase docs
        metabase_without_summary = db.query(MetabaseDoc).filter(
            MetabaseDoc.llm_summary.is_(None)
        ).limit(10).all()
        entities['metabase_docs'] = metabase_without_summary
        
        return entities
    
    def get_entities_lacking_embeddings(self, db: Session) -> Dict[str, List]:
        """Get entities from all tables that lack embeddings."""
        entities = {
            'issues': [],
            'discourse_posts': [],
            'metabase_docs': [],
            'questions': [],
            'keyword_definitions': [],
            'synonyms': []
        }
        
        # Check issues
        issues_without_embedding = db.query(Issue).filter(
            (Issue.title_embedding.is_(None)) | (Issue.issue_embedding.is_(None)) | (Issue.summary_embedding.is_(None))
        ).limit(50).all()  # Increased from 10 to 50 for faster processing
        entities['issues'] = issues_without_embedding
        logger.info(f"Found {len(issues_without_embedding)} issues needing embeddings")
        
        # Check discourse posts
        try:
            discourse_without_embedding = db.query(DiscoursePost).filter(
                (DiscoursePost.conversation_embedding.is_(None)) | (DiscoursePost.summary_embedding.is_(None))
            ).limit(50).all()  # Increased from 10 to 50 for faster processing
            entities['discourse_posts'] = discourse_without_embedding
            logger.info(f"Found {len(discourse_without_embedding)} discourse posts needing embeddings")
        except Exception as e:
            logger.error(f"Error querying discourse posts for embeddings: {e}")
            entities['discourse_posts'] = []
        
        # Check metabase docs
        metabase_without_embedding = db.query(MetabaseDoc).filter(
            (MetabaseDoc.markdown_embedding.is_(None)) | (MetabaseDoc.summary_embedding.is_(None))
        ).limit(50).all()  # Increased from 10 to 50 for faster processing
        entities['metabase_docs'] = metabase_without_embedding
        logger.info(f"Found {len(metabase_without_embedding)} metabase docs needing embeddings")
        
        # Check questions
        questions_without_embedding = db.query(Question).filter(
            (Question.question_embedding.is_(None)) | (Question.answer_embedding.is_(None))
        ).limit(50).all()  # Increased from 10 to 50 for faster processing
        entities['questions'] = questions_without_embedding
        logger.info(f"Found {len(questions_without_embedding)} questions needing embeddings")
        
        # Check keyword definitions
        keywords_without_embedding = db.query(KeywordDefinition).filter(
            KeywordDefinition.keyword_embedding.is_(None)
        ).limit(50).all()  # Increased from 10 to 50 for faster processing
        entities['keyword_definitions'] = keywords_without_embedding
        logger.info(f"Found {len(keywords_without_embedding)} keyword definitions needing embeddings")
        
        # Check synonyms
        synonyms_without_embedding = db.query(Synonym).filter(
            (Synonym.word_embedding.is_(None)) | (Synonym.synonym_embedding.is_(None))
        ).limit(50).all()  # Increased from 10 to 50 for faster processing
        entities['synonyms'] = synonyms_without_embedding
        logger.info(f"Found {len(synonyms_without_embedding)} synonyms needing embeddings")
        
        return entities
    
    def find_similar_issues(self, text: str) -> List[Dict[str, Any]]:
        """Calls the local similarity search API to find similar issues."""
        headers = {"X-API-Key": str(API_KEY)}
        payload = {"text": text}
        
        # Use v1 endpoint only (no reranker)
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    # def find_similar_issues_with_reranking(self, text: str) -> List[Dict[str, Any]]:
    #     """Find similar issues using the reranked API endpoint."""
    #     try:
    #         # Use the reranked endpoint directly
    #         headers = {"X-API-Key": API_KEY}
    #         payload = {"text": text}
    #         response = requests.post("http://localhost:8000/v2/similar-github-issues", headers=headers, json=payload)
    #         response.raise_for_status()
    #         return response.json()
    #         
    #     except Exception as e:
    #         logger.error(f"Error in reranked similarity search: {e}")
    #         return self.find_similar_issues(text)
    
    def post_comment_on_issue(self, issue_number: int, similar_issues: List[Dict[str, Any]], issue_creator: str) -> None:
        """Posts a comment on a GitHub issue with a list of potential duplicates."""
        # Filter issues with similarity score > 0.7
        high_similarity_issues = [issue for issue in similar_issues if issue['similarity_score'] > 0.7]
        
        if not high_similarity_issues:
            logger.info(f"No similar issues found with similarity > 70% for issue #{issue_number}")
            return

        comment_body = f"🤖 Hi @{issue_creator}! Our bot has found potential duplicates of this issue:\n\n"
        for issue in high_similarity_issues:
            similarity_percentage = round(issue['similarity_score'] * 100, 1)
            comment_body += f"- [#{issue['number']}: {issue['title']}]({issue['url']}) - {similarity_percentage}% similar\n"
        
        url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}/comments"
        headers = {}
        if GITHUB_WORKER_TOKEN:
            headers["Authorization"] = f"token {GITHUB_WORKER_TOKEN}"
        else:
            logger.warning("GITHUB_WORKER_TOKEN not set. Cannot post comments without a token.")
            return

        payload = {"body": comment_body}
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Posted comment on issue #{issue_number}")
    
    def save_github_issue_to_database(self, db: Session, issue_data: Dict[str, Any]) -> None:
        """Save a GitHub issue to the database with embeddings."""
        # Check if issue already exists
        existing_issue = db.query(Issue).filter(Issue.number == issue_data['number']).first()
        if existing_issue:
            logger.info(f"Issue #{issue_data['number']} already exists in database")
            return
        
        # Extract just the label names from the full label objects
        label_names = [label['name'] for label in issue_data.get('labels', [])]
        
        # Create new issue
        issue = Issue(
            number=issue_data['number'],
            title=issue_data['title'],
            body=issue_data.get('body', ''),
            state=issue_data['state'],
            created_at=datetime.fromisoformat(issue_data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(issue_data['updated_at'].replace('Z', '+00:00')),
            labels=label_names,
            user_login=issue_data['user']['login'],
            token_count=calculate_token_count(issue_data.get('body', ''))
        )
        
        # Persist first to obtain primary key, then update embeddings
        db.add(issue)
        db.commit()

        # Generate embeddings using API endpoint
        if getattr(issue, "title", None):
            title_embedding = self._create_embedding_via_api(str(issue.title))
            if title_embedding:
                db.execute(
                    text("UPDATE issues SET title_embedding = :embedding WHERE id = :issue_id"),
                    {"embedding": title_embedding, "issue_id": issue.id},
                )
        
        if getattr(issue, "body", None):
            embedding = self._create_embedding_via_api(str(issue.body))
            if embedding:
                db.execute(
                    text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"),
                    {"embedding": embedding, "issue_id": issue.id},
                )
        
        db.commit()
        logger.info(f"Saved issue #{issue.number} to database")
    
    def get_topic_conversation(self, topic_id: int, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch the full conversation for a Discourse topic."""
        url = f"{DISCOURSE_BASE_URL}/t/{slug}/{topic_id}.json"
        headers = self.get_discourse_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching conversation for topic {topic_id}: {e}")
            return None
    
    def save_discourse_post_to_database(self, db: Session, topic_data: Dict[str, Any], conversation_data: Dict[str, Any]) -> None:
        """Save a Discourse post to the database with embeddings."""
        topic_id = topic_data['id']
        
        # Check if post already exists
        existing_post = db.query(DiscoursePost).filter(DiscoursePost.topic_id == topic_id).first()
        if existing_post:
            logger.info(f"Discourse post #{topic_id} already exists in database")
            return
        
        # Combine posts into full conversation
        posts = conversation_data.get('post_stream', {}).get('posts', [])
        conversation = combine_all_discourse_posts(topic_data.get("title", ""), posts)
        creator_username = get_topic_creator_username(topic_data, posts)
        
        # Create new discourse post (align fields with model)
        post = DiscoursePost(
            topic_id=topic_id,
            title=topic_data['title'],
            slug=topic_data['slug'],
            created_at=datetime.fromisoformat(topic_data['created_at'].replace('Z', '+00:00')),
            conversation=conversation,
            token_count=calculate_token_count(conversation)
        )
        
        # Persist first, then update embeddings
        db.add(post)
        db.commit()

        # Generate embeddings using API endpoint
        if getattr(post, "conversation", None):
            embedding = self._create_embedding_via_api(str(post.conversation))
            if embedding:
                db.execute(
                    text("UPDATE discourse_posts SET conversation_embedding = :embedding WHERE id = :post_id"),
                    {"embedding": embedding, "post_id": post.id},
                )
        
        db.commit()
        logger.info(f"Saved discourse post #{post.topic_id} to database")
    
    def call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call_llm(messages)
                return response
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts")
                    return None
    
    def summarize_issue(self, issue: Issue) -> Optional[str]:
        """Generate LLM summary for an issue."""
        try:
            # Create a combined text for summary that includes both title and body
            combined_text = f"Title: {issue.title}\n\nBody: {issue.body or ''}"
            summary = self.llm_client.summarize_text(combined_text)
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize issue #{issue.number}: {e}")
            return None
    
    def summarize_discourse_post(self, post: DiscoursePost) -> Optional[str]:
        """Generate LLM summary for a discourse post."""
        try:
            summary = self.llm_client.summarize_text(str(post.conversation))
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize discourse post #{post.topic_id}: {e}")
            return None
    
    def summarize_metabase_doc(self, doc: MetabaseDoc) -> Optional[str]:
        """Generate LLM summary for a metabase doc."""
        try:
            summary = self.llm_client.summarize_text(str(doc.markdown))
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize metabase doc #{doc.id}: {e}")
            return None
    
    def get_content_for_embedding(self, entity, entity_type: str) -> Dict[str, str]:
        """Get content that needs embeddings for a given entity."""
        content = {}
        
        if entity_type == 'issues':
            if entity.title_embedding is None and entity.title:
                content['title_embedding'] = entity.title
            if entity.issue_embedding is None and entity.body:
                content['issue_embedding'] = entity.body
            if entity.summary_embedding is None and entity.llm_summary:
                content['summary_embedding'] = entity.llm_summary
                
        elif entity_type == 'discourse_posts':
            if entity.conversation_embedding is None and entity.conversation:
                content['conversation_embedding'] = entity.conversation
            if entity.summary_embedding is None and entity.llm_summary:
                content['summary_embedding'] = entity.llm_summary
                
        elif entity_type == 'metabase_docs':
            if entity.markdown_embedding is None and entity.markdown:
                content['markdown_embedding'] = entity.markdown
            if entity.summary_embedding is None and entity.llm_summary:
                content['summary_embedding'] = entity.llm_summary
                
        elif entity_type == 'questions':
            if entity.question_embedding is None and entity.question:
                content['question_embedding'] = entity.question
            if entity.answer_embedding is None and entity.answer:
                content['answer_embedding'] = entity.answer
        
        return content
    
    def process_issue_embeddings(self, issue: Issue, db: Session) -> Dict[str, bool]:
        """Process embeddings for an issue."""
        results = {'title_embedding': False, 'issue_embedding': False, 'summary_embedding': False}
        
        try:
            content = self.get_content_for_embedding(issue, 'issues')
            
            if 'title_embedding' in content:
                embedding = self._create_embedding_via_api(content['title_embedding'])
                if embedding:
                    db.execute(text("UPDATE issues SET title_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                    results['title_embedding'] = True
                    logger.info(f"Generated title embedding for issue #{issue.number}")
            
            if 'issue_embedding' in content:
                embedding = self._create_embedding_via_api(content['issue_embedding'])
                if embedding:
                    db.execute(text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                    results['issue_embedding'] = True
                    logger.info(f"Generated issue embedding for issue #{issue.number}")
            
            if 'summary_embedding' in content:
                embedding = self._create_embedding_via_api(content['summary_embedding'])
                if embedding:
                    db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                    results['summary_embedding'] = True
                    logger.info(f"Generated summary embedding for issue #{issue.number}")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for issue #{issue.number}: {e}")
        
        return results
    
    def process_discourse_embeddings(self, post: DiscoursePost, db: Session) -> Dict[str, bool]:
        """Process embeddings for a discourse post."""
        results = {'conversation_embedding': False, 'summary_embedding': False}
        
        try:
            content = self.get_content_for_embedding(post, 'discourse_posts')
            
            if 'conversation_embedding' in content:
                embedding = self._create_embedding_via_api(content['conversation_embedding'])
                if embedding:
                    db.execute(text("UPDATE discourse_posts SET conversation_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                    results['conversation_embedding'] = True
                    logger.info(f"Generated conversation embedding for discourse post #{post.topic_id}")
            
            if 'summary_embedding' in content:
                embedding = self._create_embedding_via_api(content['summary_embedding'])
                if embedding:
                    db.execute(text("UPDATE discourse_posts SET summary_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                    results['summary_embedding'] = True
                    logger.info(f"Generated summary embedding for discourse post #{post.topic_id}")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for discourse post #{post.topic_id}: {e}")
        
        return results
    
    def process_metabase_embeddings(self, doc: MetabaseDoc, db: Session) -> Dict[str, bool]:
        """Process embeddings for a metabase doc."""
        results = {'markdown_embedding': False, 'summary_embedding': False}
        
        try:
            content = self.get_content_for_embedding(doc, 'metabase_docs')
            
            if 'markdown_embedding' in content:
                embedding = self._create_embedding_via_api(content['markdown_embedding'])
                if embedding:
                    db.execute(text("UPDATE metabase_docs SET markdown_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                    results['markdown_embedding'] = True
                    logger.info(f"Generated markdown embedding for metabase doc #{doc.id}")
            
            if 'summary_embedding' in content:
                embedding = self._create_embedding_via_api(content['summary_embedding'])
                if embedding:
                    db.execute(text("UPDATE metabase_docs SET summary_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                    results['summary_embedding'] = True
                    logger.info(f"Generated summary embedding for metabase doc #{doc.id}")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for metabase doc #{doc.id}: {e}")
        
        return results
    
    def process_question_embeddings(self, question: Question, db: Session) -> Dict[str, bool]:
        """Process embeddings for a question."""
        results = {'question_embedding': False, 'answer_embedding': False}
        
        try:
            content = self.get_content_for_embedding(question, 'questions')
            
            if 'question_embedding' in content:
                embedding = self._create_embedding_via_api(content['question_embedding'])
                if embedding:
                    db.execute(text("UPDATE questions SET question_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                    results['question_embedding'] = True
                    logger.info(f"Generated question embedding for question #{question.id}")
            
            if 'answer_embedding' in content:
                embedding = self._create_embedding_via_api(content['answer_embedding'])
                if embedding:
                    db.execute(text("UPDATE questions SET answer_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                    results['answer_embedding'] = True
                    logger.info(f"Generated answer embedding for question #{question.id}")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for question #{question.id}: {e}")
        
        return results
    
    def process_keyword_embeddings(self, keyword: KeywordDefinition, db: Session) -> Dict[str, bool]:
        """Process embeddings for a keyword definition."""
        results = {'keyword_embedding': False}
        
        try:
            # Get synonyms for this keyword
            synonyms = db.query(Synonym).filter(Synonym.synonym_of == keyword.keyword).all()
            synonym_list = [syn.word for syn in synonyms]
            
            # Create text for embedding: keyword + definition + synonyms
            text_to_embed = f"keyword: {keyword.keyword}\ndefinition: {keyword.definition}"
            if synonym_list:
                text_to_embed += f"\nsynonyms: {', '.join(str(s) for s in synonym_list)}"
            
            embedding = self._create_embedding_via_api(text_to_embed)
            if embedding:
                db.execute(text("UPDATE keyword_definitions SET keyword_embedding = :embedding WHERE id = :keyword_id"), {"embedding": str(embedding), "keyword_id": keyword.id})
                results['keyword_embedding'] = True
                logger.info(f"Generated keyword embedding for keyword '{keyword.keyword}'")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for keyword '{keyword.keyword}': {e}")
        
        return results
    
    def process_synonym_embeddings(self, synonym: Synonym, db: Session) -> Dict[str, bool]:
        """Process embeddings for a synonym."""
        results = {'word_embedding': False, 'synonym_embedding': False}
        
        try:
            # Generate word embedding
            if synonym.word_embedding is None:
                word_embedding = self._create_embedding_via_api(str(synonym.word))
                if word_embedding:
                    db.execute(text("UPDATE synonyms SET word_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(word_embedding), "synonym_id": synonym.id})
                    results['word_embedding'] = True
                    logger.info(f"Generated word embedding for synonym '{synonym.word}'")
            
            # Generate synonym relationship embedding
            if synonym.synonym_embedding is None:
                synonym_text = f"word: {synonym.word}\nsynonym_of: {synonym.synonym_of}"
                synonym_embedding = self._create_embedding_via_api(synonym_text)
                if synonym_embedding:
                    db.execute(text("UPDATE synonyms SET synonym_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(synonym_embedding), "synonym_id": synonym.id})
                    results['synonym_embedding'] = True
                    logger.info(f"Generated synonym embedding for '{synonym.word}' -> '{synonym.synonym_of}'")
                
        except Exception as e:
            logger.error(f"Failed to process embeddings for synonym '{synonym.word}': {e}")
        
        return results
    
    def run_monitoring_cycle(self, db: Session) -> Dict[str, int]:
        """Run a single monitoring cycle based on the worker type."""
        processed_counts = {}
        
        try:
            if self.monitor_type == MonitorType.GITHUB_ISSUES:
                processed_counts = self._monitor_github_issues(db)
            elif self.monitor_type == MonitorType.DISCOURSE_POSTS:
                processed_counts = self._monitor_discourse_posts(db)
            elif self.monitor_type == MonitorType.LLM_SUMMARIES:
                processed_counts = self._monitor_llm_summaries(db)
            elif self.monitor_type == MonitorType.EMBEDDINGS:
                processed_counts = self._monitor_embeddings(db)
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            processed_counts = {}
        
        return processed_counts
    
    def _create_embedding_via_api(self, text: str) -> Optional[List[float]]:
        """Create an embedding using the API endpoint."""
        try:
            headers = {"X-API-Key": str(API_KEY)}
            payload = {"text": text}
            response = requests.post("http://localhost:8000/embedding", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get('embedding')
        except Exception as e:
            logger.error(f"Failed to create embedding via API: {e}")
            return None
    
    def _monitor_github_issues(self, db: Session) -> Dict[str, int]:
        """Monitor for new GitHub issues."""
        latest_issue = self.get_latest_entity_from_db(db)
        new_issues = self.get_new_github_issues(latest_issue)
        
        processed_count = 0
        
        for issue_data in new_issues:
            # Skip issues created by github-actions
            if issue_data['user']['login'] == 'github-actions[bot]':
                logger.info(f"Skipping issue #{issue_data['number']} created by github-actions")
                continue
                
            try:
                # Find similar issues first (before saving to database)
                issue_text = f"{issue_data['title']}\n\n{issue_data.get('body', '')}"
                similar_issues = self.find_similar_issues(issue_text)
                
                # Post comment if similar issues found
                if similar_issues:
                    self.post_comment_on_issue(
                        issue_data['number'], 
                        similar_issues, 
                        issue_data['user']['login']
                    )
                
                # Save issue to database after posting comment
                self.save_github_issue_to_database(db, issue_data)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing issue #{issue_data['number']}: {e}")
        
        return {'issues_processed': processed_count}
    
    def _monitor_discourse_posts(self, db: Session) -> Dict[str, int]:
        """Monitor for new Discourse posts."""
        latest_topic = self.get_latest_entity_from_db(db)
        new_topics = self.get_new_discourse_topics(latest_topic)
        
        processed_count = 0
        
        for topic_data in new_topics:
            try:
                # Get full conversation
                conversation_data = self.get_topic_conversation(topic_data['id'], topic_data['slug'])
                if not conversation_data:
                    continue
                
                # Save to database
                self.save_discourse_post_to_database(db, topic_data, conversation_data)
                
                # Find similar issues
                topic_text = f"{topic_data['title']}\n\n{conversation_data.get('post_stream', {}).get('posts', [])}"
                similar_issues = self.find_similar_issues(topic_text)
                
                if similar_issues:
                    logger.info(f"Found {len(similar_issues)} similar issues for discourse topic #{topic_data['id']}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing discourse topic #{topic_data['id']}: {e}")
        
        return {'discourse_posts_processed': processed_count}
    
    def _monitor_llm_summaries(self, db: Session) -> Dict[str, int]:
        """Monitor for entities lacking LLM summaries."""
        entities = self.get_entities_lacking_summaries(db)
        
        processed_counts = {
            'issues_summarized': 0,
            'discourse_posts_summarized': 0,
            'metabase_docs_summarized': 0
        }
        
        # Process issues
        for issue in entities['issues']:
            try:
                summary = self.summarize_issue(issue)
                if summary:
                    issue.llm_summary = summary
                    db.commit()
                    processed_counts['issues_summarized'] += 1
                    logger.info(f"Generated summary for issue #{issue.number}")
            except Exception as e:
                logger.error(f"Failed to summarize issue #{issue.number}: {e}")
        
        # Process discourse posts
        for post in entities['discourse_posts']:
            try:
                summary = self.summarize_discourse_post(post)
                if summary:
                    post.llm_summary = summary
                    db.commit()
                    processed_counts['discourse_posts_summarized'] += 1
                    logger.info(f"Generated summary for discourse post #{post.topic_id}")
            except Exception as e:
                logger.error(f"Failed to summarize discourse post #{post.topic_id}: {e}")
        
        # Process metabase docs
        for doc in entities['metabase_docs']:
            try:
                summary = self.summarize_metabase_doc(doc)
                if summary:
                    doc.llm_summary = summary
                    db.commit()
                    processed_counts['metabase_docs_summarized'] += 1
                    logger.info(f"Generated summary for metabase doc #{doc.id}")
            except Exception as e:
                logger.error(f"Failed to summarize metabase doc #{doc.id}: {e}")
        
        return processed_counts
    
    def _monitor_embeddings(self, db: Session) -> Dict[str, int]:
        """Monitor for entities lacking embeddings."""
        entities = self.get_entities_lacking_embeddings(db)
        
        processed_counts = {
            'issues_embeddings_processed': 0,
            'discourse_posts_embeddings_processed': 0,
            'metabase_docs_embeddings_processed': 0,
            'questions_embeddings_processed': 0,
            'keyword_definitions_embeddings_processed': 0,
            'synonyms_embeddings_processed': 0
        }
        
        # Process issues
        logger.info(f"Processing {len(entities['issues'])} issues...")
        for issue in entities['issues']:
            try:
                results = self.process_issue_embeddings(issue, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['issues_embeddings_processed'] += 1
                    logger.info(f"Successfully processed embeddings for issue #{issue.number}")
            except Exception as e:
                logger.error(f"Failed to process embeddings for issue #{issue.number}: {e}")
        
        # Process discourse posts
        for post in entities['discourse_posts']:
            try:
                results = self.process_discourse_embeddings(post, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['discourse_posts_embeddings_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process embeddings for discourse post #{post.topic_id}: {e}")
        
        # Process metabase docs
        for doc in entities['metabase_docs']:
            try:
                results = self.process_metabase_embeddings(doc, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['metabase_docs_embeddings_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process embeddings for metabase doc #{doc.id}: {e}")
        
        # Process questions
        for question in entities['questions']:
            try:
                results = self.process_question_embeddings(question, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['questions_embeddings_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process embeddings for question #{question.id}: {e}")
        
        # Process keyword definitions
        for keyword in entities['keyword_definitions']:
            try:
                results = self.process_keyword_embeddings(keyword, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['keyword_definitions_embeddings_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process embeddings for keyword '{keyword.keyword}': {e}")
        
        # Process synonyms
        for synonym in entities['synonyms']:
            try:
                results = self.process_synonym_embeddings(synonym, db)
                if any(results.values()):
                    db.commit()
                    processed_counts['synonyms_embeddings_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process embeddings for synonym '{synonym.word}': {e}")
        
        return processed_counts

def main():
    """Main entry point for the unified monitor worker."""
    if len(sys.argv) < 2:
        print("Unified Monitor Worker for GitHub Duplicate Issue Finder")
        print("=" * 60)
        print("\nUsage:")
        print("  python monitor_worker.py <monitor_type>")
        print("\nAvailable monitor types:")
        print("  github_issues    - Monitor GitHub repository for new issues")
        print("  discourse_posts  - Monitor Discourse forum for new posts")
        print("  llm_summaries    - Monitor for entities lacking LLM summaries")
        print("  embeddings       - Monitor for entities lacking embeddings")
        print("\nExamples:")
        print("  python monitor_worker.py github_issues")
        print("  python monitor_worker.py discourse_posts")
        print("  python monitor_worker.py llm_summaries")
        print("  python monitor_worker.py embeddings")
        return
    
    monitor_type_str = sys.argv[1]
    
    try:
        monitor_type = MonitorType(monitor_type_str)
    except ValueError:
        print(f"❌ Unknown monitor type: {monitor_type_str}")
        print("Available types: github_issues, discourse_posts, llm_summaries, embeddings")
        return
    
    # Initialize worker
    worker = UnifiedMonitorWorker(monitor_type)
    
    print(f"🚀 Starting {monitor_type.value} monitor worker...")
    print(f"Monitor type: {monitor_type.value}")
    print("-" * 50)
    
    # Main monitoring loop
    while True:
        try:
            db = SessionLocal()
            processed_counts = worker.run_monitoring_cycle(db)
            
            if processed_counts:
                total_processed = sum(processed_counts.values())
                if total_processed > 0:
                    logger.info(f"Processed: {processed_counts}")
                else:
                    logger.info("No new entities to process")
            else:
                logger.info("No entities processed in this cycle")
            
            db.close()
            
            # Sleep between cycles
            sleep_interval = WORKER_POLL_INTERVAL_SECONDS
            logger.info(f"Sleeping for {sleep_interval} seconds...")
            time.sleep(sleep_interval)
            
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(WORKER_BACKOFF_SECONDS)

if __name__ == "__main__":
    main() 