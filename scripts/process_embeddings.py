#!/usr/bin/env python3
"""
Embedding and LLM processing script for GitHub Duplicate Issue Finder.
This script handles embedding generation and LLM calls for various content types.
"""

import os
import sys
import asyncio
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import argparse

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import Session
from sqlalchemy import text
import httpx

from src.db import SessionLocal
from src.embedding_service import get_embedding_service
from src.utils import clean_llm_json_response
from json_repair import repair_json
from src.settings import LITELLM_MODEL_NAME
from src.models import Issue, DiscoursePost, MetabaseDoc, Question, KeywordDefinition, Synonym
from src.keyword_service import KeywordService
from src.prompts import get_questions_generation_prompt
from pydantic import BaseModel, ValidationError, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingModes:
    """Processing modes for different types of content."""
    LLM_QUESTIONS_METABASE = 'llm-questions-metabase'
    LLM_QUESTIONS_ISSUES = 'llm-questions-issues'
    LLM_QUESTIONS_DISCOURSE = 'llm-questions-discourse'
    MARKDOWN_ONLY = 'markdown-only'
    ALL_EMBEDDINGS = 'all-embeddings'
    KEYWORD_EMBEDDINGS = 'keyword-embeddings'
    DOCS_EMBEDDINGS = 'docs-embeddings'
    ISSUES_EMBEDDINGS = 'issues-embeddings'
    POSTS_EMBEDDINGS = 'posts-embeddings'
    QUESTIONS_EMBEDDINGS = 'questions-embeddings'
    SUMMARY_EMBEDDINGS = 'summary-embeddings'
    SYNONYM_EMBEDDINGS = 'synonym-embeddings'

class SourceTypes:
    """Source types mapping."""
    METABASE_DOC = 'metabase_doc'
    ISSUE = 'issue'
    DISCOURSE_POST = 'discourse_post'

class EmbeddingProcessor:
    """Handles embedding generation and LLM calls."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.embedding_service = get_embedding_service()
        # Back-compat alias so legacy calls still work
        self.embedding_client = self.embedding_service
        from src.settings import HTTPX_TIMEOUT
        self.client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)
        self.keyword_service = KeywordService()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def get_content_field(self, source_type: str) -> str:
        """Get the content field name for a source type."""
        field_mapping = {
            SourceTypes.METABASE_DOC: 'markdown',
            SourceTypes.ISSUE: 'body',
            SourceTypes.DISCOURSE_POST: 'conversation'
        }
        return field_mapping.get(source_type, 'markdown')
    
    def get_table_name(self, source_type: str) -> str:
        """Get the table name for a source type."""
        table_mapping = {
            SourceTypes.METABASE_DOC: 'metabase_docs',
            SourceTypes.ISSUE: 'issues',
            SourceTypes.DISCOURSE_POST: 'discourse_posts'
        }
        return table_mapping.get(source_type, 'metabase_docs')
    
    def get_prompt_for_source_type(self, source_type: str, content: str, db: Session) -> str:
        """Generate a prompt for LLM based on source type."""
        source_descriptions = {
            SourceTypes.METABASE_DOC: 'Metabase knowledge base article',
            SourceTypes.ISSUE: 'GitHub issue',
            SourceTypes.DISCOURSE_POST: 'Discourse forum post/conversation'
        }
        
        specific_instructions = {
            SourceTypes.METABASE_DOC: 'Please pay attention to the concepts covered in the article to formulate relevant questions. If there\'s a list of supported databases or supported functions, create one question for each database or supported function like \'Does Metabase support connecting to database x?\' or \'Does Metabase support function y?\'',
            SourceTypes.ISSUE: 'Please focus on the problem described, any solutions mentioned, and technical details. Create questions about the issue, its resolution, and any workarounds or fixes discussed.',
            SourceTypes.DISCOURSE_POST: 'Please focus on the main topic discussed, any solutions provided, and questions asked by users. Create questions about the discussion topic, solutions mentioned, and community responses.'
        }
        
        source_description = source_descriptions.get(source_type, 'document')
        specific_instruction = specific_instructions.get(source_type, '')
        
        base_prompt = get_questions_generation_prompt(source_description, content)
        
        if specific_instruction:
            base_prompt += f"\n\n{specific_instruction}"
        
        # Inject keyword definitions into the prompt
        return self.keyword_service.inject_keywords_into_prompt(base_prompt, db)
    
    async def create_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Create an embedding with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                embedding = self.embedding_client.create_embedding(text)
                return embedding
            except Exception as e:
                logger.error(f"Error creating embedding (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return None
                await asyncio.sleep(1)
        return None
    
    async def call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.config['litellm_url']}/chat/completions",
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.config["api_key"]}'
                    },
                    json={
                        'model': self.config['model_name'],
                        'messages': [
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ],
                        'max_tokens': 4000,
                        'temperature': 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = 60  # Default 60 seconds for rate limit
                    logger.warning(f"Rate limit exceeded (attempt {attempt}/{max_retries}). Waiting {wait_time} seconds...")
                    if attempt == max_retries:
                        logger.error(f"Failed to call LLM after {max_retries} attempts due to rate limiting")
                        return None
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM API error: {response.status_code} {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error calling LLM (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return None
                await asyncio.sleep(1)
        
        return None
    
    def parse_llm_questions_response(self, response: str, doc_id: int) -> List[Dict[str, str]]:
        """Parse LLM response to extract questions and answers."""
        class QAItem(BaseModel):
            question: str = Field(min_length=1)
            answer: str = Field(min_length=1)
        class QASchema(BaseModel):
            questions: List[QAItem]
        try:
            cleaned = clean_llm_json_response(response)
            repaired = repair_json(cleaned)
            parsed = json.loads(repaired)
            try:
                validated = QASchema.model_validate(parsed)
                return [
                    {'question': qa.question.strip(), 'answer': qa.answer.strip()}
                    for qa in validated.questions
                ]
            except ValidationError as ve:
                logger.warning(f"Schema validation failed for doc {doc_id}: {ve}")
                if parsed.get('questions') and isinstance(parsed['questions'], list):
                    return [
                        {
                            'question': str(qa.get('question', '')).strip(),
                            'answer': str(qa.get('answer', '')).strip(),
                        }
                        for qa in parsed['questions']
                        if isinstance(qa, dict)
                    ]
            logger.warning(f"Could not parse JSON response for document ID {doc_id}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse LLM response for document ID {doc_id}: {e}")
            logger.debug(f"Response: {response[:200]}...")
            return []
    
    async def process_markdown_batch(self, db: Session) -> int:
        """Process a batch of markdown embeddings."""
        try:
            logger.info("Processing markdown embeddings batch...")
            
            # Get documents without markdown embeddings
            result = db.execute(
                text("SELECT id, markdown FROM metabase_docs WHERE markdown_embedding IS NULL AND markdown IS NOT NULL LIMIT :limit"),
                {"limit": self.config['batch_size']}
            ).fetchall()
            
            if not result:
                logger.info("No documents need markdown embeddings")
                return 0
            
            logger.info(f"Found {len(result)} documents needing markdown embeddings")
            
            processed_count = 0
            
            for row in result:
                try:
                    # Create embedding for the markdown content
                    embedding = await self.create_embedding_with_retry(row.markdown)
                    
                    if embedding:
                        # Update the markdown_embedding field
                        db.execute(
                            text("UPDATE metabase_docs SET markdown_embedding = :embedding, updated_at = NOW() WHERE id = :id"),
                            {"embedding": embedding, "id": row.id}
                        )
                        
                        logger.info(f"Updated markdown embedding for document ID {row.id}")
                        processed_count += 1
                    else:
                        logger.error(f"Failed to create embedding for document ID {row.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing document ID {row.id}: {e}")
            
            db.commit()
            logger.info(f"Batch completed: {processed_count}/{len(result)} documents processed")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing markdown batch: {e}")
            db.rollback()
            return 0

    def process_all_embeddings_batch(self, db: Session) -> int:
        """Process all NULL embeddings across all tables using API endpoint."""
        try:
            logger.info("Processing all NULL embeddings...")
            
            total_processed = 0
            
            # 1. Process metabase_docs markdown embeddings
            logger.info("Processing metabase_docs markdown embeddings...")
            metabase_docs = db.query(MetabaseDoc).filter(MetabaseDoc.markdown_embedding == None, MetabaseDoc.markdown != None).limit(self.config['batch_size']).all()
            for doc in metabase_docs:
                try:
                    embedding = self.embedding_client.create_embedding(str(doc.markdown))
                    if embedding:
                        db.execute(text("UPDATE metabase_docs SET markdown_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                        logger.info(f"Generated markdown embedding for metabase_doc ID {doc.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate markdown embedding for metabase_doc ID {doc.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate markdown embedding for metabase_doc ID {doc.id}: {e}")
            
            # 2. Process metabase_docs summary embeddings (if LLM summaries exist)
            logger.info("Processing metabase_docs summary embeddings...")
            metabase_docs_summaries = db.query(MetabaseDoc).filter(MetabaseDoc.summary_embedding == None, MetabaseDoc.llm_summary != None).limit(self.config['batch_size']).all()
            for doc in metabase_docs_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(doc.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE metabase_docs SET summary_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                        logger.info(f"Generated summary embedding for metabase_doc ID {doc.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: {e}")
            
            # 3. Process issues embeddings (combined title + body)
            logger.info("Processing issues embeddings...")
            issues = db.query(Issue).filter(
                Issue.issue_embedding == None, 
                Issue.body != None,
                Issue.title != None,
                Issue.title != ''
            ).limit(self.config['batch_size']).all()
            for issue in issues:
                try:
                    text_to_embed = f"{issue.title}\n{issue.body or ''}"
                    embedding = self.embedding_client.create_embedding(text_to_embed)
                    if embedding:
                        db.execute(text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        logger.info(f"Generated issue embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate issue embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate issue embedding for issue ID {issue.id}: {e}")
            
            # 3.5. Process title embeddings (separate from issue embeddings)
            logger.info("Processing title embeddings...")
            issues_without_title_embedding = db.query(Issue).filter(
                Issue.title_embedding == None,
                Issue.title != None,
                Issue.title != ''
            ).limit(self.config['batch_size']).all()
            for issue in issues_without_title_embedding:
                try:
                    title_embedding = self.embedding_client.create_embedding(str(issue.title))
                    if title_embedding:
                        db.execute(text("UPDATE issues SET title_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(title_embedding), "issue_id": issue.id})
                        logger.info(f"Generated title embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate title embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate title embedding for issue ID {issue.id}: {e}")
            
            # 4. Process issues summary embeddings (if LLM summaries exist)
            logger.info("Processing issues summary embeddings...")
            issues_summaries = db.query(Issue).filter(Issue.summary_embedding == None, Issue.llm_summary != None).limit(self.config['batch_size']).all()
            for issue in issues_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(issue.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        logger.info(f"Generated summary embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: {e}")
            
            # 5. Process discourse_posts embeddings
            logger.info("Processing discourse_posts embeddings...")
            discourse_posts = db.query(DiscoursePost).filter(DiscoursePost.conversation_embedding == None, DiscoursePost.conversation != None).limit(self.config['batch_size']).all()
            for post in discourse_posts:
                try:
                    embedding = self.embedding_client.create_embedding(str(post.conversation))
                    if embedding:
                        db.execute(text("UPDATE discourse_posts SET conversation_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                        logger.info(f"Generated conversation embedding for discourse_post ID {post.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate conversation embedding for discourse_post ID {post.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate conversation embedding for discourse_post ID {post.id}: {e}")
            
            # 6. Process discourse_posts summary embeddings (if LLM summaries exist)
            logger.info("Processing discourse_posts summary embeddings...")
            discourse_summaries = db.query(DiscoursePost).filter(DiscoursePost.summary_embedding == None, DiscoursePost.llm_summary != None).limit(self.config['batch_size']).all()
            for post in discourse_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(post.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE discourse_posts SET summary_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                        logger.info(f"Generated summary embedding for discourse_post ID {post.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: {e}")
            
            # 7. Process questions embeddings
            logger.info("Processing questions embeddings...")
            questions_q = db.query(Question).filter(Question.question_embedding == None, Question.question != None).limit(self.config['batch_size']).all()
            for question in questions_q:
                try:
                    embedding = self.embedding_client.create_embedding(str(question.question))
                    if embedding:
                        db.execute(text("UPDATE questions SET question_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                        logger.info(f"Generated question embedding for question ID {question.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate question embedding for question ID {question.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate question embedding for question ID {question.id}: {e}")
            
            # 8. Process answer embeddings
            logger.info("Processing answer embeddings...")
            questions_a = db.query(Question).filter(Question.answer_embedding == None, Question.answer != None).limit(self.config['batch_size']).all()
            for question in questions_a:
                try:
                    embedding = self.embedding_client.create_embedding(str(question.answer))
                    if embedding:
                        db.execute(text("UPDATE questions SET answer_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                        logger.info(f"Generated answer embedding for question ID {question.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate answer embedding for question ID {question.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate answer embedding for question ID {question.id}: {e}")
            
            # 9. Process keyword embeddings
            logger.info("Processing keyword embeddings...")
            keywords = db.query(KeywordDefinition).filter(KeywordDefinition.keyword_embedding == None).limit(self.config['batch_size']).all()
            for keyword in keywords:
                try:
                    # Get synonyms for this keyword
                    synonyms = db.query(Synonym).filter(Synonym.synonym_of == keyword.keyword).all()
                    synonym_list = [syn.word for syn in synonyms]
                    
                    # Create text for embedding: keyword + definition + synonyms
                    text_to_embed = f"keyword: {keyword.keyword}\ndefinition: {keyword.definition}"
                    if synonym_list:
                        text_to_embed += f"\nsynonyms: {', '.join(str(s) for s in synonym_list)}"
                    
                    embedding = self.embedding_client.create_embedding(text_to_embed)
                    if embedding:
                        db.execute(text("UPDATE keyword_definitions SET keyword_embedding = :embedding WHERE id = :keyword_id"), {"embedding": str(embedding), "keyword_id": keyword.id})
                        logger.info(f"Generated keyword embedding for keyword '{keyword.keyword}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate keyword embedding for keyword '{keyword.keyword}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate keyword embedding for keyword '{keyword.keyword}': {e}")
            
            # 10. Process synonym word embeddings
            logger.info("Processing synonym word embeddings...")
            synonyms_word = db.query(Synonym).filter(Synonym.word_embedding == None).limit(self.config['batch_size']).all()
            for synonym in synonyms_word:
                try:
                    embedding = self.embedding_client.create_embedding(str(synonym.word))
                    if embedding:
                        db.execute(text("UPDATE synonyms SET word_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(embedding), "synonym_id": synonym.id})
                        logger.info(f"Generated word embedding for synonym '{synonym.word}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate word embedding for synonym '{synonym.word}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate word embedding for synonym '{synonym.word}': {e}")
            
            # 11. Process synonym relationship embeddings
            logger.info("Processing synonym relationship embeddings...")
            synonyms_rel = db.query(Synonym).filter(Synonym.synonym_embedding == None).limit(self.config['batch_size']).all()
            for synonym in synonyms_rel:
                try:
                    synonym_text = f"word: {synonym.word}\nsynonym_of: {synonym.synonym_of}"
                    embedding = self.embedding_client.create_embedding(synonym_text)
                    if embedding:
                        db.execute(text("UPDATE synonyms SET synonym_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(embedding), "synonym_id": synonym.id})
                        logger.info(f"Generated synonym relationship embedding for '{synonym.word}' -> '{synonym.synonym_of}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate synonym embedding for '{synonym.word}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate synonym embedding for '{synonym.word}': {e}")
            
            db.commit()
            logger.info(f"All embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_all_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_keyword_embeddings_batch(self, db: Session) -> int:
        """Process keyword embeddings only."""
        try:
            logger.info("Processing keyword embeddings...")
            
            total_processed = 0
            
            # Process keyword embeddings
            keywords = db.query(KeywordDefinition).filter(KeywordDefinition.keyword_embedding == None).limit(self.config['batch_size']).all()
            for keyword in keywords:
                try:
                    # Get synonyms for this keyword
                    synonyms = db.query(Synonym).filter(Synonym.synonym_of == keyword.keyword).all()
                    synonym_list = [syn.word for syn in synonyms]
                    
                    # Create text for embedding: keyword + definition + synonyms
                    text_to_embed = f"keyword: {keyword.keyword}\ndefinition: {keyword.definition}"
                    if synonym_list:
                        text_to_embed += f"\nsynonyms: {', '.join(str(s) for s in synonym_list)}"
                    
                    embedding = self.embedding_client.create_embedding(text_to_embed)
                    if embedding:
                        db.execute(text("UPDATE keyword_definitions SET keyword_embedding = :embedding WHERE id = :keyword_id"), {"embedding": str(embedding), "keyword_id": keyword.id})
                        logger.info(f"Generated keyword embedding for keyword '{keyword.keyword}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate keyword embedding for keyword '{keyword.keyword}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate keyword embedding for keyword '{keyword.keyword}': {e}")
            
            db.commit()
            logger.info(f"Keyword embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_keyword_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_docs_embeddings_batch(self, db: Session) -> int:
        """Process docs embeddings (markdown and summary)."""
        try:
            logger.info("Processing docs embeddings...")
            
            total_processed = 0
            
            # Process markdown embeddings
            docs_markdown = db.query(MetabaseDoc).filter(MetabaseDoc.markdown_embedding == None, MetabaseDoc.markdown != None).limit(self.config['batch_size']).all()
            for doc in docs_markdown:
                try:
                    embedding = self.embedding_client.create_embedding(str(doc.markdown))
                    if embedding:
                        db.execute(text("UPDATE metabase_docs SET markdown_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                        logger.info(f"Generated markdown embedding for metabase_doc ID {doc.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate markdown embedding for metabase_doc ID {doc.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate markdown embedding for metabase_doc ID {doc.id}: {e}")
            
            # Process summary embeddings
            docs_summaries = db.query(MetabaseDoc).filter(MetabaseDoc.summary_embedding == None, MetabaseDoc.llm_summary != None).limit(self.config['batch_size']).all()
            for doc in docs_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(doc.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE metabase_docs SET summary_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                        logger.info(f"Generated summary embedding for metabase_doc ID {doc.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: {e}")
            
            db.commit()
            logger.info(f"Docs embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_docs_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_issues_embeddings_batch(self, db: Session) -> int:
        """Process issues embeddings (issue and summary)."""
        try:
            logger.info("Processing issues embeddings...")
            
            total_processed = 0
            
            # Process issue embeddings (combined title + body)
            issues = db.query(Issue).filter(
                Issue.issue_embedding == None, 
                Issue.body != None,
                Issue.title != None,
                Issue.title != ''
            ).limit(self.config['batch_size']).all()
            for issue in issues:
                try:
                    text_to_embed = f"{issue.title}\n{issue.body or ''}"
                    embedding = self.embedding_client.create_embedding(text_to_embed)
                    if embedding:
                        db.execute(text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        logger.info(f"Generated issue embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate issue embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate issue embedding for issue ID {issue.id}: {e}")
            
            # Process title embeddings (separate from issue embeddings)
            issues_without_title_embedding = db.query(Issue).filter(
                Issue.title_embedding == None,
                Issue.title != None,
                Issue.title != ''
            ).limit(self.config['batch_size']).all()
            for issue in issues_without_title_embedding:
                try:
                    title_embedding = self.embedding_client.create_embedding(str(issue.title))
                    if title_embedding:
                        db.execute(text("UPDATE issues SET title_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(title_embedding), "issue_id": issue.id})
                        logger.info(f"Generated title embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate title embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate title embedding for issue ID {issue.id}: {e}")
            
            # Process summary embeddings
            issues_summaries = db.query(Issue).filter(Issue.summary_embedding == None, Issue.llm_summary != None).limit(self.config['batch_size']).all()
            for issue in issues_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(issue.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        logger.info(f"Generated summary embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: {e}")
            
            db.commit()
            logger.info(f"Issues embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_issues_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_posts_embeddings_batch(self, db: Session) -> int:
        """Process discourse posts embeddings (conversation and summary)."""
        try:
            logger.info("Processing discourse posts embeddings...")
            
            total_processed = 0
            
            # Process conversation embeddings
            posts = db.query(DiscoursePost).filter(DiscoursePost.conversation_embedding == None, DiscoursePost.conversation != None).limit(self.config['batch_size']).all()
            for post in posts:
                try:
                    embedding = self.embedding_client.create_embedding(str(post.conversation))
                    if embedding:
                        db.execute(text("UPDATE discourse_posts SET conversation_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                        logger.info(f"Generated conversation embedding for discourse_post ID {post.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate conversation embedding for discourse_post ID {post.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate conversation embedding for discourse_post ID {post.id}: {e}")
            
            # Process summary embeddings
            posts_summaries = db.query(DiscoursePost).filter(DiscoursePost.summary_embedding == None, DiscoursePost.llm_summary != None).limit(self.config['batch_size']).all()
            for post in posts_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(post.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE discourse_posts SET summary_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                        logger.info(f"Generated summary embedding for discourse_post ID {post.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: {e}")
            
            db.commit()
            logger.info(f"Posts embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_posts_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_questions_embeddings_batch(self, db: Session) -> int:
        """Process questions embeddings (question and answer)."""
        try:
            logger.info("Processing questions embeddings...")
            
            total_processed = 0
            
            # Process question embeddings
            questions_q = db.query(Question).filter(Question.question_embedding == None, Question.question != None).limit(self.config['batch_size']).all()
            for question in questions_q:
                try:
                    embedding = self.embedding_client.create_embedding(str(question.question))
                    if embedding:
                        db.execute(text("UPDATE questions SET question_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                        logger.info(f"Generated question embedding for question ID {question.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate question embedding for question ID {question.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate question embedding for question ID {question.id}: {e}")
            
            # Process answer embeddings
            questions_a = db.query(Question).filter(Question.answer_embedding == None, Question.answer != None).limit(self.config['batch_size']).all()
            for question in questions_a:
                try:
                    embedding = self.embedding_client.create_embedding(str(question.answer))
                    if embedding:
                        db.execute(text("UPDATE questions SET answer_embedding = :embedding WHERE id = :question_id"), {"embedding": str(embedding), "question_id": question.id})
                        logger.info(f"Generated answer embedding for question ID {question.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate answer embedding for question ID {question.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate answer embedding for question ID {question.id}: {e}")
            
            db.commit()
            logger.info(f"Questions embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_questions_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_summary_embeddings_batch(self, db: Session) -> int:
        """Process summary embeddings for all entities."""
        try:
            logger.info("Processing summary embeddings for all entities...")
            
            total_processed = 0
            
            # Process metabase_docs summary embeddings
            docs_summaries = db.query(MetabaseDoc).filter(MetabaseDoc.summary_embedding == None, MetabaseDoc.llm_summary != None).limit(self.config['batch_size']).all()
            for doc in docs_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(doc.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE metabase_docs SET summary_embedding = :embedding WHERE id = :doc_id"), {"embedding": str(embedding), "doc_id": doc.id})
                        logger.info(f"Generated summary embedding for metabase_doc ID {doc.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for metabase_doc ID {doc.id}: {e}")
            
            # Process issues summary embeddings
            issues_summaries = db.query(Issue).filter(Issue.summary_embedding == None, Issue.llm_summary != None).limit(self.config['batch_size']).all()
            for issue in issues_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(issue.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        logger.info(f"Generated summary embedding for issue ID {issue.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for issue ID {issue.id}: {e}")
            
            # Process discourse_posts summary embeddings
            posts_summaries = db.query(DiscoursePost).filter(DiscoursePost.summary_embedding == None, DiscoursePost.llm_summary != None).limit(self.config['batch_size']).all()
            for post in posts_summaries:
                try:
                    embedding = self.embedding_client.create_embedding(str(post.llm_summary))
                    if embedding:
                        db.execute(text("UPDATE discourse_posts SET summary_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
                        logger.info(f"Generated summary embedding for discourse_post ID {post.id}")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate summary embedding for discourse_post ID {post.id}: {e}")
            
            db.commit()
            logger.info(f"Summary embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_summary_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    def process_synonym_embeddings_batch(self, db: Session) -> int:
        """Process synonym embeddings only."""
        try:
            logger.info("Processing synonym embeddings...")
            
            total_processed = 0
            
            # Process synonyms without word embeddings
            synonyms_without_word_embedding = db.query(Synonym).filter(Synonym.word_embedding == None).limit(self.config['batch_size']).all()
            for synonym in synonyms_without_word_embedding:
                try:
                    embedding = self.embedding_client.create_embedding(str(synonym.word))
                    if embedding:
                        db.execute(text("UPDATE synonyms SET word_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(embedding), "synonym_id": synonym.id})
                        logger.info(f"Generated word embedding for synonym '{synonym.word}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate word embedding for synonym '{synonym.word}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate word embedding for synonym '{synonym.word}': {e}")
            
            # Process synonyms without synonym relationship embeddings
            synonyms_without_synonym_embedding = db.query(Synonym).filter(Synonym.synonym_embedding == None).limit(self.config['batch_size']).all()
            for synonym in synonyms_without_synonym_embedding:
                try:
                    synonym_text = f"word: {synonym.word}\nsynonym_of: {synonym.synonym_of}"
                    embedding = self.embedding_client.create_embedding(synonym_text)
                    if embedding:
                        db.execute(text("UPDATE synonyms SET synonym_embedding = :embedding WHERE id = :synonym_id"), {"embedding": str(embedding), "synonym_id": synonym.id})
                        logger.info(f"Generated synonym relationship embedding for '{synonym.word}' -> '{synonym.synonym_of}'")
                        total_processed += 1
                    else:
                        logger.error(f"Failed to generate synonym embedding for '{synonym.word}': API returned None")
                except Exception as e:
                    logger.error(f"Failed to generate synonym embedding for '{synonym.word}': {e}")
            
            db.commit()
            logger.info(f"Synonym embeddings batch completed: {total_processed} embeddings generated")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error in process_synonym_embeddings_batch: {e}")
            db.rollback()
            return 0
    
    async def process_llm_questions_batch(self, db: Session, source_type: str) -> int:
        """Process a batch of LLM questions."""
        try:
            logger.info(f"Processing LLM questions batch for {source_type}...")
            
            table_name = self.get_table_name(source_type)
            content_field = self.get_content_field(source_type)
            
            # Get documents that don't have questions yet
            result = db.execute(
                text(f"""
                    SELECT id, {content_field} as content FROM {table_name} 
                    WHERE id NOT IN (
                        SELECT DISTINCT source_id FROM questions WHERE source_type = :source_type
                    ) AND {content_field} IS NOT NULL
                    LIMIT :limit
                """),
                {"source_type": source_type, "limit": self.config['batch_size']}
            ).fetchall()
            
            if not result:
                logger.info(f"No {source_type} documents need LLM questions")
                return 0
            
            logger.info(f"Found {len(result)} {source_type} documents needing LLM questions")
            
            processed_count = 0
            
            for row in result:
                try:
                    prompt = self.get_prompt_for_source_type(source_type, row.content, db)
                    
                    logger.info(f"Sending {source_type} ID {row.id} to LLM...")
                    
                    # Call LLM
                    llm_response = await self.call_llm_with_retry(prompt)
                    
                    if llm_response:
                        logger.info("Received response from LLM")
                        
                        # Parse the JSON response
                        parsed_questions = self.parse_llm_questions_response(llm_response, row.id)
                        
                        if parsed_questions:
                            # Deduplicate by question text (case-insensitive) per source before insert
                            seen = set()
                            unique_qas = []
                            for qa in parsed_questions:
                                key = qa['question'].strip().lower()
                                if key and key not in seen:
                                    seen.add(key)
                                    unique_qas.append(qa)
                            # Insert questions into the database
                            for qa in unique_qas:
                                # Insert the question/answer pair
                                insert_result = db.execute(
                                    text("""
                                        INSERT INTO questions (source_type, source_id, question, answer, created_at, updated_at) 
                                        VALUES (:source_type, :source_id, :question, :answer, NOW(), NOW()) 
                                        RETURNING id
                                    """),
                                    {
                                        "source_type": source_type,
                                        "source_id": row.id,
                                        "question": qa['question'],
                                        "answer": qa['answer']
                                    }
                                ).fetchone()
                                
                                question_id = insert_result.id if insert_result else None
                                
                                # Create embeddings for question and answer
                                question_embedding = await self.create_embedding_with_retry(qa['question'])
                                answer_embedding = await self.create_embedding_with_retry(qa['answer'])
                                
                                if question_embedding and answer_embedding:
                                    # Update the question with embeddings
                                    db.execute(
                                        text("""
                                            UPDATE questions 
                                            SET question_embedding = :question_embedding, 
                                                answer_embedding = :answer_embedding, 
                                                updated_at = NOW() 
                                            WHERE id = :id
                                        """),
                                        {
                                            "question_embedding": question_embedding,
                                            "answer_embedding": answer_embedding,
                                            "id": question_id
                                        }
                                    )
                                    
                                    logger.info(f"Created question/answer with embeddings for {source_type} ID {row.id}")
                                else:
                                    logger.warning(f"Failed to create embeddings for question ID {question_id}")
                            
                            processed_count += 1
                        else:
                            logger.warning(f"No valid questions parsed for {source_type} ID {row.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing {source_type} ID {row.id}: {e}")
            
            db.commit()
            logger.info(f"Batch completed: {processed_count}/{len(result)} documents processed")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing LLM questions batch: {e}")
            db.rollback()
            return 0

async def process_embeddings(mode: str = ProcessingModes.LLM_QUESTIONS_METABASE):
    """Main function to process embeddings based on mode."""
    mode_names = {
        ProcessingModes.LLM_QUESTIONS_METABASE: 'LLM questions for Metabase docs',
        ProcessingModes.LLM_QUESTIONS_ISSUES: 'LLM questions for GitHub issues',
        ProcessingModes.LLM_QUESTIONS_DISCOURSE: 'LLM questions for Discourse posts',
        ProcessingModes.MARKDOWN_ONLY: 'markdown embeddings only',
        ProcessingModes.ALL_EMBEDDINGS: 'all NULL embeddings across all tables',
        ProcessingModes.KEYWORD_EMBEDDINGS: 'keyword embeddings only',
        ProcessingModes.DOCS_EMBEDDINGS: 'docs embeddings (markdown and summary)',
        ProcessingModes.ISSUES_EMBEDDINGS: 'issues embeddings (issue and summary)',
        ProcessingModes.POSTS_EMBEDDINGS: 'discourse posts embeddings (conversation and summary)',
        ProcessingModes.QUESTIONS_EMBEDDINGS: 'questions embeddings (question and answer)',
        ProcessingModes.SUMMARY_EMBEDDINGS: 'summary embeddings for all entities',
        ProcessingModes.SYNONYM_EMBEDDINGS: 'synonym embeddings (word and synonym relationship)'
    }
    
    logger.info(f"🔄 Processing {mode_names.get(mode, mode)}")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'api_key': os.getenv('API_KEY', 'your-api-key-here'),
        'litellm_url': os.getenv('LITELLM_URL', 'http://localhost:4000'),
        'model_name': LITELLM_MODEL_NAME,
        'requests_per_minute': int(os.getenv('REQUESTS_PER_MINUTE', '30' if 'llm-questions' in mode else '60')),
        'batch_size': int(os.getenv('BATCH_SIZE', '5' if 'llm-questions' in mode else '10')),
        'delay_between_batches': int(os.getenv('DELAY_BETWEEN_BATCHES', '3000' if 'llm-questions' in mode else '0')),
        'mode': mode
    }
    
    logger.info(f"API Key: {'Set' if config['api_key'] != 'your-api-key-here' else 'Not set'}")
    if 'llm-questions' in mode:
        logger.info(f"LiteLLM URL: {config['litellm_url']}")
        logger.info(f"Model: {config['model_name']}")
    logger.info(f"Rate Limit: {config['requests_per_minute']} requests/minute")
    logger.info(f"Batch Size: {config['batch_size']} documents per batch")
    logger.info(f"Batch Delay: {config['delay_between_batches']}ms between batches")
    logger.info(f"Processing Mode: {mode}")
    logger.info("=" * 60)
    
    # Validate required configuration (API key not needed for embedding-only modes)
    embedding_only_modes = [
        ProcessingModes.ALL_EMBEDDINGS, 
        ProcessingModes.KEYWORD_EMBEDDINGS,
        ProcessingModes.DOCS_EMBEDDINGS,
        ProcessingModes.ISSUES_EMBEDDINGS,
        ProcessingModes.POSTS_EMBEDDINGS,
        ProcessingModes.QUESTIONS_EMBEDDINGS,
        ProcessingModes.SUMMARY_EMBEDDINGS,
        ProcessingModes.SYNONYM_EMBEDDINGS
    ]
    if mode not in embedding_only_modes and (not config['api_key'] or config['api_key'] == 'your-api-key-here'):
        logger.error("❌ Error: API_KEY environment variable is required")
        logger.info("Please set the API_KEY environment variable with your API key")
        sys.exit(1)
    
    try:
        async with EmbeddingProcessor(config) as processor:
            db = SessionLocal()
            
            try:
                # Initialize source_type variable
                source_type = None
                
                # Count documents that need processing based on mode
                if mode == ProcessingModes.LLM_QUESTIONS_METABASE:
                    source_type = SourceTypes.METABASE_DOC
                    count_query = """
                        SELECT COUNT(*) FROM metabase_docs md 
                        WHERE md.id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                        ) AND md.markdown IS NOT NULL
                    """
                elif mode == ProcessingModes.LLM_QUESTIONS_ISSUES:
                    source_type = SourceTypes.ISSUE
                    count_query = """
                        SELECT COUNT(*) FROM issues i 
                        WHERE i.id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                        ) AND i.body IS NOT NULL
                    """
                elif mode == ProcessingModes.LLM_QUESTIONS_DISCOURSE:
                    source_type = SourceTypes.DISCOURSE_POST
                    count_query = """
                        SELECT COUNT(*) FROM discourse_posts dp 
                        WHERE dp.id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'DISCOURSE_POST'
                        ) AND dp.conversation IS NOT NULL
                    """
                elif mode == ProcessingModes.ALL_EMBEDDINGS:
                    # Count all NULL embeddings across all tables
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM metabase_docs WHERE markdown_embedding IS NULL AND markdown IS NOT NULL) +
                            (SELECT COUNT(*) FROM metabase_docs WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL) +
                            (SELECT COUNT(*) FROM issues WHERE issue_embedding IS NULL AND body IS NOT NULL AND title IS NOT NULL AND title != '') +
                            (SELECT COUNT(*) FROM issues WHERE title_embedding IS NULL AND title IS NOT NULL AND title != '') +
                            (SELECT COUNT(*) FROM issues WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL) +
                            (SELECT COUNT(*) FROM discourse_posts WHERE conversation_embedding IS NULL AND conversation IS NOT NULL) +
                            (SELECT COUNT(*) FROM discourse_posts WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL) +
                            (SELECT COUNT(*) FROM questions WHERE question_embedding IS NULL AND question IS NOT NULL) +
                            (SELECT COUNT(*) FROM questions WHERE answer_embedding IS NULL AND answer IS NOT NULL) +
                            (SELECT COUNT(*) FROM keyword_definitions WHERE keyword_embedding IS NULL) +
                            (SELECT COUNT(*) FROM synonyms WHERE word_embedding IS NULL) +
                            (SELECT COUNT(*) FROM synonyms WHERE synonym_embedding IS NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.KEYWORD_EMBEDDINGS:
                    # Count keyword embeddings that need processing
                    count_query = 'SELECT COUNT(*) FROM keyword_definitions WHERE keyword_embedding IS NULL'
                elif mode == ProcessingModes.DOCS_EMBEDDINGS:
                    # Count docs embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM metabase_docs WHERE markdown_embedding IS NULL AND markdown IS NOT NULL) +
                            (SELECT COUNT(*) FROM metabase_docs WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.ISSUES_EMBEDDINGS:
                    # Count issues embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM issues WHERE issue_embedding IS NULL AND body IS NOT NULL AND title IS NOT NULL AND title != '') +
                            (SELECT COUNT(*) FROM issues WHERE title_embedding IS NULL AND title IS NOT NULL AND title != '') +
                            (SELECT COUNT(*) FROM issues WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.POSTS_EMBEDDINGS:
                    # Count discourse posts embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM discourse_posts WHERE conversation_embedding IS NULL AND conversation IS NOT NULL) +
                            (SELECT COUNT(*) FROM discourse_posts WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.QUESTIONS_EMBEDDINGS:
                    # Count questions embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM questions WHERE question_embedding IS NULL AND question IS NOT NULL) +
                            (SELECT COUNT(*) FROM questions WHERE answer_embedding IS NULL AND answer IS NOT NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.SUMMARY_EMBEDDINGS:
                    # Count summary embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM metabase_docs WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL) +
                            (SELECT COUNT(*) FROM issues WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL) +
                            (SELECT COUNT(*) FROM discourse_posts WHERE summary_embedding IS NULL AND llm_summary IS NOT NULL)
                        ) AS total_count
                    """
                elif mode == ProcessingModes.SYNONYM_EMBEDDINGS:
                    # Count synonym embeddings that need processing
                    count_query = """
                        SELECT (
                            (SELECT COUNT(*) FROM synonyms WHERE word_embedding IS NULL) +
                            (SELECT COUNT(*) FROM synonyms WHERE synonym_embedding IS NULL)
                        ) AS total_count
                    """
                else:
                    count_query = 'SELECT COUNT(*) FROM metabase_docs WHERE markdown_embedding IS NULL AND markdown IS NOT NULL'
                
                count_result = db.execute(text(count_query)).fetchone()
                total_to_process = count_result[0] if count_result else 0
                
                if total_to_process == 0:
                    if 'llm-questions' in mode:
                        message = 'questions'
                    elif mode == ProcessingModes.ISSUES_EMBEDDINGS:
                        message = 'issue embeddings'
                    elif mode == ProcessingModes.POSTS_EMBEDDINGS:
                        message = 'discourse post embeddings'
                    elif mode == ProcessingModes.DOCS_EMBEDDINGS:
                        message = 'document embeddings'
                    elif mode == ProcessingModes.QUESTIONS_EMBEDDINGS:
                        message = 'question embeddings'
                    elif mode == ProcessingModes.SUMMARY_EMBEDDINGS:
                        message = 'summary embeddings'
                    elif mode == ProcessingModes.KEYWORD_EMBEDDINGS:
                        message = 'keyword embeddings'
                    elif mode == ProcessingModes.SYNONYM_EMBEDDINGS:
                        message = 'synonym embeddings'
                    elif mode == ProcessingModes.ALL_EMBEDDINGS:
                        message = 'embeddings'
                    else:
                        message = 'markdown embeddings'
                    logger.info(f"✅ All documents already have {message}!")
                    return
                
                logger.info(f"Found {total_to_process} documents needing processing")
                
                # Process in batches
                total_processed = 0
                batch_count = 0
                
                while True:
                    if 'llm-questions' in mode and source_type is not None:
                        processed_in_batch = await processor.process_llm_questions_batch(db, source_type)
                    elif mode == ProcessingModes.ALL_EMBEDDINGS:
                        processed_in_batch = processor.process_all_embeddings_batch(db)
                    elif mode == ProcessingModes.KEYWORD_EMBEDDINGS:
                        processed_in_batch = processor.process_keyword_embeddings_batch(db)
                    elif mode == ProcessingModes.DOCS_EMBEDDINGS:
                        processed_in_batch = processor.process_docs_embeddings_batch(db)
                    elif mode == ProcessingModes.ISSUES_EMBEDDINGS:
                        processed_in_batch = processor.process_issues_embeddings_batch(db)
                    elif mode == ProcessingModes.POSTS_EMBEDDINGS:
                        processed_in_batch = processor.process_posts_embeddings_batch(db)
                    elif mode == ProcessingModes.QUESTIONS_EMBEDDINGS:
                        processed_in_batch = processor.process_questions_embeddings_batch(db)
                    elif mode == ProcessingModes.SUMMARY_EMBEDDINGS:
                        processed_in_batch = processor.process_summary_embeddings_batch(db)
                    elif mode == ProcessingModes.SYNONYM_EMBEDDINGS:
                        processed_in_batch = processor.process_synonym_embeddings_batch(db)
                    else:
                        processed_in_batch = await processor.process_markdown_batch(db)
                    
                    if processed_in_batch == 0:
                        break  # No more documents need processing
                    
                    total_processed += processed_in_batch
                    batch_count += 1
                    
                    logger.info(f"✅ Batch {batch_count} completed. Total processed: {total_processed}/{total_to_process}")
                    
                    # Add delay between batches
                    if processed_in_batch > 0:
                        logger.info(f"⏳ Waiting {config['delay_between_batches']}ms before next batch...")
                        await asyncio.sleep(config['delay_between_batches'] / 1000)
                
                logger.info('\n🎉 Processing completed!')
                logger.info(f"📊 Summary:")
                logger.info(f"  - Documents processed: {total_processed}")
                logger.info(f"  - Batches processed: {batch_count}")
                
            finally:
                db.close()
        
    except Exception as e:
        logger.error('\n' + '=' * 60)
        logger.error('❌ EMBEDDING PROCESSING FAILED')
        logger.error('=' * 60)
        logger.error(f"Error: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process embeddings and LLM calls')
    parser.add_argument('mode', nargs='?', default=ProcessingModes.LLM_QUESTIONS_METABASE,
                       choices=[
                           ProcessingModes.LLM_QUESTIONS_METABASE,
                           ProcessingModes.LLM_QUESTIONS_ISSUES,
                           ProcessingModes.LLM_QUESTIONS_DISCOURSE,
                           ProcessingModes.MARKDOWN_ONLY,
                           ProcessingModes.ALL_EMBEDDINGS,
                           ProcessingModes.KEYWORD_EMBEDDINGS,
                           ProcessingModes.DOCS_EMBEDDINGS,
                           ProcessingModes.ISSUES_EMBEDDINGS,
                           ProcessingModes.POSTS_EMBEDDINGS,
                           ProcessingModes.QUESTIONS_EMBEDDINGS,
                           ProcessingModes.SUMMARY_EMBEDDINGS,
                           ProcessingModes.SYNONYM_EMBEDDINGS
                       ],
                       help='Processing mode')
    
    args = parser.parse_args()
    
    asyncio.run(process_embeddings(args.mode))

if __name__ == "__main__":
    main() 