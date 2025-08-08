#!/usr/bin/env python3
"""
LLM Processor CLI for GitHub Duplicate Issue Finder.
This script provides a command-line interface for LLM operations using the LLMAnalyzer.
"""

import os
import sys
import asyncio
import logging
import argparse
import signal
from typing import List, Dict, Optional, Any
import time

# Use the shared path setup utility
from path_setup import setup_project_path
setup_project_path()

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db import SessionLocal, engine
from src.models import Issue, DiscoursePost, MetabaseDoc
from src.settings import LITELLM_MODEL_NAME
from src.llm_analyzer import LLMAnalyzer
from src.constants import DEFAULT_BATCH_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store the database session for cleanup
current_db_session = None

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"Received signal {signum}, cleaning up...")
    if current_db_session:
        try:
            if current_db_session.is_active:
                current_db_session.rollback()
                logger.info("Rolled back database transaction due to interrupt")
            current_db_session.close()
            logger.info("Closed database session due to interrupt")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class LLMProcessor:
    """CLI wrapper for LLM operations using LLMAnalyzer."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.llm_analyzer = LLMAnalyzer()
        
    async def summarize_table(self, db: Session, table_name: str) -> int:
        """Summarize content in a specific table using LLMAnalyzer."""
        try:
            logger.info(f"Summarizing {table_name}...")
            
            fetch_batch_size = DEFAULT_BATCH_SIZE  # How many items to fetch from DB at once
            total_processed = 0
            
            # Get total count of unprocessed items
            if table_name == "issues":
                total_count = db.execute(
                    text("SELECT COUNT(*) FROM issues WHERE llm_summary IS NULL AND body IS NOT NULL")
                ).scalar()
            elif table_name == "discourse_posts":
                total_count = db.execute(
                    text("SELECT COUNT(*) FROM discourse_posts WHERE llm_summary IS NULL AND conversation IS NOT NULL")
                ).scalar()
            elif table_name == "metabase_docs":
                total_count = db.execute(
                    text("SELECT COUNT(*) FROM metabase_docs WHERE llm_summary IS NULL AND markdown IS NOT NULL")
                ).scalar()
            else:
                logger.error(f"Unknown table: {table_name}")
                return 0
            
            if total_count == 0:
                logger.info(f"No {table_name} need summarization")
                return 0
            
            logger.info(f"Found {total_count} {table_name} needing summarization")
            
            if not total_count or total_count == 0:
                logger.info(f"No {table_name} need summarization")
                return 0
            
            # Process in fetches to avoid loading too much into memory
            for offset in range(0, total_count, fetch_batch_size):
                # Get a batch of unprocessed items from database
                if table_name == "issues":
                    items_batch = db.query(Issue).filter(
                        Issue.llm_summary.is_(None),
                        Issue.body.is_not(None)
                    ).offset(offset).limit(fetch_batch_size).all()
                elif table_name == "discourse_posts":
                    items_batch = db.query(DiscoursePost).filter(
                        DiscoursePost.llm_summary.is_(None),
                        DiscoursePost.conversation.is_not(None)
                    ).offset(offset).limit(fetch_batch_size).all()
                elif table_name == "metabase_docs":
                    items_batch = db.query(MetabaseDoc).filter(
                        MetabaseDoc.llm_summary.is_(None),
                        MetabaseDoc.markdown.is_not(None)
                    ).offset(offset).limit(fetch_batch_size).all()
                
                if not items_batch:
                    break
                
                logger.info(f"Processing fetch batch {offset//fetch_batch_size + 1}: {len(items_batch)} items")
                
                # Create simple batches for LLM processing
                # Since batch processing won't exceed 1M tokens, we use the current fetch batch as is
                llm_batches = [items_batch]
                
                for batch_num, llm_batch in enumerate(llm_batches):
                    logger.info(f"  LLM batch {batch_num + 1}/{len(llm_batches)}: {len(llm_batch)} items")
                    
                    # Prepare batch data for single LLM call
                    batch_items = []
                    for item in llm_batch:
                        if table_name == "issues":
                            content = f"Title: {item.title}\nState: {item.state}\nLabels: {', '.join(item.labels)}\nBody: {item.body or ''}"
                        elif table_name == "discourse_posts":
                            content = item.conversation or ""
                        elif table_name == "metabase_docs":
                            content = item.markdown or ""
                        else:
                            content = str(item)
                        
                        batch_items.append((item.id, content))
                    
                    logger.info(f"Sending batch of {len(batch_items)} items to LLM...")
                    
                    # Send entire batch to LLM in single call
                    if table_name == "issues":
                        # For issues, use the existing batch analyzer which returns structured data
                        issues_for_llm = [(item.id, item.title, item.body or "", item.labels, item.state) for item in llm_batch]
                        batch_results = self.llm_analyzer.analyze_issues_batch(issues_for_llm)
                        summaries = {item_id: result.get("summary", "") for item_id, result in batch_results.items()}
                    else:
                        # For other content types, use the batch summarizer
                        summaries = self.llm_analyzer.summarize_batch(batch_items)
                    
                    # Process the batch results
                    batch_processed = 0
                    for item in llm_batch:
                        try:
                            summary = summaries.get(item.id, "")
                            
                            if summary:
                                logger.info(f"Received summary for {table_name} ID {item.id}")
                                
                                # Update the record with the summary
                                if table_name == "issues":
                                    db.execute(
                                        text("UPDATE issues SET llm_summary = :summary WHERE id = :id"),
                                        {"summary": summary, "id": item.id}
                                    )
                                elif table_name == "discourse_posts":
                                    db.execute(
                                        text("UPDATE discourse_posts SET llm_summary = :summary WHERE id = :id"),
                                        {"summary": summary, "id": item.id}
                                    )
                                elif table_name == "metabase_docs":
                                    db.execute(
                                        text("UPDATE metabase_docs SET llm_summary = :summary WHERE id = :id"),
                                        {"summary": summary, "id": item.id}
                                    )
                                
                                batch_processed += 1
                                total_processed += 1
                            else:
                                logger.warning(f"No summary returned for {table_name} ID {item.id}")
                            
                        except Exception as e:
                            logger.error(f"Error processing {table_name} ID {item.id}: {e}")
                    
                    db.commit()
                    logger.info(f"Batch {batch_num + 1} completed: {batch_processed}/{len(llm_batch)} items processed in single LLM call")
                    
            
            logger.info(f"Summarization completed: {total_processed} {table_name} processed")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error summarizing {table_name}: {e}")
            db.rollback()
            return 0
    
    async def create_questions_for_table(self, db: Session, table_name: str) -> int:
        """Create questions for content in a specific table using LLMAnalyzer."""
        try:
            logger.info(f"Creating questions for {table_name}...")
            
            if table_name == "metabase_docs":
                # Get metabase docs that don't have questions yet
                result = db.execute(
                    text("""
                        SELECT id, markdown FROM metabase_docs 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                        ) AND markdown IS NOT NULL
                        LIMIT :limit
                    """),
                    {"limit": self.config['batch_size']}
                ).fetchall()
                source_type = 'METABASE_DOC'
                content_field = 'markdown'
            elif table_name == "issues":
                # Get feature request issues that don't have questions yet
                result = db.execute(
                    text("""
                        SELECT id, body FROM issues 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                        ) AND body IS NOT NULL AND labels::text LIKE '%feature request%'
                        LIMIT :limit
                    """),
                    {"limit": self.config['batch_size']}
                ).fetchall()
                source_type = 'ISSUE'
                content_field = 'body'
            else:
                logger.error(f"Unknown table for questions: {table_name}")
                return 0
            
            if not result:
                logger.info(f"No {table_name} need questions")
                return 0
            
            logger.info(f"Found {len(result)} {table_name} needing questions")
            
            processed_count = 0
            
            for row in result:
                try:
                    content = getattr(row, content_field)
                    
                    logger.info(f"Sending {table_name} ID {row.id} to LLM for questions...")
                    
                    # Use LLMAnalyzer to create questions
                    parsed_questions = self.llm_analyzer.create_questions_for_content(table_name, content, row.id)
                    
                    if parsed_questions:
                        logger.info(f"Attempting to insert {len(parsed_questions)} questions for {table_name} ID {row.id}")
                        # Insert questions into the database
                        for i, qa in enumerate(parsed_questions):
                            try:
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
                                
                                if insert_result:
                                    question_id = insert_result.id
                                    logger.info(f"Created question/answer {i+1}/{len(parsed_questions)} for {table_name} ID {row.id} (ID: {question_id})")
                                else:
                                    logger.error(f"Failed to get insert ID for question {i+1} for {table_name} ID {row.id}")
                            except Exception as e:
                                logger.error(f"Failed to insert question {i+1} for {table_name} ID {row.id}: {e}")
                                raise  # Re-raise to trigger rollback
                        
                        processed_count += 1
                        logger.info(f"Successfully processed {table_name} ID {row.id} with {len(parsed_questions)} questions")
                        
                        # Commit immediately to ensure changes are visible
                        db.commit()
                        logger.info(f"Committed questions for {table_name} ID {row.id}")
                        
                    else:
                        logger.warning(f"No valid questions parsed for {table_name} ID {row.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing {table_name} ID {row.id}: {e}")
            
            logger.info(f"Questions creation completed: {processed_count}/{len(result)} {table_name} processed")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error creating questions for {table_name}: {e}")
            db.rollback()
            return 0

    async def create_questions_and_concepts_for_table(self, db: Session, table_name: str) -> int:
        """Create questions and extract concepts for content in a specific table using LLMAnalyzer."""
        try:
            logger.info(f"Creating questions and concepts for {table_name}...")
            
            if table_name == "metabase_docs":
                # Get metabase docs that don't have questions yet
                result = db.execute(
                    text("""
                        SELECT id, markdown FROM metabase_docs 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                        ) AND markdown IS NOT NULL
                        LIMIT :limit
                    """),
                    {"limit": self.config['batch_size']}
                ).fetchall()
                source_type = 'METABASE_DOC'
                content_field = 'markdown'
            elif table_name == "issues":
                # Get feature request issues that don't have questions yet
                result = db.execute(
                    text("""
                        SELECT id, body FROM issues 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                        ) AND body IS NOT NULL AND labels::text LIKE '%feature request%'
                        LIMIT :limit
                    """),
                    {"limit": self.config['batch_size']}
                ).fetchall()
                source_type = 'ISSUE'
                content_field = 'body'
            else:
                logger.error(f"Unknown table for questions and concepts: {table_name}")
                return 0
            
            if not result:
                logger.info(f"No {table_name} need questions and concepts")
                return 0
            
            logger.info(f"Found {len(result)} {table_name} needing questions and concepts")
            
            processed_count = 0
            
            for row in result:
                try:
                    content = getattr(row, content_field)
                    
                    logger.info(f"Sending {table_name} ID {row.id} to LLM for questions and concepts...")
                    
                    # Use LLMAnalyzer to create questions and concepts
                    parsed_data = self.llm_analyzer.create_questions_and_concepts_for_content(table_name, content, row.id)
                    
                    if parsed_data and (parsed_data.get('questions') or parsed_data.get('concepts')):
                        questions = parsed_data.get('questions', [])
                        concepts = parsed_data.get('concepts', [])
                        
                        logger.info(f"Attempting to insert {len(questions)} questions and {len(concepts)} concepts for {table_name} ID {row.id}")
                        
                        # Insert questions into the database
                        for i, qa in enumerate(questions):
                            try:
                                # Check if question already exists
                                existing_question = db.execute(
                                    text("""
                                        SELECT id, answer FROM questions 
                                        WHERE source_type = :source_type 
                                          AND source_id = :source_id 
                                          AND question = :question
                                    """),
                                    {
                                        "source_type": source_type,
                                        "source_id": row.id,
                                        "question": qa['question']
                                    }
                                ).fetchone()
                                
                                if not existing_question:
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
                                    
                                    if insert_result:
                                        question_id = insert_result.id
                                        logger.info(f"Created question/answer {i+1}/{len(questions)} for {table_name} ID {row.id} (ID: {question_id})")
                                    else:
                                        logger.error(f"Failed to get insert ID for question {i+1} for {table_name} ID {row.id}")
                                else:
                                    # Merge existing answer with new answer using LLM
                                    logger.info(f"Merging answers for existing question '{qa['question']}'")
                                    merged_answer = self.llm_analyzer.merge_question_answers(
                                        qa['question'],
                                        existing_question.answer,
                                        qa['answer']
                                    )
                                    
                                    if merged_answer:
                                        # Update the existing question with the merged answer
                                        db.execute(
                                            text("""
                                                UPDATE questions 
                                                SET answer = :answer, updated_at = NOW() 
                                                WHERE id = :id
                                            """),
                                            {
                                                "answer": merged_answer,
                                                "id": existing_question.id
                                            }
                                        )
                                        logger.info(f"Successfully merged answers for question '{qa['question']}'")
                                    else:
                                        logger.warning(f"Failed to merge answers for question '{qa['question']}', keeping existing")
                            except Exception as e:
                                logger.error(f"Failed to insert/update question {i+1} for {table_name} ID {row.id}: {e}")
                                raise  # Re-raise to trigger rollback
                        
                        # Commit questions immediately to ensure they are saved
                        db.commit()
                        logger.info(f"Committed questions for {table_name} ID {row.id}")
                        
                        # Now process concepts in a separate transaction
                        for i, concept in enumerate(concepts):
                            try:
                                # Check if concept already exists (exact match)
                                existing = db.execute(
                                    text("SELECT id, definition, category FROM keyword_definitions WHERE keyword = :keyword"),
                                    {"keyword": concept['concept']}
                                ).fetchone()
                                
                                if not existing:
                                    # Insert the concept/definition pair
                                    insert_result = db.execute(
                                        text("""
                                            INSERT INTO keyword_definitions (keyword, definition, category, is_active, created_at, updated_at) 
                                            VALUES (:keyword, :definition, :category, :is_active, NOW(), NOW()) 
                                            RETURNING id
                                        """),
                                        {
                                            "keyword": concept['concept'],
                                            "definition": concept['definition'],
                                            "category": "LLM_Extracted",
                                            "is_active": "true"
                                        }
                                    ).fetchone()
                                    
                                    if insert_result:
                                        concept_id = insert_result.id
                                        logger.info(f"Created concept/definition {i+1}/{len(concepts)} for {table_name} ID {row.id} (ID: {concept_id})")
                                    else:
                                        logger.error(f"Failed to get insert ID for concept {i+1} for {table_name} ID {row.id}")
                                else:
                                    # Handle existing concept based on category
                                    if existing.category == 'Glossary':
                                        # Don't modify glossary entries
                                        logger.info(f"Concept '{concept['concept']}' exists as Glossary entry, skipping")
                                    elif existing.category == 'LLM_Extracted':
                                        # Merge definitions using LLM
                                        logger.info(f"Merging definitions for existing LLM_Extracted concept '{concept['concept']}'")
                                        merged_definition = self.llm_analyzer.merge_concept_definitions(
                                            concept['concept'], 
                                            existing.definition, 
                                            concept['definition']
                                        )
                                        
                                        if merged_definition:
                                            db.execute(
                                                text("""
                                                    UPDATE keyword_definitions 
                                                    SET definition = :definition, updated_at = NOW() 
                                                    WHERE id = :id
                                                """),
                                                {
                                                    "definition": merged_definition,
                                                    "id": existing.id
                                                }
                                            )
                                            logger.info(f"Successfully merged definitions for concept '{concept['concept']}'")
                                        else:
                                            logger.warning(f"Failed to merge definitions for concept '{concept['concept']}', keeping existing")
                                    else:
                                        # For other categories, use the simple prepend approach
                                        updated_definition = f"Also means: {concept['concept']}. {existing.definition}"
                                        db.execute(
                                            text("""
                                                UPDATE keyword_definitions 
                                                SET definition = :definition, updated_at = NOW() 
                                                WHERE id = :id
                                            """),
                                            {
                                                "definition": updated_definition,
                                                "id": existing.id
                                            }
                                        )
                                        logger.info(f"Updated existing concept '{concept['concept']}' with additional definition")
                            except Exception as e:
                                logger.error(f"Failed to insert/update concept {i+1} for {table_name} ID {row.id}: {e}")
                                # Don't raise here, just log the error and continue with concepts
                                # But make sure the transaction is in a clean state
                                try:
                                    db.rollback()
                                except Exception as rollback_error:
                                    logger.error(f"Failed to rollback after concept error: {rollback_error}")
                                continue
                        
                        # Commit concepts separately
                        try:
                            db.commit()
                            logger.info(f"Committed concepts for {table_name} ID {row.id}")
                        except Exception as e:
                            logger.error(f"Failed to commit concepts for {table_name} ID {row.id}: {e}")
                            try:
                                db.rollback()
                                logger.info(f"Rolled back concepts for {table_name} ID {row.id}")
                            except Exception as rollback_error:
                                logger.error(f"Failed to rollback concepts for {table_name} ID {row.id}: {rollback_error}")
                        
                        processed_count += 1
                        logger.info(f"Successfully processed {table_name} ID {row.id} with {len(questions)} questions and {len(concepts)} concepts")
                        
                    else:
                        logger.warning(f"No valid questions or concepts parsed for {table_name} ID {row.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing {table_name} ID {row.id}: {e}")
            
            logger.info(f"Questions and concepts creation completed: {processed_count}/{len(result)} {table_name} processed")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error creating questions and concepts for {table_name}: {e}")
            db.rollback()
            return 0
    
    def delete_llm_summaries_from_table(self, db: Session, table_name: str) -> int:
        """Delete LLM summaries from a specific table."""
        try:
            logger.info(f"Deleting LLM summaries from {table_name}...")
            
            if table_name == "issues":
                result = db.execute(
                    text("UPDATE issues SET llm_summary = NULL")
                )
            elif table_name == "discourse_posts":
                result = db.execute(
                    text("UPDATE discourse_posts SET llm_summary = NULL")
                )
            elif table_name == "metabase_docs":
                result = db.execute(
                    text("UPDATE metabase_docs SET llm_summary = NULL")
                )
            else:
                logger.error(f"Unknown table: {table_name}")
                return 0
            
            db.commit()
            logger.info(f"LLM summaries deleted from {table_name}")
            return 1
            
        except Exception as e:
            logger.error(f"Error deleting LLM summaries from {table_name}: {e}")
            db.rollback()
            return 0

async def process_llm_operations(operation: str, target: str):
    """Main function to process LLM operations."""
    
    logger.info(f"🔄 Processing LLM operation: {operation} for {target}")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'api_key': os.getenv('LITELLM_API_KEY', 'your-litellm-proxy-api-key'),
        'litellm_url': os.getenv('LITELLM_URL', 'http://localhost:4000'),
        'model_name': LITELLM_MODEL_NAME,
        'requests_per_minute': int(os.getenv('REQUESTS_PER_MINUTE', '30')),
        'batch_size': int(os.getenv('BATCH_SIZE', '5')),
        'delay_between_batches': int(os.getenv('DELAY_BETWEEN_BATCHES', '3000')),
    }
    
    if operation == "delete":
        logger.info(f"API Key: Not required for delete operation")
        logger.info(f"Operation: {operation}")
        logger.info(f"Target: {target}")
        logger.info("=" * 60)
    else:
        logger.info(f"API Key: {'Set' if config['api_key'] != 'your-litellm-proxy-api-key' else 'Not set'}")
        logger.info(f"LiteLLM URL: {config['litellm_url']}")
        logger.info(f"Model: {config['model_name']}")
        logger.info(f"Rate Limit: {config['requests_per_minute']} requests/minute")
        logger.info(f"Batch Size: {config['batch_size']} documents per batch")
        logger.info(f"Operation: {operation}")
        logger.info(f"Target: {target}")
        logger.info("=" * 60)
    
    # Validate required configuration (only for operations that need LLM API)
    # For local LiteLLM, API key might not be required
    if operation != "delete" and (not config['api_key'] or config['api_key'] == 'your-litellm-proxy-api-key'):
        logger.warning("⚠️  Warning: LITELLM_API_KEY not set, attempting to proceed without authentication")
        logger.info("If you encounter authentication errors, please set the LITELLM_API_KEY environment variable")
        # Don't exit, let it try to proceed
    
    try:
        processor = LLMProcessor(config)
        db = SessionLocal()
        global current_db_session
        current_db_session = db
        
        try:
            if operation == "summarize":
                if target not in ["issues", "discourse_posts", "metabase_docs"]:
                    logger.error(f"❌ Invalid target for summarize: {target}")
                    logger.info("Valid targets: issues, discourse_posts, metabase_docs")
                    sys.exit(1)
                
                total_processed = 0
                batch_count = 0
                
                while True:
                    processed_in_batch = await processor.summarize_table(db, target)
                    
                    if processed_in_batch == 0:
                        break  # No more documents need processing
                    
                    total_processed += processed_in_batch
                    batch_count += 1
                    
                    logger.info(f"✅ Batch {batch_count} completed. Total processed: {total_processed}")
                
                logger.info('\n🎉 Summarization completed!')
                logger.info(f"📊 Summary: {total_processed} documents processed")
            
            elif operation == "create-questions":
                if target not in ["metabase_docs", "issues"]:
                    logger.error(f"❌ Invalid target for create-questions: {target}")
                    logger.info("Valid targets: metabase_docs, issues")
                    sys.exit(1)
                
                total_processed = 0
                batch_count = 0
                
                while True:
                    processed_in_batch = await processor.create_questions_for_table(db, target)
                    
                    if processed_in_batch == 0:
                        break  # No more documents need processing
                    
                    total_processed += processed_in_batch
                    batch_count += 1
                    
                    logger.info(f"✅ Batch {batch_count} completed. Total processed: {total_processed}")
                    
                    # Add delay between batches
                    if processed_in_batch > 0:
                        logger.info(f"⏳ Waiting {config['delay_between_batches']}ms before next batch...")
                        await asyncio.sleep(config['delay_between_batches'] / 1000)
                
                logger.info('\n🎉 Questions creation completed!')
                logger.info(f"📊 Summary: {total_processed} documents processed")
            
            elif operation == "create-questions-and-concepts":
                if target not in ["metabase_docs", "issues"]:
                    logger.error(f"❌ Invalid target for create-questions-and-concepts: {target}")
                    logger.info("Valid targets: metabase_docs, issues")
                    sys.exit(1)
                
                total_processed = 0
                batch_count = 0
                
                while True:
                    processed_in_batch = await processor.create_questions_and_concepts_for_table(db, target)
                    
                    if processed_in_batch == 0:
                        break  # No more documents need processing
                    
                    total_processed += processed_in_batch
                    batch_count += 1
                    
                    logger.info(f"✅ Batch {batch_count} completed. Total processed: {total_processed}")
                    
                    # Add delay between batches
                    if processed_in_batch > 0:
                        logger.info(f"⏳ Waiting {config['delay_between_batches']}ms before next batch...")
                        await asyncio.sleep(config['delay_between_batches'] / 1000)
                
                logger.info('\n🎉 Questions and concepts creation completed!')
                logger.info(f"📊 Summary: {total_processed} documents processed")
            
            elif operation == "delete":
                if target not in ["issues", "discourse_posts", "metabase_docs"]:
                    logger.error(f"❌ Invalid target for delete: {target}")
                    logger.info("Valid targets: issues, discourse_posts, metabase_docs")
                    sys.exit(1)
                
                processor.delete_llm_summaries_from_table(db, target)
                logger.info(f"✅ LLM summaries deleted from {target}")
            
            else:
                logger.error(f"❌ Unknown operation: {operation}")
                logger.info("Valid operations: summarize, create-questions, delete")
                sys.exit(1)
            
        finally:
            try:
                # Only commit if there are uncommitted changes
                if db.is_active:
                    db.commit()
                    logger.info("Final database commit completed")
            except Exception as e:
                logger.warning(f"Could not commit database changes: {e}")
                try:
                    db.rollback()
                    logger.info("Database rollback completed")
                except Exception as rollback_error:
                    logger.error(f"Could not rollback database changes: {rollback_error}")
            finally:
                try:
                    db.close()
                    logger.info("Database session closed")
                except Exception as close_error:
                    logger.error(f"Could not close database session: {close_error}")
                finally:
                    current_db_session = None  # Clear the global reference
        
    except Exception as e:
        logger.error('\n' + '=' * 60)
        logger.error('❌ LLM PROCESSING FAILED')
        logger.error('=' * 60)
        logger.error(f"Error: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process LLM operations')
    parser.add_argument('operation', choices=['summarize', 'create-questions', 'create-questions-and-concepts', 'delete'],
                       help='LLM operation to perform')
    parser.add_argument('target', help='Target for the operation')
    
    args = parser.parse_args()
    
    asyncio.run(process_llm_operations(args.operation, args.target))

if __name__ == "__main__":
    main() 