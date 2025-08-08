#!/usr/bin/env python3
"""
Batch processor for OpenAI batch operations.
This service handles creating, sending, and monitoring batch requests to OpenAI.
"""

import os
import json
import asyncio
import logging
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import httpx
from src.settings import HTTPX_TIMEOUT

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db import SessionLocal
from src.models import BatchProcess, Issue, DiscoursePost, MetabaseDoc
from src.settings import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_BATCH_MODEL, OPENAI_BATCH_ENTITIES_PER_BATCH
from src.llm_analyzer import LLMAnalyzer
from src.keyword_service import KeywordService
from src.utils import build_keyword_context
from src.prompts import (
    get_base_global_prompt, 
    get_github_issue_analyzer_prompt,
    get_discourse_summarizer_prompt,
    get_documentation_summarizer_prompt,
    get_questions_generator_prompt,
    get_questions_concepts_generator_prompt
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles OpenAI batch processing operations using LiteLLM proxy."""

    def __init__(self):
        self.api_base = OPENAI_API_BASE  # Use OpenAI directly for batch operations
        self.api_key = OPENAI_API_KEY  # Direct OpenAI API key
        self.model = OPENAI_BATCH_MODEL  # Use OpenAI model for batch
        self.batch_dir = Path(__file__).parent.parent / "batch"
        self.batch_dir.mkdir(exist_ok=True)
        
        # Create sent and received subdirectories
        self.sent_dir = self.batch_dir / "sent"
        self.received_dir = self.batch_dir / "received"
        self.sent_dir.mkdir(exist_ok=True)
        self.received_dir.mkdir(exist_ok=True)
        
        self.llm_analyzer = LLMAnalyzer()
        self.keyword_service = KeywordService()
        self.entities_per_batch = OPENAI_BATCH_ENTITIES_PER_BATCH  # Number of entities per batch request

    def _get_relevant_keywords_for_content(self, content: str, db: Session) -> List[Dict[str, str]]:
        """
        Get relevant keywords for the given content using the keyword service.
        
        Args:
            content: The content to analyze for relevant keywords
            db: Database session
            
        Returns:
            List of relevant keyword definitions
        """
        try:
            return self.keyword_service.get_relevant_keywords(content, db)
        except Exception as e:
            logger.error(f"Error getting relevant keywords: {e}")
            return []

    def _build_keyword_context(self, keywords: List[Dict[str, str]]) -> str:
        return build_keyword_context(keywords)

    def _create_entity_batches(self, items: List, table_name: str) -> List[List]:
        """
        Create batches of entities for efficient processing.
        
        Args:
            items: List of database entities
            table_name: Name of the table being processed
            
        Returns:
            List of batches, where each batch contains up to self.entities_per_batch entities
        """
        batches = []
        for i in range(0, len(items), self.entities_per_batch):
            batch = items[i:i + self.entities_per_batch]
            batches.append(batch)
        return batches

    def _format_entities_for_batch(self, entities: List, table_name: str) -> str:
        """
        Format a list of entities into a single string for batch processing.
        
        Args:
            entities: List of database entities
            table_name: Name of the table being processed
            
        Returns:
            Formatted string containing all entities
        """
        formatted_entities = []
        
        for entity in entities:
            if table_name == "issues":
                labels_value = getattr(entity, 'labels', None)
                if labels_value and isinstance(labels_value, list):
                    labels_str = ', '.join(labels_value)
                else:
                    labels_str = str(labels_value) if labels_value else ''
                
                # Limit body size to manage token usage
                body_content = (entity.body or '')[:2000]
                formatted_entity = f"ID: {entity.id}\nTitle: {entity.title}\nState: {entity.state}\nLabels: {labels_str}\nBody: {body_content}\n"
                
            elif table_name == "discourse_posts":
                conversation_content = (entity.conversation or '')[:2000]
                formatted_entity = f"ID: {entity.id}\nConversation: {conversation_content}\n"
                
            elif table_name == "metabase_docs":
                markdown_content = (entity.markdown or '')[:2000]
                formatted_entity = f"ID: {entity.id}\nContent: {markdown_content}\n"
            
            formatted_entities.append(formatted_entity)
        
        return "\n---\n".join(formatted_entities)

    def _get_batch_prompt(self, table_name: str, operation_type: str) -> str:
        """
        Get the appropriate prompt for batch processing.
        
        Args:
            table_name: Name of the table being processed
            operation_type: Type of operation (summarize, questions, etc.)
            
        Returns:
            Prompt string for the batch operation
        """
        if operation_type == "summarize":
            if table_name == "issues":
                return """TASK: Analyze the following GitHub issues and provide a JSON response for each issue.

For each issue, extract:
1. A concise summary focusing on the core problem and key details
2. The reported version if mentioned (look for version numbers that match the pattern xx.x, e.g., 55.5 or 46.1)
3. Stack trace filename if mentioned (look for file paths, filenames with extensions like .clj, .js or .jsx that are relevant to Metabase source code)

Return your response as a JSON array where each element corresponds to an issue in the same order:
[
  {
    "id": "issue_id",
    "summary": "Your concise summary here",
    "reported_version": "version string or null",
    "stack_trace_file": "filename or null"
  },
  ...
]"""
            elif table_name == "discourse_posts":
                return """TASK: Summarize the following Discourse conversations.

For each conversation, provide a concise summary that captures the main points, questions, and conclusions.

Return your response as a JSON array where each element corresponds to a conversation in the same order:
[
  {
    "id": "conversation_id",
    "summary": "Your concise summary here"
  },
  ...
]"""
            elif table_name == "metabase_docs":
                return """TASK: Summarize the following Metabase documentation.

For each document, provide a concise summary that captures the key concepts, features, and important information.

Return your response as a JSON array where each element corresponds to a document in the same order:
[
  {
    "id": "document_id",
    "summary": "Your concise summary here"
  },
  ...
]"""
        
        elif operation_type == "questions":
            if table_name == "issues":
                return """TASK: Generate questions from the following GitHub issues (focus on feature requests).

For each issue, generate 2-3 relevant questions that could help understand the feature request better.

Return your response as a JSON array where each element corresponds to an issue in the same order:
[
  {
    "id": "issue_id",
    "questions": [
      "Question 1",
      "Question 2",
      "Question 3"
    ]
  },
  ...
]"""
            elif table_name == "metabase_docs":
                return """TASK: Generate questions from the following Metabase documentation.

For each document, generate 2-3 relevant questions that could help understand the content better.

Return your response as a JSON array where each element corresponds to a document in the same order:
[
  {
    "id": "document_id",
    "questions": [
      "Question 1",
      "Question 2",
      "Question 3"
    ]
  },
  ...
]"""
        
        elif operation_type == "questions_and_concepts":
            if table_name == "issues":
                return """TASK: Generate questions and extract key concepts from the following GitHub issues (focus on feature requests).

For each issue, generate:
1. 2-3 relevant questions
2. Key concepts mentioned in the issue

Return your response as a JSON array where each element corresponds to an issue in the same order:
[
  {
    "id": "issue_id",
    "questions": [
      "Question 1",
      "Question 2",
      "Question 3"
    ],
    "concepts": [
      "Concept 1",
      "Concept 2",
      "Concept 3"
    ]
  },
  ...
]"""
            elif table_name == "metabase_docs":
                return """TASK: Generate questions and extract key concepts from the following Metabase documentation.

For each document, generate:
1. 2-3 relevant questions
2. Key concepts mentioned in the document

Return your response as a JSON array where each element corresponds to a document in the same order:
[
  {
    "id": "document_id",
    "questions": [
      "Question 1",
      "Question 2",
      "Question 3"
    ],
    "concepts": [
      "Concept 1",
      "Concept 2",
      "Concept 3"
    ]
  },
  ...
]"""
        
        raise ValueError(f"Unknown operation type: {operation_type}")

    async def create_efficient_batch_file(self, db: Session, table_name: str, operation_type: str) -> Tuple[Optional[str], int]:
        """
        Create an efficient batch file that groups multiple entities per request.
        
        Args:
            db: Database session
            table_name: Name of the table to process
            operation_type: Type of operation (summarize, questions, questions_and_concepts)
            
        Returns:
            Tuple of (file_path, total_requests)
        """
        logger.info(f"Creating efficient batch file for {operation_type} {table_name}...")
        
        # Get items that need processing
        if table_name == "issues":
            if operation_type == "summarize":
                items = db.query(Issue).filter(
                    Issue.llm_summary.is_(None),
                    Issue.body.is_not(None)
                ).limit(50000).all()
            else:  # questions or questions_and_concepts
                result = db.execute(
                    text("""
                        SELECT id, title, body, state, labels FROM issues 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                        ) AND body IS NOT NULL AND labels::text LIKE '%feature request%'
                        LIMIT 50000
                    """)
                ).fetchall()
                # Convert to Issue objects for consistency
                items = [Issue(id=r.id, title=r.title, body=r.body, state=r.state, labels=r.labels) for r in result]
                
        elif table_name == "discourse_posts":
            if operation_type == "summarize":
                items = db.query(DiscoursePost).filter(
                    DiscoursePost.llm_summary.is_(None),
                    DiscoursePost.conversation.is_not(None)
                ).limit(50000).all()
            else:
                result = db.execute(
                    text("""
                        SELECT id, conversation FROM discourse_posts 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'DISCOURSE_POST'
                        ) AND conversation IS NOT NULL
                        LIMIT 50000
                    """)
                ).fetchall()
                items = [DiscoursePost(id=r.id, conversation=r.conversation) for r in result]
                
        elif table_name == "metabase_docs":
            if operation_type == "summarize":
                items = db.query(MetabaseDoc).filter(
                    MetabaseDoc.llm_summary.is_(None),
                    MetabaseDoc.markdown.is_not(None)
                ).limit(50000).all()
            else:
                result = db.execute(
                    text("""
                        SELECT id, markdown FROM metabase_docs 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                        ) AND markdown IS NOT NULL
                        LIMIT 50000
                    """)
                ).fetchall()
                items = [MetabaseDoc(id=r.id, markdown=r.markdown) for r in result]
        else:
            raise ValueError(f"Unknown table: {table_name}")

        if not items:
            logger.info(f"No {table_name} need {operation_type}")
            return None, 0

        logger.info(f"Found {len(items)} {table_name} needing {operation_type}")

        # Create batches of entities
        entity_batches = self._create_entity_batches(items, table_name)
        logger.info(f"Created {len(entity_batches)} batches of up to {self.entities_per_batch} entities each")

        # Create batch file
        batch_id = str(uuid.uuid4())
        file_path = self.sent_dir / f"efficient_{operation_type}_{table_name}_{batch_id}.jsonl"
        
        # Get base global prompt
        base_global_prompt = get_base_global_prompt()
        
        requests = []
        for batch_num, entity_batch in enumerate(entity_batches):
            # Format all entities in this batch into a single string
            formatted_entities = self._format_entities_for_batch(entity_batch, table_name)
            
            # Get relevant keywords for the entire batch content
            relevant_keywords = self._get_relevant_keywords_for_content(formatted_entities, db)
            keyword_context = self._build_keyword_context(relevant_keywords)
            
            # Combine context and task prompt
            system_prompt = base_global_prompt
            if keyword_context:
                system_prompt += keyword_context
            system_prompt += "\n\n" + self._get_batch_prompt(table_name, operation_type)
            
            # Create entity IDs string for tracking
            entity_ids = [str(entity.id) for entity in entity_batch]
            entity_ids_str = ",".join(entity_ids)
            
            request = {
                "custom_id": f"efficient_{operation_type}_{table_name}_batch_{batch_num}_{entity_ids_str}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please process the following {len(entity_batch)} {table_name}:\n\n{formatted_entities}"}
                    ],
                    "max_tokens": 2000 if operation_type == "summarize" else 3000,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            }
            
            requests.append(request)

        # Write batch file
        with open(file_path, 'w') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')

        logger.info(f"Created efficient batch file {file_path} with {len(requests)} requests")
        return str(file_path), len(requests)



    async def submit_batch(self, file_path: str, operation_type: str, table_name: str, total_requests: int) -> str:
        """Submit a batch file to OpenAI."""
        logger.info(f"Submitting batch file {file_path} to OpenAI...")
        
        try:
            # First, upload the file
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f, 'application/jsonl')}
                    data = {'purpose': 'batch'}
                    
                    logger.info(f"Uploading file {file_path} ({os.path.getsize(file_path)} bytes)...")
                    upload_response = await client.post(
                        f"{self.api_base}/v1/files",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data=data
                    )
                    
                    logger.info(f"Upload response status: {upload_response.status_code}")
                    if upload_response.status_code != 200:
                        error_msg = f"Failed to upload file: {upload_response.status_code} - {upload_response.text}"
                        logger.error(error_msg)
                        logger.error(f"Response headers: {upload_response.headers}")
                        raise Exception(error_msg)
                    
                    upload_result = upload_response.json()
                    file_id = upload_result['id']
                    logger.info(f"File uploaded successfully: {file_id}")

                # Create the batch
                batch_data = {
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h"
                }
                
                logger.info(f"Creating batch with data: {batch_data}")
                batch_response = await client.post(
                    f"{self.api_base}/v1/batches",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=batch_data
                )
                
                logger.info(f"Batch creation response status: {batch_response.status_code}")
                if batch_response.status_code != 200:
                    error_msg = f"Failed to create batch: {batch_response.status_code} - {batch_response.text}"
                    logger.error(error_msg)
                    logger.error(f"Response headers: {batch_response.headers}")
                    raise Exception(error_msg)
                
                batch_result = batch_response.json()
                batch_id = batch_result['id']
                logger.info(f"Batch created successfully: {batch_id}")

        except httpx.TimeoutException as e:
            error_msg = f"Timeout during batch submission: {e}"
            logger.error(error_msg)
            raise TimeoutError(error_msg) from e
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error during batch submission: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Network error during batch submission: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Invalid response format during batch submission: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during batch submission: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Record in database
        db = SessionLocal()
        try:
            batch_process = BatchProcess(
                batch_id=batch_id,
                provider="openai",
                operation_type=operation_type,
                table_name=table_name,
                total_requests=total_requests,
                sent_at=datetime.now(timezone.utc),
                status="sent",
                input_file_path=file_path
            )
            db.add(batch_process)
            db.commit()
            logger.info(f"Batch process recorded in database with ID: {batch_process.id}")
        finally:
            db.close()

        return batch_id

    async def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check the status of a batch."""
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.get(
                f"{self.api_base}/v1/batches/{batch_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to check batch status: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            return response.json()

    async def download_batch_results(self, batch_id: str, output_file_id: str) -> str:
        """Download batch results."""
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.get(
                f"{self.api_base}/v1/files/{output_file_id}/content",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to download results: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Save to file
            output_path = self.received_dir / f"results_{batch_id}.jsonl"
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            logger.info(f"Results downloaded to {output_path}")
            return str(output_path)

    async def process_batch_results(self, batch_id: str, output_file_path: str, operation_type: str, table_name: str):
        """Process batch results and update the database."""
        logger.info(f"Processing batch results for {batch_id}...")
        
        # All batches now use efficient processing
        return await self.process_efficient_batch_results(batch_id, output_file_path, operation_type, table_name)

    async def process_efficient_batch_results(self, batch_id: str, output_file_path: str, operation_type: str, table_name: str):
        """
        Process efficient batch results and update the database.
        
        Args:
            batch_id: OpenAI batch ID
            output_file_path: Path to the output file
            operation_type: Type of operation performed
            table_name: Name of the table processed
        """
        logger.info(f"Processing efficient batch results for {batch_id}...")
        
        db = SessionLocal()
        try:
            with open(output_file_path, 'r') as f:
                results = [json.loads(line) for line in f]
            
            processed_count = 0
            error_count = 0
            
            for result in results:
                try:
                    custom_id = result['custom_id']
                    
                    if result.get('error'):
                        logger.error(f"Error in result for {custom_id}: {result['error']}")
                        error_count += 1
                        continue
                    
                    # Safely extract response content
                    response_obj = (
                        result.get('response', {})
                              .get('body', {})
                              .get('choices', [{}])[0]
                              .get('message', {})
                    )
                    response_content = response_obj.get('content')

                    # Validate presence and type of content
                    if not isinstance(response_content, str):
                        logger.error(f"No textual content for {custom_id}: {type(response_content).__name__}")
                        error_count += 1
                        continue

                    # Clean up the response content - remove extra whitespace and normalize
                    response_content = response_content.strip()
                    
                    # Check if response is empty or too short
                    if not response_content or len(response_content) < 10:
                        logger.error(f"Response content too short for {custom_id}: {response_content}")
                        error_count += 1
                        continue
                    
                    # Check if response appears to be truncated (ends abruptly)
                    if response_content.endswith('...') or response_content.endswith('"') or response_content.endswith(','):
                        logger.warning(f"Response appears to be truncated for {custom_id}")
                    
                    # Parse the JSON response
                    try:
                        batch_results = json.loads(response_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response for {custom_id}: {e}")
                        logger.error(f"Response content: {response_content}")
                        
                        # Try to extract partial results from malformed JSON
                        try:
                            # Look for JSON objects in the response
                            # More robust pattern that handles nested objects better
                            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                            matches = re.findall(json_pattern, response_content)
                            
                            if matches:
                                # Try to parse each match as JSON
                                parsed_results = []
                                for match in matches:
                                    try:
                                        parsed = json.loads(match)
                                        if isinstance(parsed, dict) and 'id' in parsed:
                                            parsed_results.append(parsed)
                                    except json.JSONDecodeError:
                                        continue
                                
                                if parsed_results:
                                    logger.info(f"Extracted {len(parsed_results)} partial results from malformed JSON for {custom_id}")
                                    batch_results = parsed_results
                                else:
                                    error_count += 1
                                    continue
                            else:
                                # Try alternative approach - look for individual JSON objects
                                # Split by common delimiters and try to parse each part
                                parts = re.split(r'[,\s]+', response_content)
                                parsed_results = []
                                for part in parts:
                                    part = part.strip()
                                    if part.startswith('{') and part.endswith('}'):
                                        try:
                                            parsed = json.loads(part)
                                            if isinstance(parsed, dict) and 'id' in parsed:
                                                parsed_results.append(parsed)
                                        except json.JSONDecodeError:
                                            continue
                                
                                if parsed_results:
                                    logger.info(f"Extracted {len(parsed_results)} results using alternative parsing for {custom_id}")
                                    batch_results = parsed_results
                                else:
                                    error_count += 1
                                    continue
                        except Exception as extract_error:
                            logger.error(f"Failed to extract partial results: {extract_error}")
                            error_count += 1
                            continue
                    
                    # Ensure batch_results is a list for processing
                    if isinstance(batch_results, dict):
                        # Extract from common container keys produced by models
                        for key in ['results', 'issues', 'documents', 'conversations', 'items', 'entries']:
                            if key in batch_results and isinstance(batch_results[key], list):
                                batch_results = batch_results[key]
                                break
                        else:
                            # If it's a single result dict, wrap it in a list
                            batch_results = [batch_results]
                    elif not isinstance(batch_results, list):
                        logger.error(f"Unexpected response format for {custom_id}: {type(batch_results)}")
                        error_count += 1
                        continue
                    
                    # Extract entity IDs from custom_id
                    # Format: efficient_operation_table_batch_num_id1,id2,id3
                    parts = custom_id.split('_')
                    if len(parts) < 5:
                        logger.error(f"Invalid custom_id format: {custom_id}")
                        error_count += 1
                        continue
                    
                    entity_ids_str = parts[-1]  # Last part contains comma-separated IDs
                    entity_ids = [int(id_str) for id_str in entity_ids_str.split(',')]
                    
                    # Process each entity result
                    for entity_result in batch_results:
                        # Ensure entity_result is a dictionary
                        if not isinstance(entity_result, dict):
                            logger.error(f"Entity result is not a dictionary for {custom_id}: {entity_result}")
                            error_count += 1
                            continue
                            
                        entity_id = entity_result.get('id')
                        if not entity_id:
                            logger.error(f"Missing entity ID in result: {entity_result}")
                            error_count += 1
                            continue
                        
                        # Convert entity_id to int if it's a string
                        if isinstance(entity_id, str):
                            try:
                                entity_id = int(entity_id)
                            except ValueError:
                                logger.error(f"Invalid entity ID format: {entity_id}")
                                error_count += 1
                                continue
                        
                        # Validate that this entity_id was in our original batch
                        if entity_id not in entity_ids:
                            logger.error(f"Entity ID {entity_id} not in original batch {entity_ids}")
                            error_count += 1
                            continue
                        
                        # Update database based on operation type
                        if operation_type == "summarize":
                            summary = entity_result.get('summary', '')
                            
                            if table_name == "issues":
                                reported_version = entity_result.get('reported_version')
                                stack_trace_file = entity_result.get('stack_trace_file')
                                
                                db.execute(
                                    text("""UPDATE issues 
                                         SET llm_summary = :summary, 
                                             reported_version = :reported_version, 
                                             stack_trace_file = :stack_trace_file 
                                         WHERE id = :id"""),
                                    {
                                        "summary": summary, 
                                        "reported_version": reported_version,
                                        "stack_trace_file": stack_trace_file,
                                        "id": entity_id
                                    }
                                )
                            elif table_name == "discourse_posts":
                                db.execute(
                                    text("UPDATE discourse_posts SET llm_summary = :summary WHERE id = :id"),
                                    {"summary": summary, "id": entity_id}
                                )
                            elif table_name == "metabase_docs":
                                db.execute(
                                    text("UPDATE metabase_docs SET llm_summary = :summary WHERE id = :id"),
                                    {"summary": summary, "id": entity_id}
                                )
                        
                        elif operation_type in ["questions", "questions_and_concepts"]:
                            questions = entity_result.get('questions', [])
                            
                            # For questions_and_concepts, we don't have answers, so skip inserting into questions table
                            if operation_type == "questions":
                                # Insert questions into the questions table (only for questions operation that has answers)
                                for question in questions:
                                    if isinstance(question, str) and question.strip():  # Skip empty or non-string questions
                                        source_type = self._get_source_type(table_name)
                                        logger.debug(f"Inserting question with source_type: {source_type}, table_name: {table_name}")
                                        db.execute(
                                            text("""INSERT INTO questions (question, source_id, source_type, created_at) 
                                                   VALUES (:question, :source_id, :source_type, NOW())"""),
                                            {
                                                "question": question,
                                                "source_id": entity_id,
                                                "source_type": source_type
                                            }
                                        )
                            
                            # If processing concepts as well
                            if operation_type == "questions_and_concepts":
                                concepts = entity_result.get('concepts', [])
                                
                                # Insert concepts into the keyword_definitions table
                                for concept in concepts:
                                    if concept.strip():  # Skip empty concepts
                                        db.execute(
                                            text("""INSERT INTO keyword_definitions (keyword, definition, category, created_at) 
                                                   VALUES (:keyword, :definition, :category, NOW())"""),
                                            {
                                                "keyword": concept,
                                                "definition": f"Concept extracted from {table_name} {entity_id}",
                                                "category": "Extracted Concepts"
                                            }
                                        )
                        
                        processed_count += 1
                        logger.info(f"Processed {operation_type} for {table_name} {entity_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing result for {custom_id}: {e}")
                    error_count += 1
                    # Rollback the transaction for this result and continue
                    db.rollback()
                    continue
            
            # Only commit if we have processed successfully
            if processed_count > 0:
                db.commit()
            logger.info(f"Efficient batch processing completed. Processed: {processed_count}, Errors: {error_count}")
            
            # Delete batch files from OpenAI after successful processing
            try:
                await self.delete_batch_files(batch_id)
            except Exception as e:
                logger.warning(f"Failed to delete batch files for {batch_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing efficient batch results: {e}")
            db.rollback()
        finally:
            db.close()

    def _get_source_type(self, table_name: str) -> str:
        """Get the source type string for the given table name."""
        mapping = {
            "issues": "ISSUE",
            "discourse_posts": "DISCOURSE_POST", 
            "metabase_docs": "METABASE_DOC"
        }
        return mapping.get(table_name, table_name.upper())

    async def create_and_submit_efficient_batch(self, operation_type: str, table_name: str) -> Optional[str]:
        """
        Create and submit an efficient batch for processing.
        
        Args:
            operation_type: Type of operation (summarize, questions, questions_and_concepts)
            table_name: Name of the table to process
            
        Returns:
            Batch ID if successful, None otherwise
        """
        try:
            # Create efficient batch file
            file_path, total_requests = await self.create_efficient_batch_file(
                SessionLocal(), table_name, operation_type
            )
            
            if not file_path:
                logger.info(f"No {table_name} need {operation_type}")
                return None
            
            # Submit batch
            batch_id = await self.submit_batch(file_path, f"efficient_{operation_type}", table_name, total_requests)
            
            logger.info(f"Submitted efficient batch {batch_id} for {operation_type} {table_name}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error creating and submitting efficient batch: {e}")
            return None

    async def create_and_submit_batch(self, operation_type: str, table_name: str) -> Optional[str]:
        """Create and submit a batch for the given operation and table using efficient batching."""
        # Map old operation types to new efficient ones
        operation_mapping = {
            "summarize": "summarize",
            "create-questions": "questions", 
            "create-questions-and-concepts": "questions_and_concepts"
        }
        
        efficient_operation = operation_mapping.get(operation_type)
        if not efficient_operation:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        return await self.create_and_submit_efficient_batch(efficient_operation, table_name)

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file from OpenAI."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base}/v1/files/{file_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code in (200, 204):
                    logger.info(f"Successfully deleted file {file_id} from OpenAI")
                    return True
                if response.status_code == 404:
                    # Consider 404 as already-deleted: treat as success to avoid noisy warnings
                    logger.info(f"File {file_id} not found at provider (already deleted). Treating as success.")
                    return True
                logger.warning(f"Failed to delete file {file_id}: {response.status_code} - {response.text}")
                return False
                    
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False

    async def delete_batch_files(self, batch_id: str) -> bool:
        """Delete both input and output files for a batch from OpenAI."""
        try:
            # Get batch status to find file IDs
            batch_status = await self.check_batch_status(batch_id)
            
            input_file_id = batch_status.get('input_file_id')
            output_file_id = batch_status.get('output_file_id')
            
            deleted_files = []
            
            # Delete input file
            if input_file_id:
                if await self.delete_file(input_file_id):
                    deleted_files.append(f"input file {input_file_id}")
            
            # Delete output file
            if output_file_id:
                if await self.delete_file(output_file_id):
                    deleted_files.append(f"output file {output_file_id}")
            
            if deleted_files:
                logger.info(f"Deleted {len(deleted_files)} files for batch {batch_id}: {', '.join(deleted_files)}")
                return True
            else:
                logger.warning(f"No files found to delete for batch {batch_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting files for batch {batch_id}: {e}")
            return False

    async def cleanup_completed_batch_processes(self) -> int:
        """Clean up files for all completed batch processes in the database."""
        db = SessionLocal()
        try:
            # Get all completed batch processes
            completed_batches = db.query(BatchProcess).filter(
                BatchProcess.status.in_(['completed', 'failed', 'expired', 'cancelled'])
            ).all()
            
            logger.info(f"Found {len(completed_batches)} completed batch processes in database")
            
            deleted_count = 0
            for batch_process in completed_batches:
                try:
                    if await self.delete_batch_files(batch_process.batch_id):
                        logger.info(f"Deleted files for batch {batch_process.batch_id}")
                        deleted_count += 1
                    else:
                        logger.warning(f"Failed to delete files for batch {batch_process.batch_id}")
                except Exception as e:
                    logger.error(f"Error deleting files for batch {batch_process.batch_id}: {e}")
            
            logger.info(f"Successfully deleted files for {deleted_count} out of {len(completed_batches)} completed batches")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up completed batch processes: {e}")
            return 0
        finally:
            db.close()
