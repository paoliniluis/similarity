import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError
from json_repair import repair_json
from . import settings
from .llm_client import llm_client
from .keyword_service import KeywordService
from .constants import MAX_RETRIES
from .prompts import (
    get_base_global_prompt,
    get_concept_definitions_merger_prompt,
    get_question_answers_merger_prompt,
    get_batch_issues_analyzer_prompt,
    get_single_issue_analyzer_prompt,
    get_batch_content_summarizer_prompt,
    get_discourse_conversation_analyzer_prompt,
    get_discourse_conversation_user_prompt
)
from .utils import build_keyword_context, clean_llm_json_response

# Configure logging
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Handles all interactions with the LLM via LiteLLM."""

    def __init__(self):
        """Initializes the LLMAnalyzer."""
        self.keyword_service = KeywordService()

    def _get_relevant_keywords_for_content(self, content: str, db=None) -> List[Dict[str, str]]:
        """
        Get relevant keywords for the given content using the keyword service.
        
        Args:
            content: The content to analyze for relevant keywords
            db: Database session (optional, will create one if not provided)
            
        Returns:
            List of relevant keyword definitions
        """
        try:
            if db is None:
                from .db import SessionLocal
                db = SessionLocal()
                try:
                    return self.keyword_service.get_relevant_keywords(content, db)
                finally:
                    db.close()
            else:
                return self.keyword_service.get_relevant_keywords(content, db)
        except Exception as e:
            logger.error(f"Error getting relevant keywords: {e}")
            return []

    def _build_keyword_context(self, keywords: List[Dict[str, str]]) -> str:
        return build_keyword_context(keywords)

    def _enhance_prompt_with_context(self, base_prompt: str, content: str, db=None) -> str:
        """
        Enhance a prompt with base global context and relevant keywords.
        
        Args:
            base_prompt: The base system prompt
            content: The content to analyze for relevant keywords
            db: Database session (optional)
            
        Returns:
            Enhanced prompt with global context and keywords
        """
        # Get base global context (background knowledge about Metabase)
        base_global_context = get_base_global_prompt()
        
        # Get relevant keywords for this content
        relevant_keywords = self._get_relevant_keywords_for_content(content, db)
        keyword_context = self._build_keyword_context(relevant_keywords)
        
        # Check if the global context is already present to avoid duplication
        # Look for key phrases that would indicate the context is already included
        if "CONTEXT: You are working with Metabase" in base_prompt or "ABOUT METABASE:" in base_prompt:
            logger.debug("Global Metabase context already present in base_prompt, skipping injection")
            enhanced_prompt = base_prompt
        else:
            # Combine: global context + keywords + specific task prompt
            enhanced_prompt = base_global_context
            if keyword_context:
                enhanced_prompt += keyword_context
            enhanced_prompt += "\n\n" + base_prompt
        
        return enhanced_prompt



    def _call_llm_with_retry(self, messages: List[Dict[str, str]], max_retries: int = MAX_RETRIES, 
                            response_format: Optional[Dict] = None, model: str = "openai-fast") -> Optional[str]:
        """Call LLM with retry logic and consistent error handling."""
        return llm_client.call_llm(
            messages=messages,
            model=model,
            max_retries=max_retries,
            response_format=response_format
        )

    def clean_json_response(self, response: str) -> str:
        """Clean up common issues in LLM responses."""
        # Remove leading/trailing whitespace and common prefixes
        return clean_llm_json_response(response)

    def parse_llm_questions_response(self, response: str, doc_id: int) -> List[Dict[str, str]]:
        """Parse LLM response to extract questions and answers using json-repair."""
        class QAItem(BaseModel):
            question: str = Field(min_length=1)
            answer: str = Field(min_length=1)
        class QASchema(BaseModel):
            questions: List[QAItem]
        try:
            # Clean up the response first
            cleaned_response = self.clean_json_response(response)
            
            # Use json-repair to automatically fix and parse the JSON
            try:
                repaired_json = repair_json(cleaned_response)
                parsed_data = json.loads(repaired_json)
                
                # Extract questions from parsed data
                try:
                    validated = QASchema.model_validate(parsed_data)
                    questions = [
                        {'question': qa.question.strip(), 'answer': qa.answer.strip()}
                        for qa in validated.questions
                    ]
                    if questions:
                        logger.info(f"Successfully parsed {len(questions)} questions for document ID {doc_id}")
                        return questions
                except ValidationError as ve:
                    logger.warning(f"Schema validation failed for doc {doc_id}: {ve}")
                    if parsed_data and parsed_data.get('questions') and isinstance(parsed_data['questions'], list):
                        questions = []
                        for qa in parsed_data['questions']:
                            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                                questions.append({
                                    'question': str(qa.get('question', '')).strip(),
                                    'answer': str(qa.get('answer', '')).strip()
                                })
                        if questions:
                            return questions
                
                logger.warning(f"Could not extract valid questions from LLM response for document ID {doc_id}")
                logger.debug(f"Full response: {response}")
                return []
                
            except Exception as e:
                logger.error(f"Failed to parse LLM response for document ID {doc_id}: {e}")
                logger.error(f"Full response: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for document ID {doc_id}: {e}")
            logger.error(f"Full response: {response}")
            return []

    def parse_llm_questions_and_concepts_response(self, response: str, doc_id: int) -> Dict[str, Any]:
        """Parse LLM response to extract both questions and concepts using json-repair."""
        class QAItem(BaseModel):
            question: str = Field(min_length=1)
            answer: str = Field(min_length=1)
        class ConceptItem(BaseModel):
            concept: str = Field(min_length=1)
            definition: str = Field(min_length=1)
        class QACSchema(BaseModel):
            questions: List[QAItem]
            concepts: List[ConceptItem]
        try:
            # Clean up the response first
            cleaned_response = self.clean_json_response(response)
            
            # Use json-repair to automatically fix and parse the JSON
            try:
                repaired_json = repair_json(cleaned_response)
                parsed_data = json.loads(repaired_json)
                
                try:
                    validated = QACSchema.model_validate(parsed_data)
                    result = {
                        'questions': [
                            {'question': qa.question.strip(), 'answer': qa.answer.strip()}
                            for qa in validated.questions
                        ],
                        'concepts': [
                            {'concept': c.concept.strip(), 'definition': c.definition.strip()}
                            for c in validated.concepts
                        ],
                    }
                    if result['questions'] or result['concepts']:
                        logger.info(f"Successfully parsed {len(result['questions'])} questions and {len(result['concepts'])} concepts for document ID {doc_id}")
                        return result
                except ValidationError as ve:
                    logger.warning(f"Schema validation failed for doc {doc_id}: {ve}")
                    result = {"questions": [], "concepts": []}
                    if parsed_data and parsed_data.get('questions') and isinstance(parsed_data['questions'], list):
                        for qa in parsed_data['questions']:
                            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                                result['questions'].append({
                                    'question': str(qa.get('question', '')).strip(),
                                    'answer': str(qa.get('answer', '')).strip()
                                })
                    if parsed_data and parsed_data.get('concepts') and isinstance(parsed_data['concepts'], list):
                        for concept in parsed_data['concepts']:
                            if isinstance(concept, dict) and concept.get('concept') and concept.get('definition'):
                                result['concepts'].append({
                                    'concept': str(concept.get('concept', '')).strip(),
                                    'definition': str(concept.get('definition', '')).strip()
                                })
                    if result['questions'] or result['concepts']:
                        return result
                
                logger.warning(f"Could not extract valid questions or concepts from LLM response for document ID {doc_id}")
                logger.debug(f"Full response: {response}")
                return {"questions": [], "concepts": []}
                
            except Exception as e:
                logger.error(f"Failed to parse LLM response for document ID {doc_id}: {e}")
                logger.error(f"Full response: {response}")
                return {"questions": [], "concepts": []}
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for document ID {doc_id}: {e}")
            logger.error(f"Full response: {response}")
            return {"questions": [], "concepts": []}

    def merge_concept_definitions(self, concept_name: str, existing_definition: str, new_definition: str) -> Optional[str]:
        """Merge two concept definitions using LLM to create a single, comprehensive definition."""
        try:
            base_prompt = get_concept_definitions_merger_prompt(concept_name, existing_definition, new_definition)

            # Enhance the prompt with global context 
            content_for_keywords = f"{concept_name}\n{existing_definition}\n{new_definition}"
            enhanced_prompt = self._enhance_prompt_with_context(base_prompt, content_for_keywords)

            response = self._call_llm_with_retry([
                {"role": "user", "content": enhanced_prompt}
            ])
            
            if response:
                merged_definition = response.strip()
                logger.info(f"Successfully merged definitions for concept '{concept_name}'")
                return merged_definition
            else:
                logger.error(f"Failed to get LLM response for merging definitions of concept '{concept_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error merging definitions for concept '{concept_name}': {e}")
            return None

    def merge_question_answers(self, question: str, existing_answer: str, new_answer: str) -> Optional[str]:
        """Merge two answers for the same question using LLM to create a single, comprehensive answer."""
        try:
            base_prompt = get_question_answers_merger_prompt(question, existing_answer, new_answer)

            # Enhance the prompt with global context
            content_for_keywords = f"{question}\n{existing_answer}\n{new_answer}"
            enhanced_prompt = self._enhance_prompt_with_context(base_prompt, content_for_keywords)

            response = self._call_llm_with_retry([
                {"role": "user", "content": enhanced_prompt}
            ])
            
            if response:
                merged_answer = response.strip()
                logger.info(f"Successfully merged answers for question: {question[:100]}...")
                return merged_answer
            else:
                logger.error(f"Failed to get LLM response for merging answers to question: {question[:100]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error merging answers for question '{question[:100]}...': {e}")
            return None



    def get_questions_prompt_for_source_type(self, source_type: str, content: str) -> str:
        """Generate a questions prompt for LLM based on source type."""
        source_descriptions = {
            'metabase_docs': 'Metabase knowledge base article',
            'issues': 'GitHub issue',
            'discourse_posts': 'Discourse forum post/conversation'
        }
        
        specific_instructions = {
            'metabase_docs': 'Please pay attention to the concepts covered in the article to formulate relevant questions. E.g. if there\'s a list of supported databases or supported functions, create one question for each database or supported function like \'Does Metabase support connecting to database x?\' or \'Does Metabase support function y?\'',
            # 'issues': 'Please focus on the problem described, any solutions mentioned, and technical details. Create questions about the issue, its resolution, and any workarounds or fixes discussed.',
            # 'discourse_posts': 'Please focus on the main topic discussed, any solutions provided, and questions asked by users. Create questions about the discussion topic, solutions mentioned, and community responses.'
        }
        
        source_description = source_descriptions.get(source_type, 'document')
        specific_instruction = specific_instructions.get(source_type, '')
        
        base_prompt = f"""Here's a {source_description} with content: {content}

Please generate all the questions that this content answers. For each question, provide a clear and concise answer.

Respond in the following JSON format only:
{{
  "questions": [
    {{
      "question": "What is the question here?",
      "answer": "The answer to the question."
    }},
    {{
      "question": "Another question?",
      "answer": "Another answer."
    }}
  ]
}}

{specific_instruction}"""

        # Enhance the prompt with global context and keywords
        return self._enhance_prompt_with_context(base_prompt, content)

    def get_questions_and_concepts_prompt_for_source_type(self, source_type: str, content: str) -> str:
        """Generate a prompt for questions and concepts extraction for LLM based on source type."""
        source_descriptions = {
            'metabase_docs': 'Metabase knowledge base article',
            'issues': 'GitHub issue',
            'discourse_posts': 'Discourse forum post/conversation'
        }
        
        source_description = source_descriptions.get(source_type, 'document')
        
        base_prompt = f"""Here's a {source_description} with content: {content}

Please perform two tasks:

1. **Extract Core Concepts**: Read the text carefully and extract only the core concepts it introduces. For each concept, provide a concise definition based on how it is used within the context of the text. Focus only on high-level ideas that are essential to understanding the subject. Do not include implementation steps, examples, or tangential details — just the concepts and their meanings.

2. **Generate Questions**: Generate all the questions that this content answers. For each question, provide a clear and concise answer.

Respond in the following JSON format only:
{{
  "concepts": [
    {{
      "concept": "Concept name",
      "definition": "Concise definition of the concept based on the text context."
    }},
    {{
      "concept": "Another concept",
      "definition": "Definition of another concept."
    }}
  ],
  "questions": [
    {{
      "question": "What is the question here?",
      "answer": "The answer to the question."
    }},
    {{
      "question": "Another question?",
      "answer": "Another answer."
    }}
  ]
}}

Important guidelines:
- For concepts: Focus on high-level ideas, technical terms, features, or methodologies that are essential to understanding the subject matter.
- For questions: Create questions that users might ask about the content, covering key information, features, capabilities, and usage scenarios. As an example, if there's a table with supported databases, create questions like "Does Metabase support database x?" or "Does Metabase support database y?"
- Keep definitions concise but informative.
- Ensure both concepts and questions are relevant to the content provided."""

        # Enhance the prompt with global context and keywords
        return self._enhance_prompt_with_context(base_prompt, content)

    def create_questions_for_content(self, source_type: str, content: str, doc_id: int) -> List[Dict[str, str]]:
        """Create questions for a single piece of content."""
        prompt = self.get_questions_prompt_for_source_type(source_type, content)
        
        response = self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])
        
        if response:
            qas = self.parse_llm_questions_response(response, doc_id)
            # Deduplicate within this content by question text (case-insensitive)
            seen = set()
            deduped: List[Dict[str, str]] = []
            for qa in qas:
                key = qa['question'].strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(qa)
            return deduped
        else:
            logger.error(f"Failed to get LLM response for questions generation (doc_id: {doc_id})")
            return []

    def create_questions_and_concepts_for_content(self, source_type: str, content: str, doc_id: int) -> Dict[str, Any]:
        """Create questions and extract concepts for a single piece of content."""
        prompt = self.get_questions_and_concepts_prompt_for_source_type(source_type, content)
        
        response = self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])
        
        if response:
            result = self.parse_llm_questions_and_concepts_response(response, doc_id)
            # Deduplicate questions by text
            seen = set()
            deduped_qs: List[Dict[str, str]] = []
            for qa in result.get('questions', []):
                key = qa['question'].strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    deduped_qs.append(qa)
            result['questions'] = deduped_qs
            return result
        else:
            logger.error(f"Failed to get LLM response for questions and concepts generation (doc_id: {doc_id})")
            return {"questions": [], "concepts": []}

    def analyze_issues_batch(self, issues: List[Tuple[int, str, str, List[str], str]]) -> Dict[int, Dict[str, Any]]:
        """
        Analyzes multiple issues in a single LLM call.
        
        Args:
            issues: List of tuples (issue_id, title, body, labels, state)
            
        Returns:
            Dictionary mapping issue_id to analysis results
        """
        if not issues:
            return {}

        # Build content string from all issues for keyword analysis
        all_content = "\n".join([f"{title}\n{body or ''}" for _, title, body, _, _ in issues])
        
        base_system_prompt = get_batch_issues_analyzer_prompt()

        # Enhance system prompt with global context and keywords
        enhanced_system_prompt = self._enhance_prompt_with_context(base_system_prompt, all_content)

        # Build the user prompt with all issues
        user_prompt_parts = ["Analyze the following GitHub issues:\n"]
        
        for issue_id, title, body, labels, state in issues:
            body_text = body or ""
            user_prompt_parts.append(f"""
--- ISSUE {issue_id} ---
**Title:** {title}
**State:** {state}
**Labels:** {', '.join(labels)}
**Body:**
{body_text}
""")
        
        user_prompt = "\n".join(user_prompt_parts)

        response = self._call_llm_with_retry([
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format={"type": "json_object"})
        
        if not response:
            # Return empty results for all issues in case of error
            return {issue_id: {"summary": None, "reported_version": None, "stack_trace_file": None} 
                   for issue_id, _, _, _, _ in issues}
        
        try:
            batch_results = json.loads(response)
            
            # Parse results back to individual issue analyses
            parsed_results = {}
            for issue_id, _, _, _, _ in issues:
                issue_key = f"issue_{issue_id}"
                if issue_key in batch_results:
                    analysis = batch_results[issue_key]
                    parsed_results[issue_id] = {
                        "summary": analysis.get("summary"),
                        "reported_version": analysis.get("reported_version"),
                        "stack_trace_file": analysis.get("stack_trace_file"),
                    }
                else:
                    # Fallback if LLM didn't return analysis for this issue
                    parsed_results[issue_id] = {
                        "summary": None,
                        "reported_version": None,
                        "stack_trace_file": None,
                    }
            
            return parsed_results

        except Exception as e:
            logger.error(f"Error parsing batch LLM analysis: {e}")
            # Return empty results for all issues in case of error
            return {issue_id: {"summary": None, "reported_version": None, "stack_trace_file": None} 
                   for issue_id, _, _, _, _ in issues}

    def analyze_issue(self, issue_title: str, issue_body: str, labels: list, state: str) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis of a GitHub issue using a single LLM call.
        
        This method is kept for backward compatibility but batch processing is preferred.
        """
        system_prompt = get_single_issue_analyzer_prompt()
        
        body_text = issue_body or ""
        user_prompt = f"""
Analyze the following GitHub issue:

**Title:** {issue_title}

**State:** {state}

**Labels:** {', '.join(labels)}

**Body:**
{body_text}
"""

        response = self._call_llm_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format={"type": "json_object"})
        
        if not response:
            return {
                "summary": None, "reported_version": None,
                "stack_trace_file": None
            }
        
        try:
            analysis_result = json.loads(response)
            
            # Ensure all keys are present, defaulting to None if missing
            return {
                "summary": analysis_result.get("summary"),
                "reported_version": analysis_result.get("reported_version"),
                "stack_trace_file": analysis_result.get("stack_trace_file"),
            }

        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return {
                "summary": None, "reported_version": None,
                "stack_trace_file": None
            }
    
    def summarize_text(self, text: str) -> str:
        """
        Generates a concise summary of the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            A concise summary of the text
        """
        return llm_client.summarize_text(text, model="openai-fast") or ""
    
    def summarize_batch(self, items: List[Tuple[int, str]]) -> Dict[int, str]:
        """
        Generates summaries for multiple texts in a single LLM call.
        
        Args:
            items: List of tuples (item_id, text_content)
            
        Returns:
            Dictionary mapping item_id to summary
        """
        if not items:
            return {}
        
        # Build content string from all items for keyword analysis
        all_content = "\n".join([content for _, content in items])
        
        base_system_prompt = get_batch_content_summarizer_prompt()

        # Enhance system prompt with global context and keywords
        enhanced_system_prompt = self._enhance_prompt_with_context(base_system_prompt, all_content)
        
        # Build the user prompt with all items
        user_prompt_parts = ["Please provide concise summaries for the following content pieces:\n"]
        
        for item_id, content in items:
            user_prompt_parts.append(f"""
--- ID: {item_id} ---
{content}
""")
        
        user_prompt = "\n".join(user_prompt_parts)
        user_prompt += "\n\nProvide summaries in the JSON format specified."

        response = self._call_llm_with_retry([
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format={"type": "json_object"})
        
        if not response:
            # Return empty summaries for all items in case of error
            return {item_id: "" for item_id, _ in items}
        
        try:
            batch_results = json.loads(response)
            
            # Parse results back to individual summaries
            summaries = {}
            if "summaries" in batch_results:
                for item_id, content in items:
                    summary = batch_results["summaries"].get(str(item_id))
                    if summary:
                        summaries[item_id] = summary
                    else:
                        logger.warning(f"No summary returned for ID {item_id}")
                        summaries[item_id] = ""
            
            return summaries

        except Exception as e:
            logger.error(f"Error parsing batch summaries: {e}")
            # Return empty summaries for all items in case of error
            return {item_id: "" for item_id, _ in items}

    def analyze_discourse_conversation(self, conversation: str) -> Dict[str, Any]:
        """
        Analyzes a discourse conversation to extract key information.
        
        Args:
            conversation: The full conversation text from a discourse topic
            
        Returns:
            Dictionary containing:
            - llm_summary: Summary of the conversation
            - type_of_topic: Type of conversation (bug, help, feature_request)
            - solution: Solution if any found in the conversation
            - version: Version mentioned if any
            - reference: URL reference mentioned if any
        """
        system_prompt = get_discourse_conversation_analyzer_prompt()
        user_prompt = get_discourse_conversation_user_prompt(conversation)

        response = self._call_llm_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format={"type": "json_object"})
        
        if not response:
            return {
                "llm_summary": "",
                "type_of_topic": None,
                "solution": None,
                "version": None,
                "reference": None
            }
        
        try:
            # Clean up the response first
            cleaned_response = self.clean_json_response(response)
            
            # Use json-repair to automatically fix and parse the JSON
            try:
                repaired_json = repair_json(cleaned_response)
                parsed_data = json.loads(repaired_json)
                
                return {
                    "llm_summary": parsed_data.get("llm_summary", ""),
                    "type_of_topic": parsed_data.get("type_of_topic"),
                    "solution": parsed_data.get("solution"),
                    "version": parsed_data.get("version"),
                    "reference": parsed_data.get("reference")
                }
                
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error parsing LLM response for discourse conversation: {e}")
                return {
                    "llm_summary": "",
                    "type_of_topic": None,
                    "solution": None,
                    "version": None,
                    "reference": None
                }
                
        except Exception as e:
            logger.error(f"Error processing LLM response for discourse conversation: {e}")
            return {
                "llm_summary": "",
                "type_of_topic": None,
                "solution": None,
                "version": None,
                "reference": None
            } 